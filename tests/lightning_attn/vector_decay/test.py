import math

import pytest
import torch
import torch.nn.functional as F

from xopes.ops.lightning_attn.vector_decay import (
    lavd_chunk_parallel_torch,
    lavd_chunk_torch,
    lavd_torch,
)
from xopes.utils import get_threshold


def get_params():
    shapes = [
        (2, 128, 8, 64, 32),
        (4, 256, 12, 128, 64),
        (2, 1023, 16, 128, 64),
        (2, 63, 16, 128, 64),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("share_k", [True, False])
@pytest.mark.parametrize("share_v", [True, False])
@pytest.mark.parametrize("use_initial_state", [True, False])
@pytest.mark.parametrize("use_zero_ld", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test(shape, share_k, share_v, use_initial_state, use_zero_ld, dtype):
    use_ldk = True
    use_ldv = True
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, n, h, d, e = shape
    test_chunk = n <= 128
    chunk_size = int(2 ** (int(math.log2(n)) - 1))

    # Generate input tensors
    q = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()

    if share_k:
        k = F.sigmoid(
            torch.randn((b, n, h, d), dtype=dtype, device=device)
        ).requires_grad_()
        ldk = None
    else:
        k = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
        if use_zero_ld:
            ldk = torch.zeros((b, n, h, d), dtype=dtype, device=device).requires_grad_()
        else:
            ldk = F.logsigmoid(
                torch.randn((b, n, h, d), dtype=dtype, device=device)
            ).requires_grad_()

    if share_v:
        v = F.sigmoid(
            torch.randn((b, n, h, e), dtype=dtype, device=device)
        ).requires_grad_()
        ldv = None
    else:
        v = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
        if use_zero_ld:
            ldv = torch.zeros((b, n, h, e), dtype=dtype, device=device).requires_grad_()
        else:
            ldv = F.logsigmoid(
                torch.randn((b, n, h, e), dtype=dtype, device=device)
            ).requires_grad_()

    if use_initial_state:
        initial_state = torch.randn(
            (b, h, d, e), dtype=dtype, device=device
        ).requires_grad_()
    else:
        initial_state = None

    do = torch.randn((), dtype=dtype, device=device)

    ##### Forward pass
    # baseline
    o_torch, s_torch = lavd_torch(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        initial_state=initial_state,
    )
    output_torch = o_torch.sum() + s_torch.sum()

    if test_chunk:
        # chunk torch
        o_chunk, s_chunk = lavd_chunk_torch(
            q=q,
            k=k,
            v=v,
            ldk=ldk,
            ldv=ldv,
            use_ldk=use_ldk,
            use_ldv=use_ldv,
            initial_state=initial_state,
            chunk_size=chunk_size,
        )
        output_chunk = o_chunk.sum() + s_chunk.sum()

    # chunk parallel torch
    o_chunk_parallel, s_chunk_parallel = lavd_chunk_parallel_torch(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )
    output_chunk_parallel = o_chunk_parallel.sum() + s_chunk_parallel.sum()

    ##### Backward pass
    # baseline
    output_torch.backward(do, retain_graph=True)
    dq_torch, q.grad = q.grad.clone(), None
    dk_torch, k.grad = k.grad.clone(), None
    dv_torch, v.grad = v.grad.clone(), None
    if not share_k:
        dldk_torch, ldk.grad = ldk.grad.clone(), None
    if not share_v:
        dldv_torch, ldv.grad = ldv.grad.clone(), None
    if use_initial_state:
        ds_torch, initial_state.grad = initial_state.grad.clone(), None

    if test_chunk:
        # chunk torch
        output_chunk.backward(do, retain_graph=True)
        dq_chunk, q.grad = q.grad.clone(), None
        dk_chunk, k.grad = k.grad.clone(), None
        dv_chunk, v.grad = v.grad.clone(), None
        if not share_k:
            dldk_chunk, ldk.grad = ldk.grad.clone(), None
        if not share_v:
            dldv_chunk, ldv.grad = ldv.grad.clone(), None
        if use_initial_state:
            ds_chunk, initial_state.grad = initial_state.grad.clone(), None

    # chunk parallel torch
    output_chunk_parallel.backward(do, retain_graph=True)
    dq_chunk_parallel, q.grad = q.grad.clone(), None
    dk_chunk_parallel, k.grad = k.grad.clone(), None
    dv_chunk_parallel, v.grad = v.grad.clone(), None
    if not share_k:
        dldk_chunk_parallel, ldk.grad = ldk.grad.clone(), None
    if not share_v:
        dldv_chunk_parallel, ldv.grad = ldv.grad.clone(), None
    if use_initial_state:
        ds_chunk_parallel, initial_state.grad = initial_state.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    ##### Check forward pass results
    if test_chunk:
        # chunk torch
        print(
            "o diff max (Vs chunk torch): ", torch.abs(o_torch - o_chunk).max().item()
        )
        print("o diff norm (Vs chunk torch): ", torch.norm(o_torch - o_chunk).item())
        assert torch.allclose(o_torch, o_chunk, atol=atol, rtol=rtol)

        print(
            "s diff max (Vs chunk torch): ", torch.abs(s_torch - s_chunk).max().item()
        )
        print("s diff norm (Vs chunk torch): ", torch.norm(s_torch - s_chunk).item())
        assert torch.allclose(s_torch, s_chunk, atol=atol, rtol=rtol)

    # chunk parallel torch
    print(
        "o diff max (Vs chunk parallel torch): ",
        torch.abs(o_torch - o_chunk_parallel).max().item(),
    )
    print(
        "o diff norm (Vs chunk parallel torch): ",
        torch.norm(o_torch - o_chunk_parallel).item(),
    )
    assert torch.allclose(o_torch, o_chunk_parallel, atol=atol, rtol=rtol)

    print(
        "s diff max (Vs chunk parallel torch): ",
        torch.abs(s_torch - s_chunk_parallel).max().item(),
    )
    print(
        "s diff norm (Vs chunk parallel torch): ",
        torch.norm(s_torch - s_chunk_parallel).item(),
    )
    assert torch.allclose(s_torch, s_chunk_parallel, atol=atol, rtol=rtol)

    if test_chunk:
        ##### Check backward pass results
        # chunk torch
        print(
            "dq diff max (Vs chunk torch): ",
            torch.abs(dq_torch - dq_chunk).max().item(),
        )
        print("dq diff norm (Vs chunk torch): ", torch.norm(dq_torch - dq_chunk).item())

        print(
            "dk diff max (Vs chunk torch): ",
            torch.abs(dk_torch - dk_chunk).max().item(),
        )
        print("dk diff norm (Vs chunk torch): ", torch.norm(dk_torch - dk_chunk).item())

        print(
            "dv diff max (Vs chunk torch): ",
            torch.abs(dv_torch - dv_chunk).max().item(),
        )
        print("dv diff norm (Vs chunk torch): ", torch.norm(dv_torch - dv_chunk).item())

        if not share_k:
            print(
                "dldk diff max (Vs chunk torch): ",
                torch.abs(dldk_torch - dldk_chunk).max().item(),
            )
            print(
                "dldk diff norm (Vs chunk torch): ",
                torch.norm(dldk_torch - dldk_chunk).item(),
            )

        if not share_v:
            print(
                "dldv diff max (Vs chunk torch): ",
                torch.abs(dldv_torch - dldv_chunk).max().item(),
            )
            print(
                "dldv diff norm (Vs chunk torch): ",
                torch.norm(dldv_torch - dldv_chunk).item(),
            )

        if use_initial_state:
            print(
                "ds diff max (Vs chunk torch): ",
                torch.abs(ds_torch - ds_chunk).max().item(),
            )
            print(
                "ds diff norm (Vs chunk torch): ",
                torch.norm(ds_torch - ds_chunk).item(),
            )

    # chunk parallel torch
    print(
        "dq diff max (Vs chunk parallel torch): ",
        torch.abs(dq_torch - dq_chunk_parallel).max().item(),
    )
    print(
        "dq diff norm (Vs chunk parallel torch): ",
        torch.norm(dq_torch - dq_chunk_parallel).item(),
    )
    assert torch.allclose(dq_torch, dq_chunk_parallel, atol=atol, rtol=rtol)

    print(
        "dk diff max (Vs chunk parallel torch): ",
        torch.abs(dk_torch - dk_chunk_parallel).max().item(),
    )
    print(
        "dk diff norm (Vs chunk parallel torch): ",
        torch.norm(dk_torch - dk_chunk_parallel).item(),
    )
    assert torch.allclose(dk_torch, dk_chunk_parallel, atol=atol, rtol=rtol)

    print(
        "dv diff max (Vs chunk parallel torch): ",
        torch.abs(dv_torch - dv_chunk_parallel).max().item(),
    )
    print(
        "dv diff norm (Vs chunk parallel torch): ",
        torch.norm(dv_torch - dv_chunk_parallel).item(),
    )
    assert torch.allclose(dv_torch, dv_chunk_parallel, atol=atol, rtol=rtol)

    if not share_k:
        print(
            "dldk diff max (Vs chunk parallel torch): ",
            torch.abs(dldk_torch - dldk_chunk_parallel).max().item(),
        )
        print(
            "dldk diff norm (Vs chunk parallel torch): ",
            torch.norm(dldk_torch - dldk_chunk_parallel).item(),
        )
        assert torch.allclose(dldk_torch, dldk_chunk_parallel, atol=atol, rtol=rtol)

    if not share_v:
        print(
            "dldv diff max (Vs chunk parallel torch): ",
            torch.abs(dldv_torch - dldv_chunk_parallel).max().item(),
        )
        print(
            "dldv diff norm (Vs chunk parallel torch): ",
            torch.norm(dldv_torch - dldv_chunk_parallel).item(),
        )
        assert torch.allclose(dldv_torch, dldv_chunk_parallel, atol=atol, rtol=rtol)

    if use_initial_state:
        print(
            "ds diff max (Vs chunk parallel torch): ",
            torch.abs(ds_torch - ds_chunk_parallel).max().item(),
        )
        print(
            "ds diff norm (Vs chunk parallel torch): ",
            torch.norm(ds_torch - ds_chunk_parallel).item(),
        )
        assert torch.allclose(ds_torch, ds_chunk_parallel, atol=atol, rtol=rtol)
