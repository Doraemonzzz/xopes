import pytest
import torch
import torch.nn.functional as F

from xopes.ops.lightning_attn.constant_decay import (  # noqa
    lacd_parallel_triton,
    lacd_recurrence_triton,
    lacd_torch,
)
from xopes.utils import get_threshold


def get_params():
    shapes = [
        # standard shape
        (2, 256, 12, 128, 128),
        (2, 1024, 8, 32, 16),
        # BLOCK_N +- 1
        (2, 257, 8, 64, 32),
        (2, 255, 8, 64, 32),
        (2, 65, 7, 33, 63),
        # BLOCK_N +- C
        (2, 270, 8, 64, 32),
        (2, 270, 8, 33, 16),
        (2, 1125, 8, 43, 33),
        # LARGE D, E
        (2, 1125, 8, 255, 257),
        (2, 1025, 8, 255, 257),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_initial_state", [True, False])
@pytest.mark.parametrize("use_log_decay", [True, False])
@pytest.mark.parametrize("use_varlen", [False])
@pytest.mark.parametrize("no_dstate", [True, False])
@pytest.mark.parametrize("use_chunk_loop", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test(
    shape,
    use_initial_state,
    use_log_decay,
    use_varlen,
    no_dstate,
    use_chunk_loop,
    dtype,
):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, n, h, d, e = shape

    if use_varlen:
        b = 1
        m = n // 5
        cu_seqlens = torch.tensor(
            [0, m - 2, 2 * m + 1, 3 * m - 1, 4 * m, n], dtype=torch.long, device=device
        )
    else:
        cu_seqlens = None

    # Generate input tensors
    q = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
    if use_log_decay:
        ld = F.logsigmoid(torch.randn(h, dtype=dtype, device=device)).requires_grad_()
    else:
        ld = None

    if no_dstate:
        do = torch.randn((b, n, h, e), dtype=dtype, device=device)
    else:
        do = torch.randn((), dtype=dtype, device=device)

    if use_initial_state:
        initial_state = torch.randn(
            (b, h, d, e), dtype=dtype, device=device
        ).requires_grad_()
    else:
        initial_state = None

    ##### Forward pass
    # baseline
    o_torch, s_torch = lacd_torch(
        q=q,
        k=k,
        v=v,
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )
    if no_dstate:
        output_torch = o_torch
    else:
        output_torch = o_torch.mean() + s_torch.mean()

    # triton recurrence
    o_triton, s_triton = lacd_recurrence_triton(
        q=q,
        k=k,
        v=v,
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )
    if no_dstate:
        output_triton = o_triton
    else:
        output_triton = o_triton.mean() + s_triton.mean()

    # triton parallel
    o_parallel_triton, s_parallel_triton = lacd_parallel_triton(
        q=q, k=k, v=v, ld=ld, initial_state=initial_state, use_chunk_loop=use_chunk_loop
    )
    if no_dstate:
        output_parallel_triton = o_parallel_triton
    else:
        output_parallel_triton = o_parallel_triton.mean() + s_parallel_triton.mean()

    ##### Backward pass
    # baseline
    output_torch.backward(do, retain_graph=True)
    dq_torch, q.grad = q.grad.clone(), None
    dk_torch, k.grad = k.grad.clone(), None
    dv_torch, v.grad = v.grad.clone(), None
    if use_log_decay:
        ld_torch, ld.grad = ld.grad.clone(), None
    else:
        ld_torch = None
    if use_initial_state:
        ds_torch, initial_state.grad = initial_state.grad.clone(), None

    # triton recurrence
    output_triton.backward(do, retain_graph=True)
    dq_triton, q.grad = q.grad.clone(), None
    dk_triton, k.grad = k.grad.clone(), None
    dv_triton, v.grad = v.grad.clone(), None
    if use_log_decay:
        ld_triton, ld.grad = ld.grad.clone(), None
    else:
        ld_triton = None
    if use_initial_state:
        ds_triton, initial_state.grad = initial_state.grad.clone(), None

    # triton parallel
    output_parallel_triton.backward(do, retain_graph=True)
    dq_parallel_triton, q.grad = q.grad.clone(), None
    dk_parallel_triton, k.grad = k.grad.clone(), None
    dv_parallel_triton, v.grad = v.grad.clone(), None
    if use_log_decay:
        ld_parallel_triton, ld.grad = ld.grad.clone(), None
    else:
        ld_parallel_triton = None
    if use_initial_state:
        ds_parallel_triton, initial_state.grad = initial_state.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    ##### Check forward pass results
    # triton recurrence
    print(
        "o diff max (torch recurrence Vs triton recurrence): ",
        torch.abs(o_torch - o_triton).max().item(),
    )
    print(
        "o diff norm (torch recurrence Vs triton recurrence): ",
        torch.norm(o_torch - o_triton).item(),
    )
    assert torch.allclose(o_torch, o_triton, atol=atol, rtol=rtol)

    print(
        "state diff max (torch recurrence Vs triton recurrence): ",
        torch.abs(s_torch - s_triton).max().item(),
    )
    print(
        "state diff norm (torch recurrence Vs triton recurrence): ",
        torch.norm(s_torch - s_triton).item(),
    )
    assert torch.allclose(s_torch, s_triton, atol=atol, rtol=rtol)

    # triton parallel
    print(
        "o diff max (torch parallel Vs triton parallel): ",
        torch.abs(o_torch - o_parallel_triton).max().item(),
    )
    print(
        "o diff norm (torch parallel Vs triton parallel): ",
        torch.norm(o_torch - o_parallel_triton).item(),
    )
    assert torch.allclose(o_torch, o_parallel_triton, atol=atol, rtol=rtol)

    print(
        "state diff max (torch recurrence Vs triton parallel): ",
        torch.abs(s_torch - s_parallel_triton).max().item(),
    )
    print(
        "state diff norm (torch recurrence Vs triton parallel): ",
        torch.norm(s_torch - s_parallel_triton).item(),
    )
    assert torch.allclose(s_torch, s_parallel_triton, atol=atol, rtol=rtol)

    ##### Check backward pass results
    # triton recurrence
    print(
        "dq diff max (Vs triton recurrence): ",
        torch.abs(dq_torch - dq_triton).max().item(),
    )
    print(
        "dq diff norm (Vs triton recurrence): ", torch.norm(dq_torch - dq_triton).item()
    )
    assert torch.allclose(dq_torch, dq_triton, atol=atol, rtol=rtol)

    print(
        "dk diff max (Vs triton recurrence): ",
        torch.abs(dk_torch - dk_triton).max().item(),
    )
    print(
        "dk diff norm (Vs triton recurrence): ", torch.norm(dk_torch - dk_triton).item()
    )
    assert torch.allclose(dk_torch, dk_triton, atol=atol, rtol=rtol)

    print(
        "dv diff max (Vs triton recurrence): ",
        torch.abs(dv_torch - dv_triton).max().item(),
    )
    print(
        "dv diff norm (Vs triton recurrence): ", torch.norm(dv_torch - dv_triton).item()
    )
    assert torch.allclose(dv_torch, dv_triton, atol=atol, rtol=rtol)

    if use_initial_state:
        print(
            "ds diff max (Vs triton recurrence): ",
            torch.abs(ds_torch - ds_triton).max().item(),
        )
        print(
            "ds diff norm (Vs triton recurrence): ",
            torch.norm(ds_torch - ds_triton).item(),
        )
        assert torch.allclose(ds_torch, ds_triton, atol=atol, rtol=rtol)

    # triton parallel
    print(
        "dq diff max (torch parallel Vs triton parallel): ",
        torch.abs(dq_torch - dq_parallel_triton).max().item(),
    )
    print(
        "dq diff norm (torch parallel Vs triton parallel): ",
        torch.norm(dq_torch - dq_parallel_triton).item(),
    )

    c = 16
    m = (n + c - 1) // c
    for i in range(m):
        start = i * c
        end = min(start + c, n)
        print(
            i,
            torch.norm((dq_torch - dq_parallel_triton)[:, start:end, :, :]).item(),
        )
    assert torch.allclose(dq_torch, dq_parallel_triton, atol=atol, rtol=rtol)

    print(
        "dk diff max (torch parallel Vs triton parallel): ",
        torch.abs(dk_torch - dk_parallel_triton).max().item(),
    )
    print(
        "dk diff norm (torch parallel Vs triton parallel): ",
        torch.norm(dk_torch - dk_parallel_triton).item(),
    )

    print("Start print dk diff")
    for i in range(m):
        start = i * c
        end = min(start + c, n)
        print(
            i,
            torch.norm((dk_torch - dk_parallel_triton)[:, start:end, :, :]).item(),
        )

    print(
        "dv diff max (torch parallel Vs triton parallel): ",
        torch.abs(dv_torch - dv_parallel_triton).max().item(),
    )
    print(
        "dv diff norm (torch parallel Vs triton parallel): ",
        torch.norm(dv_torch - dv_parallel_triton).item(),
    )
    print("Start print dv diff")
    for i in range(m):
        start = i * c
        end = min(start + c, n)
        print(
            i,
            torch.norm((dv_torch - dv_parallel_triton)[:, start:end, :, :]).item(),
        )

    assert torch.allclose(dk_torch, dk_parallel_triton, atol=atol, rtol=rtol)
    assert torch.allclose(dv_torch, dv_parallel_triton, atol=atol, rtol=rtol)

    if use_initial_state:
        print(
            "ds diff max (torch parallel Vs triton parallel): ",
            torch.abs(ds_torch - ds_parallel_triton).max().item(),
        )
        print(
            "ds diff norm (torch parallel Vs triton parallel): ",
            torch.norm(ds_torch - ds_parallel_triton).item(),
        )
        assert torch.allclose(ds_torch, ds_parallel_triton, atol=atol, rtol=rtol)

    if use_log_decay:
        print(
            "ld diff max (torch parallel Vs triton recurrence): ",
            torch.abs(ld_torch - ld_triton).max().item(),
        )
        print(
            "ld diff norm (torch parallel Vs triton recurrence): ",
            torch.norm(ld_torch - ld_triton).item(),
        )
        assert torch.allclose(ld_torch, ld_triton, atol=atol, rtol=rtol)

        print(
            "ld diff max (torch parallel Vs triton parallel): ",
            torch.abs(ld_torch - ld_parallel_triton).max().item(),
        )
        print(
            "ld diff norm (torch parallel Vs triton parallel): ",
            torch.norm(ld_torch - ld_parallel_triton).item(),
        )
        assert torch.allclose(ld_torch, ld_parallel_triton, atol=atol, rtol=rtol)
