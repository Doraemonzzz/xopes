import pytest
import torch
import torch.nn.functional as F

from xopes.ops.lightning_attn.vector_decay import lavd_chunk_torch, lavd_torch
from xopes.utils import get_threshold


def get_params():
    shapes = [
        (2, 128, 8, 64, 32),
        (4, 256, 12, 128, 64),
        # (2, 1024, 16, 128, 128),
        (2, 32, 8, 64, 32),
        # (1, 4, 1, 2, 2),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
# @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])


@pytest.mark.parametrize("use_k", [True, False])
@pytest.mark.parametrize("use_v", [True, False])
@pytest.mark.parametrize("use_state", [True, False])
@pytest.mark.parametrize("use_zero_ld", [True, False])


# @pytest.mark.parametrize("use_k", [True])
# @pytest.mark.parametrize("use_v", [True])
# @pytest.mark.parametrize("use_state", [False])
# @pytest.mark.parametrize("use_zero_ld", [True])
@pytest.mark.parametrize("dtype", [torch.float32])
def test(shape, use_k, use_v, use_state, use_zero_ld, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, n, h, d, e = shape
    chunk_size = n // 2
    # chunk_size = n

    # Generate input tensors
    q = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    if use_zero_ld:
        ldk = (torch.zeros((b, n, h, d), dtype=dtype, device=device)).requires_grad_()
        ldv = (torch.zeros((b, n, h, e), dtype=dtype, device=device)).requires_grad_()
    else:
        ldk = F.logsigmoid(
            torch.randn((b, n, h, d), dtype=dtype, device=device)
        ).requires_grad_()
        ldv = F.logsigmoid(
            torch.randn((b, n, h, e), dtype=dtype, device=device)
        ).requires_grad_()

    if use_k:
        k = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    else:
        k = None

    if use_v:
        v = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
    else:
        v = None

    if use_state:
        state = torch.randn((b, h, d, e), dtype=dtype, device=device).requires_grad_()
    else:
        state = None

    do = torch.randn((), dtype=dtype, device=device)

    # Forward pass
    o_torch, s_torch = lavd_torch(
        q=q,
        ldk=ldk,
        ldv=ldv,
        k=k,
        v=v,
        state=state,
    )
    output_torch = o_torch.sum() + s_torch.sum()
    o_chunk, s_chunk = lavd_chunk_torch(
        q=q,
        ldk=ldk,
        ldv=ldv,
        k=k,
        v=v,
        state=state,
        chunk_size=chunk_size,
    )
    output_chunk = o_chunk.sum() + s_chunk.sum()

    # Backward pass
    output_torch.backward(do, retain_graph=True)
    # o_torch.backward(do, retain_graph=True)
    dq_torch, q.grad = q.grad.clone(), None
    dldk_torch, ldk.grad = ldk.grad.clone(), None
    dldv_torch, ldv.grad = ldv.grad.clone(), None
    if use_k:
        dk_torch, k.grad = k.grad.clone(), None
    if use_v:
        dv_torch, v.grad = v.grad.clone(), None
    if use_state:
        ds_torch, state.grad = state.grad.clone(), None

    output_chunk.backward(do, retain_graph=True)
    # o_chunk.backward(do, retain_graph=True)
    dq_chunk, q.grad = q.grad.clone(), None
    dldk_chunk, ldk.grad = ldk.grad.clone(), None
    dldv_chunk, ldv.grad = ldv.grad.clone(), None
    if use_k:
        dk_chunk, k.grad = k.grad.clone(), None
    if use_v:
        dv_chunk, v.grad = v.grad.clone(), None
    if use_state:
        ds_chunk, state.grad = state.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    # Check forward pass results
    print("o diff max: ", torch.abs(o_torch - o_chunk).max().item())
    print("o diff norm: ", torch.norm(o_torch - o_chunk).item())
    assert torch.allclose(o_torch, o_chunk, atol=atol, rtol=rtol)

    print("s diff max: ", torch.abs(s_torch - s_chunk).max().item())
    print("s diff norm: ", torch.norm(s_torch - s_chunk).item())
    assert torch.allclose(s_torch, s_chunk, atol=atol, rtol=rtol)

    # Check backward pass results
    print("dq diff max: ", torch.abs(dq_torch - dq_chunk).max().item())
    print("dq diff norm: ", torch.norm(dq_torch - dq_chunk).item())
    assert torch.allclose(dq_torch, dq_chunk, atol=atol, rtol=rtol)

    print("dldk diff max: ", torch.abs(dldk_torch - dldk_chunk).max().item())
    print("dldk diff norm: ", torch.norm(dldk_torch - dldk_chunk).item())
    assert torch.allclose(dldk_torch, dldk_chunk, atol=atol, rtol=rtol)

    if use_k:
        print("dk diff max: ", torch.abs(dk_torch - dk_chunk).max().item())
        print("dk diff norm: ", torch.norm(dk_torch - dk_chunk).item())
        assert torch.allclose(dk_torch, dk_chunk, atol=atol, rtol=rtol)

    if use_v:
        print("dv diff max: ", torch.abs(dv_torch - dv_chunk).max().item())
        print("dv diff norm: ", torch.norm(dv_torch - dv_chunk).item())
        assert torch.allclose(dv_torch, dv_chunk, atol=atol, rtol=rtol)

    if use_state:
        print("ds diff max: ", torch.abs(ds_torch - ds_chunk).max().item())
        print("ds diff norm: ", torch.norm(ds_torch - ds_chunk).item())
        assert torch.allclose(ds_torch, ds_chunk, atol=atol, rtol=rtol)
