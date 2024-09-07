import pytest
import torch

from xopes.ops import flao_non_causal_torch, lao_non_causal_torch
from xopes.utils import get_threshold


def get_params():
    shape = [
        (6, 8, 512, 256, 128, 64),
    ]

    return shape


@pytest.mark.parametrize("b, h, n, m, d, e", get_params())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(b, h, n, m, d, e, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")

    q = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((b, h, m, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, h, m, e), dtype=dtype, device=device).requires_grad_()
    g = torch.randn((b, h, n, e), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((b, h, n, e), dtype=dtype, device=device)

    # forward
    o_lao_torch = lao_non_causal_torch(q, k, v, g)
    o_flao_torch = flao_non_causal_torch(q, k, v, g)

    # backward
    o_lao_torch.backward(do, retain_graph=True)
    dq_lao_torch, q.grad = q.grad.clone(), None
    dk_lao_torch, k.grad = k.grad.clone(), None
    dv_lao_torch, v.grad = v.grad.clone(), None
    dg_lao_torch, g.grad = g.grad.clone(), None

    o_flao_torch.backward(do, retain_graph=True)
    dq_flao_torch, q.grad = q.grad.clone(), None
    dk_flao_torch, k.grad = k.grad.clone(), None
    dv_flao_torch, v.grad = v.grad.clone(), None
    dg_flao_torch, g.grad = g.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    # forward
    assert torch.allclose(
        o_lao_torch, o_flao_torch, atol=atol, rtol=rtol
    ), f"o diff: {torch.abs(o_lao_torch - o_flao_torch).max().item()}"

    # backward
    assert torch.allclose(
        dq_lao_torch, dq_flao_torch, atol=atol, rtol=rtol
    ), f"dq diff: {torch.abs(dq_lao_torch - dq_flao_torch).max().item()}"
    assert torch.allclose(
        dk_lao_torch, dk_flao_torch, atol=atol, rtol=rtol
    ), f"dk diff: {torch.abs(dk_lao_torch - dk_flao_torch).max().item()}"
    assert torch.allclose(
        dv_lao_torch, dv_flao_torch, atol=atol, rtol=rtol
    ), f"dv diff: {torch.abs(dv_lao_torch - dv_flao_torch).max().item()}"
    assert torch.allclose(
        dg_lao_torch, dg_flao_torch, atol=atol, rtol=rtol
    ), f"dg diff: {torch.abs(dg_lao_torch - dg_flao_torch).max().item()}"
