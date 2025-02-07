import pytest
import torch

from xopes.ops.normalize import srms_norm_torch, srms_norm_triton
from xopes.utils import get_threshold


def get_params():
    shape = [(6, 128), (4, 8, 256), (6, 2048, 768)]

    return shape


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_residual", [True, False])
@pytest.mark.parametrize("return_residual", [True, False])
@pytest.mark.parametrize("c", [1, 16])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(shape, use_residual, return_residual, c, eps, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    d = shape[-1]
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    do = torch.randn(shape, dtype=dtype, device=device)

    if use_residual:
        residual = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    else:
        residual = None

    # forward
    o_torch = srms_norm_torch(
        x,
        dim=d,
        eps=eps,
        residual=residual,
        return_residual=return_residual,
    )

    if isinstance(o_torch, tuple):
        o_srms_norm_torch, o_update_residual_torch = o_torch
    else:
        o_srms_norm_torch = o_torch
        o_update_residual_torch = None

    if use_residual and return_residual:
        o_srms_norm_torch = o_srms_norm_torch + o_update_residual_torch

    o_triton = srms_norm_triton(
        x,
        dim=d,
        eps=eps,
        residual=residual,
        return_residual=return_residual,
    )

    if isinstance(o_triton, tuple):
        o_srms_norm_triton, o_update_residual_triton = o_triton
    else:
        o_srms_norm_triton = o_triton
        o_update_residual_triton = None

    if use_residual and return_residual:
        o_srms_norm_triton = o_srms_norm_triton + o_update_residual_triton

    # backward
    o_srms_norm_torch.backward(do, retain_graph=True)
    dx_srms_norm_torch, x.grad = x.grad.clone(), None

    if use_residual:
        dr_srms_norm_torch, residual.grad = residual.grad.clone(), None
    else:
        dr_srms_norm_torch = None

    o_srms_norm_triton.backward(do, retain_graph=True)
    dx_srms_norm_triton, x.grad = x.grad.clone(), None

    if use_residual:
        dr_srms_norm_triton, residual.grad = residual.grad.clone(), None
    else:
        dr_srms_norm_triton = None

    atol, rtol = get_threshold(dtype)

    ##### fwd
    print(
        "o diff max: ", torch.abs(o_srms_norm_torch - o_srms_norm_triton).max().item()
    )
    print("o diff norm: ", torch.norm(o_srms_norm_torch - o_srms_norm_triton).item())
    assert torch.allclose(o_srms_norm_torch, o_srms_norm_triton, atol=atol, rtol=rtol)

    ##### bwd
    print(
        "dx diff max: ",
        torch.abs(dx_srms_norm_torch - dx_srms_norm_triton).max().item(),
    )
    print("dx diff norm: ", torch.norm(dx_srms_norm_torch - dx_srms_norm_triton).item())
    assert torch.allclose(dx_srms_norm_torch, dx_srms_norm_triton, atol=atol, rtol=rtol)

    if use_residual:
        print(
            "dr diff max: ",
            torch.abs(dr_srms_norm_torch - dr_srms_norm_triton).max().item(),
        )
        print(
            "dr diff norm: ",
            torch.norm(dr_srms_norm_torch - dr_srms_norm_triton).item(),
        )
        assert torch.allclose(
            dr_srms_norm_torch, dr_srms_norm_triton, atol=atol, rtol=rtol
        )
