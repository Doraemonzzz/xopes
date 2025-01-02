import pytest
import torch

from xopes.ops.normalize import rmsnorm_torch, rmsnorm_triton
from xopes.utils import get_threshold


def get_params():
    shape = [(6, 128), (4, 8, 256)]

    return shape


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_residual", [True, False])
@pytest.mark.parametrize("c", [1, 16])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(shape, use_residual, c, eps, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    d = shape[-1]
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    do = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((d,), dtype=dtype, device=device).requires_grad_()

    if use_residual:
        residual = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    else:
        residual = None

    # forward
    o_rmsnorm_torch, o_update_residual_torch = rmsnorm_torch(
        x,
        weight=weight,
        dim=d,
        eps=eps,
        residual=residual,
    )
    if use_residual:
        o_rmsnorm_torch = o_rmsnorm_torch + o_update_residual_torch

    o_rmsnorm_triton, o_update_residual_triton = rmsnorm_triton(
        x,
        weight=weight,
        dim=d,
        eps=eps,
        residual=residual,
    )
    if use_residual:
        o_rmsnorm_triton = o_rmsnorm_triton + o_update_residual_triton

    # backward
    o_rmsnorm_torch.backward(do, retain_graph=True)
    dx_rmsnorm_torch, x.grad = x.grad.clone(), None
    dw_rmsnorm_torch, weight.grad = weight.grad.clone(), None

    if use_residual:
        dr_rmsnorm_torch, residual.grad = residual.grad.clone(), None
    else:
        dr_rmsnorm_torch = None

    o_rmsnorm_triton.backward(do, retain_graph=True)
    dx_rmsnorm_triton, x.grad = x.grad.clone(), None
    dw_rmsnorm_triton, weight.grad = weight.grad.clone(), None

    if use_residual:
        dr_rmsnorm_triton, residual.grad = residual.grad.clone(), None
    else:
        dr_rmsnorm_triton = None

    atol, rtol = get_threshold(dtype)

    ##### fwd
    print("o diff max: ", torch.abs(o_rmsnorm_torch - o_rmsnorm_triton).max().item())
    print("o diff norm: ", torch.norm(o_rmsnorm_torch - o_rmsnorm_triton).item())
    assert torch.allclose(o_rmsnorm_torch, o_rmsnorm_triton, atol=atol, rtol=rtol)

    ##### bwd
    print(
        "dx diff max: ",
        torch.abs(dx_rmsnorm_torch - dx_rmsnorm_triton).max().item(),
    )
    print("dx diff norm: ", torch.norm(dx_rmsnorm_torch - dx_rmsnorm_triton).item())
    assert torch.allclose(dx_rmsnorm_torch, dx_rmsnorm_triton, atol=atol, rtol=rtol)

    print(
        "dw diff max: ",
        torch.abs(dw_rmsnorm_torch - dw_rmsnorm_triton).max().item(),
    )
    print(
        "dw diff norm: ",
        torch.norm(dw_rmsnorm_torch - dw_rmsnorm_triton).item(),
    )
    assert torch.allclose(dw_rmsnorm_torch, dw_rmsnorm_triton, atol=atol, rtol=rtol)

    if use_residual:
        print(
            "dr diff max: ",
            torch.abs(dr_rmsnorm_torch - dr_rmsnorm_triton).max().item(),
        )
        print(
            "dr diff norm: ",
            torch.norm(dr_rmsnorm_torch - dr_rmsnorm_triton).item(),
        )
        assert torch.allclose(dr_rmsnorm_torch, dr_rmsnorm_triton, atol=atol, rtol=rtol)
