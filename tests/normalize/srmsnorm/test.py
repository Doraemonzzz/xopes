import pytest
import torch

from xopes.ops.normalize import srmsnorm_torch, srmsnorm_triton
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

    if use_residual:
        residual = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    else:
        residual = None

    # forward
    o_srmsnorm_torch = srmsnorm_torch(
        x,
        dim=d,
        eps=eps,
        residual=residual,
    )

    o_srmsnorm_triton = srmsnorm_triton(
        x,
        dim=d,
        eps=eps,
        residual=residual,
    )

    # backward
    o_srmsnorm_torch.backward(do, retain_graph=True)
    dx_srmsnorm_torch, x.grad = x.grad.clone(), None

    if use_residual:
        dr_srmsnorm_torch, residual.grad = residual.grad.clone(), None
    else:
        dr_srmsnorm_torch = None

    o_srmsnorm_triton.backward(do, retain_graph=True)
    dx_srmsnorm_triton, x.grad = x.grad.clone(), None

    if use_residual:
        dr_srmsnorm_triton, residual.grad = residual.grad.clone(), None
    else:
        dr_srmsnorm_triton = None

    atol, rtol = get_threshold(dtype)

    ##### fwd
    print("o diff max: ", torch.abs(o_srmsnorm_torch - o_srmsnorm_triton).max().item())
    print("o diff norm: ", torch.norm(o_srmsnorm_torch - o_srmsnorm_triton).item())
    assert torch.allclose(o_srmsnorm_torch, o_srmsnorm_triton, atol=atol, rtol=rtol)

    ##### bwd
    print(
        "dx diff max: ",
        torch.abs(dx_srmsnorm_torch - dx_srmsnorm_triton).max().item(),
    )
    print("dx diff norm: ", torch.norm(dx_srmsnorm_torch - dx_srmsnorm_triton).item())
    assert torch.allclose(dx_srmsnorm_torch, dx_srmsnorm_triton, atol=atol, rtol=rtol)

    if use_residual:
        print(
            "dr diff max: ",
            torch.abs(dr_srmsnorm_torch - dr_srmsnorm_triton).max().item(),
        )
        print(
            "dr diff norm: ",
            torch.norm(dr_srmsnorm_torch - dr_srmsnorm_triton).item(),
        )
        assert torch.allclose(
            dr_srmsnorm_torch, dr_srmsnorm_triton, atol=atol, rtol=rtol
        )
