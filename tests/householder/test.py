import pytest
import torch

from xopes.ops.householder import householder_torch, householder_triton
from xopes.utils import get_threshold


def get_params():
    shapes = [(6, 128), (4, 8, 256), (4, 1024, 1024), (4, 1024, 4096)]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("by", [-1, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(shape, by, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")

    # Generate input tensors
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    if by == -1:
        y = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    else:
        y = torch.randn((shape[-1]), dtype=dtype, device=device).requires_grad_()
    do = torch.randn(shape, dtype=dtype, device=device)

    # forward
    o_householder_torch = householder_torch(x, y)
    o_householder_triton = householder_triton(x, y)

    # backward
    o_householder_torch.backward(do, retain_graph=True)
    dx_householder_torch, x.grad = x.grad.clone(), None
    dy_householder_torch, y.grad = y.grad.clone(), None

    o_householder_triton.backward(do, retain_graph=True)
    dx_householder_triton, x.grad = x.grad.clone(), None
    dy_householder_triton, y.grad = y.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    # forward check
    print(
        "o diff max: ",
        torch.abs(o_householder_torch - o_householder_triton).max().item(),
    )
    print(
        "o diff norm: ", torch.norm(o_householder_torch - o_householder_triton).item()
    )
    assert torch.allclose(
        o_householder_torch, o_householder_triton, atol=atol, rtol=rtol
    )

    # backward check
    print(
        "dx diff max: ",
        torch.abs(dx_householder_torch - dx_householder_triton).max().item(),
    )
    print(
        "dx diff norm: ",
        torch.norm(dx_householder_torch - dx_householder_triton).item(),
    )
    assert torch.allclose(
        dx_householder_torch, dx_householder_triton, atol=atol, rtol=rtol
    )

    print(
        "dy diff max: ",
        torch.abs(dy_householder_torch - dy_householder_triton).max().item(),
    )
    print(
        "dy diff norm: ",
        torch.norm(dy_householder_torch - dy_householder_triton).item(),
    )
    assert torch.allclose(
        dy_householder_torch, dy_householder_triton, atol=atol, rtol=rtol
    )
