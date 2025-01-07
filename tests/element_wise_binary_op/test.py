import pytest
import torch

from xopes.ops.element_wise_binary_op import ewbo_torch, ewbo_triton
from xopes.utils import get_threshold


def get_params():
    shapes = [
        ((6, 128), (6,)),
        ((4, 8, 256), (4, 8)),
        ((4, 1024, 1024), (4, 1024, 1024)),
        ((512, 2000), ()),
    ]

    return shapes


@pytest.mark.parametrize("shapes", get_params())
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("op", ["add", "mul", "sub", "div"])
def test(shapes, op, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    x_shape, y_shape = shapes

    # Generate input tensors
    x = torch.randn(x_shape, dtype=dtype, device=device).requires_grad_()
    y = torch.randn(y_shape, dtype=dtype, device=device) ** 2 + 1
    y.requires_grad_()
    do = torch.randn(x_shape, dtype=dtype, device=device)

    # forward
    o_torch = ewbo_torch(x, y, op)
    o_triton = ewbo_triton(x, y, op)

    # backward
    o_torch.backward(do, retain_graph=True)
    dx_torch, x.grad = x.grad.clone(), None
    dy_torch, y.grad = y.grad.clone(), None

    o_triton.backward(do, retain_graph=True)
    dx_triton, x.grad = x.grad.clone(), None
    dy_triton, y.grad = y.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    # forward check
    print(f"o diff max: {torch.abs(o_torch - o_triton).max().item()}")
    print(f"o diff norm: {torch.norm(o_torch - o_triton).item()}")
    assert torch.allclose(o_torch, o_triton, atol=atol, rtol=rtol)

    # backward check
    print(f"dx diff max: {torch.abs(dx_torch - dx_triton).max().item()}")
    print(f"dx diff norm: {torch.norm(dx_torch - dx_triton).item()}")
    assert torch.allclose(dx_torch, dx_triton, atol=atol, rtol=rtol)

    print(f"dy diff max: {torch.abs(dy_torch - dy_triton).max().item()}")
    print(f"dy diff norm: {torch.norm(dy_torch - dy_triton).item()}")
    assert torch.allclose(dy_torch, dy_triton, atol=atol, rtol=rtol)
