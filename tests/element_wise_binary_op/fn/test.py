import pytest
import torch

from xopes.ops.element_wise_binary_op import ewbo_torch, ewbo_triton_fwd_fn
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
@pytest.mark.parametrize("inplace", [True, False])
def test(shapes, op, dtype, inplace):
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
    o_triton = ewbo_triton_fwd_fn(x, y, op, inplace)

    atol, rtol = get_threshold(dtype)

    # forward check
    print(f"o diff max: {torch.abs(o_torch - o_triton).max().item()}")
    print(f"o diff norm: {torch.norm(o_torch - o_triton).item()}")
    assert torch.allclose(o_torch, o_triton, atol=atol, rtol=rtol)
