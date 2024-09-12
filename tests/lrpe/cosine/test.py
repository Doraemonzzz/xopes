import pytest
import torch

from xopes.ops.lrpe.cosine import lrpe_cosine_torch, lrpe_cosine_triton
from xopes.utils import get_threshold


def get_params():
    shape = [
        (6, 8, 128, 64),
        (6, 8, 127, 128),
        (6, 8, 1024, 128),
    ]

    return shape


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("offset", [0, 10])
# without dim
# @pytest.mark.parametrize("act", ["relu", "sigmoid", "silu", "none"])
# @pytest.mark.parametrize("dim", [None])
# with dim
@pytest.mark.parametrize("act", ["softmax"])
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(shape, offset, act, dim, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, h, n, d = shape
    x = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
    theta = torch.randn((h, d), dtype=dtype, device=device)
    do = torch.randn((b, h, n, 2 * d), dtype=dtype, device=device)

    # forward
    o_lrpe_cosine_torch = lrpe_cosine_torch(x, theta, offset, act, dim)
    o_lrpe_cosine_triton = lrpe_cosine_triton(x, theta, offset, act, dim)

    # backward
    o_lrpe_cosine_torch.backward(do, retain_graph=True)
    dx_lrpe_cosine_torch, x.grad = x.grad.clone(), None

    o_lrpe_cosine_triton.backward(do, retain_graph=True)
    dx_lrpe_cosine_triton, x.grad = x.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    # forward
    assert torch.allclose(
        o_lrpe_cosine_torch, o_lrpe_cosine_triton, atol=atol, rtol=rtol
    ), f"o diff: {torch.abs(o_lrpe_cosine_torch - o_lrpe_cosine_triton).max().item()}"

    # backward
    assert torch.allclose(
        dx_lrpe_cosine_torch, dx_lrpe_cosine_triton, atol=atol, rtol=rtol
    ), f"dx diff: {torch.abs(dx_lrpe_cosine_torch - dx_lrpe_cosine_triton).max().item()}"
