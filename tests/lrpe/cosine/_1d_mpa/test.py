import pytest
import torch

from xopes.ops.lrpe.cosine._1d import lrpe_cosine_1d_sp_triton, lrpe_cosine_1d_torch
from xopes.utils import get_threshold


def get_params():
    shape = [
        (
            6,
            128,
            64,
        ),
        (6, 128, 128),
        (6, 128, 33),
        (6, 33, 48),
    ]

    return shape


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("offset", [0, 10])
# without dim
@pytest.mark.parametrize("act", ["none", "relu", "sigmoid", "silu"])
@pytest.mark.parametrize("dim", [None])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
# # with dim
# @pytest.mark.parametrize("act", ["softmax"])
# @pytest.mark.parametrize("dim", [1, -1])
# @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(shape, offset, act, dim, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, n, d = shape
    x = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()
    theta = torch.randn((1, d), dtype=dtype, device=device)
    do = torch.randn((b, n, 2 * d), dtype=dtype, device=device)

    # forward
    o_lrpe_cosine_1d_torch = lrpe_cosine_1d_torch(
        x=x, theta=theta, offset=offset, act=act, dim=dim
    )
    o_lrpe_cosine_1d_sp_triton = lrpe_cosine_1d_sp_triton(
        x=x, theta=theta, offset=offset, act=act, dim=dim
    )
    # backward
    o_lrpe_cosine_1d_torch.backward(do, retain_graph=True)
    dx_lrpe_cosine_1d_torch, x.grad = x.grad.clone(), None

    o_lrpe_cosine_1d_sp_triton.backward(do, retain_graph=True)
    dx_lrpe_cosine_1d_sp_triton, x.grad = x.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    # forward
    print(
        "o diff max: ",
        torch.abs(o_lrpe_cosine_1d_torch - o_lrpe_cosine_1d_sp_triton).max().item(),
    )
    print(
        "o diff norm: ",
        torch.norm(o_lrpe_cosine_1d_torch - o_lrpe_cosine_1d_sp_triton).item(),
    )
    assert torch.allclose(
        o_lrpe_cosine_1d_torch, o_lrpe_cosine_1d_sp_triton, atol=atol, rtol=rtol
    ), f"o diff: {torch.abs(o_lrpe_cosine_1d_torch - o_lrpe_cosine_1d_sp_triton).max().item()}"

    # backward
    print(
        "dx diff max: ",
        torch.abs(dx_lrpe_cosine_1d_torch - dx_lrpe_cosine_1d_sp_triton).max().item(),
    )
    print(
        "dx diff norm: ",
        torch.norm(dx_lrpe_cosine_1d_torch - dx_lrpe_cosine_1d_sp_triton).item(),
    )
    assert torch.allclose(
        dx_lrpe_cosine_1d_torch, dx_lrpe_cosine_1d_sp_triton, atol=atol, rtol=rtol
    ), f"dx diff: {torch.abs(dx_lrpe_cosine_1d_torch - dx_lrpe_cosine_1d_sp_triton).max().item()}"
