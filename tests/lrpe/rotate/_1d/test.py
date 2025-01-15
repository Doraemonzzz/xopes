import pytest
import torch

from xopes.ops.lrpe.rotate._1d import lrpe_rotate_1d_sp_triton, lrpe_rotate_1d_torch
from xopes.utils import get_threshold


def get_params():
    shape = [
        (6, 128, 8, 64),
        (6, 128, 7, 128),
        (6, 33, 8, 48),
    ]
    return shape


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("shape_t", [[-1, -1], [1, -1]])
@pytest.mark.parametrize("offset", [0, 10])
# without dim
# @pytest.mark.parametrize("act", ["none", "relu", "sigmoid", "silu",])
# @pytest.mark.parametrize("dim", [None])
# @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
# with dim
@pytest.mark.parametrize("act", ["softmax"])
@pytest.mark.parametrize("dim", [1, -1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(shape, shape_t, offset, act, dim, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, n, h, d = shape
    h_t, d_t = shape_t
    if h_t == -1:
        h_t = h
    if d_t == -1:
        d_t = d // 2  # Note: rotate uses d//2 instead of d for theta
    x = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    theta = torch.randn((h_t, d_t), dtype=dtype, device=device)
    do = torch.randn((b, n, h, d), dtype=dtype, device=device)

    # forward
    o_lrpe_rotate_1d_torch = lrpe_rotate_1d_torch(
        x=x, theta=theta, offset=offset, act=act, dim=dim
    )
    o_lrpe_rotate_1d_sp_triton = lrpe_rotate_1d_sp_triton(
        x=x, theta=theta, offset=offset, act=act, dim=dim
    )

    # backward
    o_lrpe_rotate_1d_torch.backward(do, retain_graph=True)
    dx_lrpe_rotate_1d_torch, x.grad = x.grad.clone(), None

    o_lrpe_rotate_1d_sp_triton.backward(do, retain_graph=True)
    dx_lrpe_rotate_1d_sp_triton, x.grad = x.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    # forward
    print(
        "o diff max: ",
        torch.abs(o_lrpe_rotate_1d_torch - o_lrpe_rotate_1d_sp_triton).max().item(),
    )
    print(
        "o diff norm: ",
        torch.norm(o_lrpe_rotate_1d_torch - o_lrpe_rotate_1d_sp_triton).item(),
    )
    assert torch.allclose(
        o_lrpe_rotate_1d_torch, o_lrpe_rotate_1d_sp_triton, atol=atol, rtol=rtol
    ), f"o diff: {torch.abs(o_lrpe_rotate_1d_torch - o_lrpe_rotate_1d_sp_triton).max().item()}"

    # backward
    print(
        "dx diff max: ",
        torch.abs(dx_lrpe_rotate_1d_torch - dx_lrpe_rotate_1d_sp_triton).max().item(),
    )
    print(
        "dx diff norm: ",
        torch.norm(dx_lrpe_rotate_1d_torch - dx_lrpe_rotate_1d_sp_triton).item(),
    )
    assert torch.allclose(
        dx_lrpe_rotate_1d_torch, dx_lrpe_rotate_1d_sp_triton, atol=atol, rtol=rtol
    ), f"dx diff: {torch.abs(dx_lrpe_rotate_1d_torch - dx_lrpe_rotate_1d_sp_triton).max().item()}"
