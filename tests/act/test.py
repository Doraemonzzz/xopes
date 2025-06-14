import pytest
import torch

from xopes.ops.act import act_bwd_torch, act_torch, act_triton
from xopes.utils import get_threshold


def get_params():
    shape = [
        (64, 1),
        (16, 2),
        (6, 128, 64),
        (6, 256, 127),
        (6, 8, 256, 127),
    ]

    return shape


@pytest.mark.parametrize("shape", get_params())
# without dim
@pytest.mark.parametrize("act", ["relu", "sigmoid", "silu", "none"])
# @pytest.mark.parametrize("act", ["relu",])
@pytest.mark.parametrize("dim", [None])
# # with dim
# @pytest.mark.parametrize(
#     "act",
#     [
#         "softmax",
#     ],
# )
# @pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(shape, act, dim, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    x = torch.randn(*shape, dtype=dtype, device=device).requires_grad_()
    do = torch.randn(*shape, dtype=dtype, device=device)

    # forward
    o_act_torch = act_torch(x, act, dim)
    o_act_triton = act_triton(x, act, dim)

    # backward
    o_act_torch.backward(do, retain_graph=True)
    dx_act_torch, x.grad = x.grad.clone(), None

    o_act_triton.backward(do, retain_graph=True)
    dx_act_triton, x.grad = x.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    # forward check
    print("o diff max: ", torch.abs(o_act_torch - o_act_triton).max().item())
    print("o diff norm: ", torch.norm(o_act_torch - o_act_triton).item())
    assert torch.allclose(
        o_act_torch, o_act_triton, atol=atol, rtol=rtol
    ), f"o diff: {torch.abs(o_act_torch - o_act_triton).max().item()}"

    # backward check
    print("dx diff max: ", torch.abs(dx_act_torch - dx_act_triton).max().item())
    print("dx diff norm: ", torch.norm(dx_act_torch - dx_act_triton).item())
    assert torch.allclose(
        dx_act_torch, dx_act_triton, atol=atol, rtol=rtol
    ), f"dx diff: {torch.abs(dx_act_torch - dx_act_triton).max().item()}"

    if act not in ["softmax"]:
        dx_act_bwd_torch = act_bwd_torch(x, do, act, dim)
        print("dx diff max: ", torch.abs(dx_act_bwd_torch - dx_act_torch).max().item())
        print("dx diff norm: ", torch.norm(dx_act_bwd_torch - dx_act_torch).item())
        assert torch.allclose(
            dx_act_bwd_torch, dx_act_torch, atol=atol, rtol=rtol
        ), f"do diff: {torch.abs(dx_act_bwd_torch - dx_act_torch).max().item()}"
