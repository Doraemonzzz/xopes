import pytest
import torch

from xopes.ops import (
    logcumsumexp_block_recurrence_triton,
    logcumsumexp_recurrence_triton,
    logcumsumexp_torch,
)


def get_params():
    shape = [
        (2, 1, 64),
        (6, 1, 256),
        (1, 2, 4),
        (6, 100, 256),
        (6, 100, 1000),
        (6, 100, 257),
    ]

    return shape


@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", get_params())
# @pytest.mark.parametrize("dim", [-1])
# @pytest.mark.parametrize("dtype", [torch.float32])
# @pytest.mark.parametrize("shape", get_params())
def test_lightning2(dim, dtype, shape):
    torch.manual_seed(2024)
    atol = 5e-2
    rtol = 5e-2
    device = torch.device("cuda")
    x = torch.randn(*shape, dtype=dtype, device=device).requires_grad_()

    # forward
    o_logcumsumexp_torch = logcumsumexp_torch(x, dim=dim)

    o_logcumsumexp_recurrence_triton = logcumsumexp_recurrence_triton(x, dim=dim)
    o_logcumsumexp_block_triton = logcumsumexp_block_recurrence_triton(x, dim=dim)

    assert torch.allclose(
        o_logcumsumexp_torch, o_logcumsumexp_recurrence_triton, atol=atol, rtol=rtol
    ), f"o diff: {torch.abs(o_logcumsumexp_torch - o_logcumsumexp_recurrence_triton).max().item()}"
    assert torch.allclose(
        o_logcumsumexp_torch, o_logcumsumexp_block_triton, atol=atol, rtol=rtol
    ), f"o diff: {torch.abs(o_logcumsumexp_torch - o_logcumsumexp_recurrence_triton).max().item()}"

    # print(torch.norm(o_logcumsumexp_block_triton - o_logcumsumexp_torch).item())

    # print(o_logcumsumexp_block_triton[0])
    # print(o_logcumsumexp_torch[0])
    # print(torch.norm(o_logcumsumexp_block_triton[0] - o_logcumsumexp_torch[0]).item())
    # print(torch.norm(o_logcumsumexp_block_triton[1] - o_logcumsumexp_torch[1]).item())
    # assert False
