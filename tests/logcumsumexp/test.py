import pytest
import torch

from xopes.ops import (
    logcumsumexp_block_parallel_triton,
    logcumsumexp_block_recurrence_triton,
    logcumsumexp_recurrence_triton,
    logcumsumexp_torch,
)


def get_params():
    shape = [
        # (1, 1, 4),
        # (1, 2, 4),
        # (1, 16, 4),
        # (1, 17, 4),
        # (1, 100, 4),
        # (1, 1000, 4),
        # (1, 1, 5),
        (1, 1, 16),
        # (1, 2, 16),
        # (1, 1000, 16), # not pass
        # (2, 1, 64),
        # (6, 1, 256),
        # (2, 16, 4),
        # (1, 100, 16),
        # (6, 100, 256),
        # (6, 100, 1000),
        # (6, 100, 257),
        # (6, 128, 257),
        # (6, 129, 257),
        # (6, 1000, 257),
        # (6, 1000, 256),
        # (1, 1000, 256),
    ]

    return shape


# @pytest.mark.parametrize("dim", [-1, -2])
# @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
# @pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("shape", get_params())
def test(dim, dtype, shape):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    x = torch.randn(*shape, dtype=dtype, device=device).requires_grad_()
    do = torch.randn(*shape, dtype=dtype, device=device)

    # forward
    o_logcumsumexp_torch = logcumsumexp_torch(x, dim=dim)
    o_logcumsumexp_recurrence_triton = logcumsumexp_recurrence_triton(x, dim=dim)
    o_logcumsumexp_block_recurrence_triton = logcumsumexp_block_recurrence_triton(
        x, dim=dim
    )
    o_logcumsumexp_block_parallel_triton = logcumsumexp_block_parallel_triton(
        x, dim=dim
    )

    # backward
    o_logcumsumexp_torch.backward(do, retain_graph=True)
    dx_logcumsumexp_torch, x.grad = x.grad.clone(), None

    o_logcumsumexp_recurrence_triton.backward(do, retain_graph=True)
    dx_logcumsumexp_recurrence_triton, x.grad = x.grad.clone(), None

    o_logcumsumexp_block_recurrence_triton.backward(do, retain_graph=True)
    dx_logcumsumexp_block_recurrence_triton, x.grad = x.grad.clone(), None

    # # forward
    # assert torch.allclose(
    #     o_logcumsumexp_torch, o_logcumsumexp_recurrence_triton, atol=atol, rtol=rtol
    # ), f"o diff: {torch.abs(o_logcumsumexp_torch - o_logcumsumexp_recurrence_triton).max().item()}"
    # assert torch.allclose(
    #     o_logcumsumexp_torch,
    #     o_logcumsumexp_block_recurrence_triton,
    #     atol=atol,
    #     rtol=rtol,
    # ), f"o diff: {torch.abs(o_logcumsumexp_torch - o_logcumsumexp_block_recurrence_triton).max().item()}"
    # assert torch.allclose(
    #     o_logcumsumexp_torch, o_logcumsumexp_block_parallel_triton, atol=atol, rtol=rtol
    # ), f"o diff: {torch.abs(o_logcumsumexp_torch - o_logcumsumexp_block_parallel_triton).max().item()}"

    # # backward
    # assert torch.allclose(
    #     dx_logcumsumexp_torch, dx_logcumsumexp_recurrence_triton, atol=atol, rtol=rtol
    # ), f"dx diff: {torch.abs(dx_logcumsumexp_torch - dx_logcumsumexp_recurrence_triton).max().item()}"

    # print(o_logcumsumexp_block_triton[0])
    # print(o_logcumsumexp_torch[0])
    # print(torch.norm(o_logcumsumexp_block_triton[0] - o_logcumsumexp_torch[0]).item())
    # print(torch.norm(o_logcumsumexp_block_triton[1] - o_logcumsumexp_torch[1]).item())

    # print(torch.norm(o_logcumsumexp_recurrence_triton - o_logcumsumexp_torch).item())
    # print(torch.norm(o_logcumsumexp_block_recurrence_triton - o_logcumsumexp_torch).item())
    # print(torch.norm(o_logcumsumexp_block_parallel_triton - o_logcumsumexp_torch).item())

    # print(torch.norm(dx_logcumsumexp_torch - dx_logcumsumexp_recurrence_triton).item())
    print(
        torch.norm(
            dx_logcumsumexp_torch - dx_logcumsumexp_block_recurrence_triton
        ).item()
    )
    print(dx_logcumsumexp_torch[0, -1, :5])
    print(dx_logcumsumexp_block_recurrence_triton[0, -1, :5])
    print(dx_logcumsumexp_torch[0, 0, :5])
    print(dx_logcumsumexp_block_recurrence_triton[0, 0, :5])
    # print(dx_logcumsumexp_torch[-1, -1, :5])
    # print(dx_logcumsumexp_block_recurrence_triton[-1, -1, :5])
    # print(dx_logcumsumexp_recurrence_triton[0, -1])
    assert False
