import pytest
import torch

from xopes.ops.cumsum import cumsum_torch
from xopes.ops.cumsum.chunk_cumsum import chunk_cumsum_torch
from xopes.ops.cumsum.chunk_cumsum_reduce import (
    chunk_cumsum_reduce_torch,
    chunk_cumsum_reduce_triton,
)
from xopes.utils import get_threshold


def get_params():
    shapes = [(6, 128), (4, 8, 256), (4, 1024, 4096), (12, 32, 15)]

    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize(
    "dim",
    [
        -1,
    ],
)
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("use_cu_seqlens", [False])
@pytest.mark.parametrize("chunk_size", [32, 64])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
    ],
)
def test(shape, dim, reverse, use_cu_seqlens, chunk_size, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")

    if use_cu_seqlens:
        n = int(torch.randint(1024, 4096, (1,)).item())
        m = n // 5
        cu_seqlens = torch.tensor(
            [0, m - 2, 2 * m + 1, 3 * m - 1, 4 * m, n], dtype=torch.long, device=device
        )
        d = 768
        dim = 0
        shape = (n, d)
    else:
        cu_seqlens = None

    # Generate input tensor
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()

    # Skip invalid dim values
    if abs(dim) >= len(shape):
        return

    # forward
    o_cumsum_torch = cumsum_torch(x, dim=dim, reverse=reverse, cu_seqlens=cu_seqlens)
    o_chunk_cumsum_torch = chunk_cumsum_torch(
        x, dim=dim, reverse=reverse, chunk_size=chunk_size
    )
    o_cumsum_reduce = chunk_cumsum_reduce_torch(
        o_chunk_cumsum_torch, dim=dim, reverse=reverse, chunk_size=chunk_size
    )
    o_cumsum_reduce_triton = chunk_cumsum_reduce_triton(
        o_chunk_cumsum_torch, dim=dim, reverse=reverse, chunk_size=chunk_size
    )

    atol, rtol = get_threshold(dtype)

    # forward check
    print(
        "o diff max: (Vs torch cumsum reduce)",
        torch.abs(o_cumsum_torch - o_cumsum_reduce).max().item(),
    )
    print(
        "o diff norm: (Vs torch cumsum reduce)",
        torch.norm(o_cumsum_torch - o_cumsum_reduce).item(),
    )
    assert torch.allclose(o_cumsum_torch, o_cumsum_reduce, atol=atol, rtol=rtol)

    print(
        "o diff max: (Vs torch cumsum reduce triton)",
        torch.abs(o_cumsum_torch - o_cumsum_reduce_triton).max().item(),
    )
    print(
        "o diff norm: (Vs torch cumsum reduce triton)",
        torch.norm(o_cumsum_torch - o_cumsum_reduce_triton).item(),
    )
    assert torch.allclose(o_cumsum_torch, o_cumsum_reduce_triton, atol=atol, rtol=rtol)
