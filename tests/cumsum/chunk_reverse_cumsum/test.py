import pytest
import torch

from xopes.ops.cumsum.chunk_reverse_cumsum import (
    chunk_reverse_cumsum_torch,
    chunk_reverse_cumsum_triton,
)
from xopes.utils import get_threshold


def get_params():
    shapes = [(6, 128), (4, 8, 256), (4, 1024, 4096), (12, 32, 15), (2, 129), (8, 255)]

    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("chunk_size", [128, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_chunk_reverse_cumsum(shape, dim, chunk_size, dtype):
    # Set random seed for reproducibility
    torch.manual_seed(2024)
    device = torch.device("cuda")

    # Generate input tensor
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    do = torch.randn(shape, dtype=dtype, device=device)

    # Skip invalid dim values
    if abs(dim) >= len(shape):
        return

    # Forward pass
    o_chunk_reverse_cumsum_torch = chunk_reverse_cumsum_torch(
        x, dim=dim, chunk_size=chunk_size
    )
    o_chunk_reverse_cumsum_triton = chunk_reverse_cumsum_triton(
        x, dim=dim, chunk_size=chunk_size
    )

    # Get tolerance based on dtype
    atol, rtol = get_threshold(dtype)

    # Forward check
    print(
        f"o diff max: {torch.abs(o_chunk_reverse_cumsum_torch - o_chunk_reverse_cumsum_triton).max().item()}"
    )
    print(
        f"o diff norm: {torch.norm(o_chunk_reverse_cumsum_torch - o_chunk_reverse_cumsum_triton).item()}"
    )
    assert torch.allclose(
        o_chunk_reverse_cumsum_torch,
        o_chunk_reverse_cumsum_triton,
        atol=atol,
        rtol=rtol,
    )
