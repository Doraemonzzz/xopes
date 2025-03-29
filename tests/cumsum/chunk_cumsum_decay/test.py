import pytest
import torch

from xopes.ops.cumsum.chunk_cumsum import chunk_cumsum_torch
from xopes.ops.cumsum.chunk_cumsum_decay import chunk_cumsum_decay_triton
from xopes.ops.cumsum.chunk_reverse_cumsum import chunk_reverse_cumsum_torch
from xopes.utils import get_threshold


def get_params():
    # Test with various shapes to ensure robustness
    shapes = [
        # b n h
        (2, 128, 64),
        (4, 256, 128),
        (4, 256, 129),
        (4, 256, 127),
        (1, 129, 64),
        (8, 255, 128),
        # b n h d
        (2, 128, 16, 64),
        (4, 256, 16, 128),
        (4, 256, 16, 129),
        (4, 256, 16, 127),
        (1, 129, 16, 64),
        (8, 255, 16, 128),
    ]

    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("chunk_size", [128, 64, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_chunk_cumsum_decay(shape, reverse, chunk_size, dtype):
    # Set random seed for reproducibility
    torch.manual_seed(2024)
    device = torch.device("cuda")

    # Generate input tensor
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()

    # Forward pass
    if reverse:
        fn = chunk_reverse_cumsum_torch
    else:
        fn = chunk_cumsum_torch

    o_torch = fn(x, dim=1, reverse=reverse, chunk_size=chunk_size)
    o_triton = chunk_cumsum_decay_triton(x, reverse=reverse, chunk_size=chunk_size)

    # Get tolerance based on dtype
    atol, rtol = get_threshold(dtype)

    # Forward check
    print(
        f"Shape: {shape}, Reverse: {reverse}, Chunk size: {chunk_size}, Dtype: {dtype}"
    )
    print(f"o diff max: {torch.abs(o_torch - o_triton).max().item()}")
    print(f"o diff norm: {torch.norm(o_torch - o_triton).item()}")
    assert torch.allclose(
        o_torch,
        o_triton,
        atol=atol,
        rtol=rtol,
    )
