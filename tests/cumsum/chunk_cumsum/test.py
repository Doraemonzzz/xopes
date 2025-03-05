import pytest
import torch

from xopes.ops.cumsum.chunk_cumsum import chunk_cumsum_torch, chunk_cumsum_triton
from xopes.utils import get_threshold


def get_params():
    shapes = [(6, 128), (4, 8, 256), (4, 1024, 4096), (12, 32, 15), (2, 129), (8, 255)]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("chunk_size", [128, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_chunk_cumsum(shape, dim, reverse, chunk_size, dtype):
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
    o_chunk_cumsum_torch = chunk_cumsum_torch(
        x, dim=dim, reverse=reverse, chunk_size=chunk_size
    )
    o_chunk_cumsum_triton = chunk_cumsum_triton(
        x, dim=dim, reverse=reverse, chunk_size=chunk_size
    )

    # Backward pass
    o_chunk_cumsum_torch.backward(do, retain_graph=True)
    dx_chunk_cumsum_torch, x.grad = x.grad.clone(), None

    o_chunk_cumsum_triton.backward(do, retain_graph=True)
    dx_chunk_cumsum_triton, x.grad = x.grad.clone(), None

    # Get tolerance based on dtype
    atol, rtol = get_threshold(dtype)

    # Forward check
    print(
        f"o diff max: {torch.abs(o_chunk_cumsum_torch - o_chunk_cumsum_triton).max().item()}"
    )
    print(
        f"o diff norm: {torch.norm(o_chunk_cumsum_torch - o_chunk_cumsum_triton).item()}"
    )
    o_chunk_cumsum_torch - o_chunk_cumsum_triton
    assert torch.allclose(
        o_chunk_cumsum_torch, o_chunk_cumsum_triton, atol=atol, rtol=rtol
    )

    # Backward check
    print(
        f"dx diff max: {torch.abs(dx_chunk_cumsum_torch - dx_chunk_cumsum_triton).max().item()}"
    )
    print(
        f"dx diff norm: {torch.norm(dx_chunk_cumsum_torch - dx_chunk_cumsum_triton).item()}"
    )
    assert torch.allclose(
        dx_chunk_cumsum_torch, dx_chunk_cumsum_triton, atol=atol, rtol=rtol
    )
