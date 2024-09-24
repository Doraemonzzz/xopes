import pytest
import torch

from xopes.ops.multinomial import (
    multinomial_torch,
    online_multinomial_torch,
    online_multinomial_triton,
    online_with_cache_multinomial_torch,
    parallel_multinomial_triton,
)
from xopes.utils import get_threshold


def get_params():
    shape = [(12, 4096, 2048), (12, 16, 16), (12, 16, 256)]

    return shape


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("num_samples", [1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n", [8])
def test(shape, num_samples, dtype, n):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, d, V = shape
    x = torch.ones((b, d), dtype=dtype, device=device)
    value = -1e5

    v = V // n
    for i in range(n):
        # for i in [6]:
        W = torch.full((d, V), fill_value=value, dtype=dtype, device=device)
        index = torch.tensor([i * v], dtype=torch.int64, device=device)
        W[:, index] = 1

        # forward
        sample_torch = multinomial_torch(x, W, num_samples)
        sample_online_torch = online_multinomial_torch(x, W, num_samples)
        sample_online_with_cache_torch = online_with_cache_multinomial_torch(
            x, W, num_samples
        )
        sample_online_triton = online_multinomial_triton(x, W, num_samples)
        sample_parallel_triton = parallel_multinomial_triton(x, W, num_samples)

        atol, rtol = get_threshold(dtype)

        assert torch.allclose(
            sample_torch, sample_online_torch, atol=atol, rtol=rtol
        ), f"o diff: {torch.abs(sample_torch - sample_online_torch).max().item()}"

        assert torch.allclose(
            sample_torch, sample_online_with_cache_torch, atol=atol, rtol=rtol
        ), f"o diff: {torch.abs(sample_torch - sample_online_with_cache_torch).max().item()}"

        assert torch.allclose(
            sample_torch, sample_online_triton, atol=atol, rtol=rtol
        ), f"o diff: {torch.abs(sample_torch - sample_online_triton).max().item()}"

        assert torch.allclose(
            sample_torch, sample_parallel_triton, atol=atol, rtol=rtol
        ), f"o diff: {torch.abs(sample_torch - sample_parallel_triton).max().item()}"
