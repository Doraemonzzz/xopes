import pytest
import torch

from xopes.ops.md_lrpe.cosine import (
    md_lrpe_cosine_cache_triton,
    md_lrpe_cosine_parallel_triton,
    md_lrpe_cosine_torch,
    md_lrpe_cosine_triton,
)
from xopes.utils import get_threshold, next_power_of_two


def get_params():
    shape = [
        # 1d
        (6, 8, 128, 64),
        (6, 8, 127, 128),
        # 2d
        (6, 8, 32, 32, 64),
        (6, 8, 2, 3, 64),
        # 3d
        (6, 8, 8, 32, 32, 64),
    ]

    return shape


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(shape, dtype):
    torch.manual_seed(2024)
    device = torch.cuda.current_device()

    h = shape[1]
    d = shape[-1]
    m = len(shape) - 3
    e = next_power_of_two((d + m - 1) // m)
    x = (torch.randn(shape, dtype=dtype, device=device)).requires_grad_()
    theta = torch.randn((h, e), dtype=dtype, device=device)
    shape = shape[:-1] + (shape[-1] * 2,)

    do = torch.randn(shape, dtype=dtype, device=device)

    # forward
    o_md_lrpe_cosine_torch = md_lrpe_cosine_torch(x, theta)
    o_md_lrpe_cosine_triton = md_lrpe_cosine_triton(x, theta)
    o_md_lrpe_cosine_parallel_triton = md_lrpe_cosine_parallel_triton(x, theta)
    o_md_lrpe_cosine_cache_triton = md_lrpe_cosine_cache_triton(x, theta)

    # backward
    o_md_lrpe_cosine_torch.backward(do, retain_graph=True)
    dx_md_lrpe_cosine_torch, x.grad = x.grad.clone(), None

    o_md_lrpe_cosine_triton.backward(do, retain_graph=True)
    dx_md_lrpe_cosine_triton, x.grad = x.grad.clone(), None

    o_md_lrpe_cosine_parallel_triton.backward(do, retain_graph=True)
    dx_md_lrpe_cosine_parallel_triton, x.grad = x.grad.clone(), None

    o_md_lrpe_cosine_cache_triton.backward(do, retain_graph=True)
    dx_md_lrpe_cosine_cache_triton, x.grad = x.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    # forward
    assert torch.allclose(
        o_md_lrpe_cosine_torch, o_md_lrpe_cosine_triton, atol=atol, rtol=rtol
    ), f"o diff: {torch.abs(o_md_lrpe_cosine_torch - o_md_lrpe_cosine_triton).max().item()}"
    assert torch.allclose(
        o_md_lrpe_cosine_torch, o_md_lrpe_cosine_parallel_triton, atol=atol, rtol=rtol
    ), f"o diff: {torch.abs(o_md_lrpe_cosine_torch - o_md_lrpe_cosine_parallel_triton).max().item()}"
    assert torch.allclose(
        o_md_lrpe_cosine_torch, o_md_lrpe_cosine_cache_triton, atol=atol, rtol=rtol
    ), f"o diff: {torch.abs(o_md_lrpe_cosine_torch - o_md_lrpe_cosine_cache_triton).max().item()}"

    # backward
    assert torch.allclose(
        dx_md_lrpe_cosine_torch, dx_md_lrpe_cosine_triton, atol=atol, rtol=rtol
    ), f"dx diff: {torch.abs(dx_md_lrpe_cosine_torch - dx_md_lrpe_cosine_triton).max().item()}"
    assert torch.allclose(
        dx_md_lrpe_cosine_torch, dx_md_lrpe_cosine_parallel_triton, atol=atol, rtol=rtol
    ), f"dx diff: {torch.abs(dx_md_lrpe_cosine_torch - dx_md_lrpe_cosine_parallel_triton).max().item()}"
    assert torch.allclose(
        dx_md_lrpe_cosine_torch, dx_md_lrpe_cosine_cache_triton, atol=atol, rtol=rtol
    ), f"dx diff: {torch.abs(dx_md_lrpe_cosine_torch - dx_md_lrpe_cosine_cache_triton).max().item()}"
