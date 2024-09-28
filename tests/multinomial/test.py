import pytest
import torch

from xopes.ops.multinomial import (
    multinomial_torch,
    online_multinomial_torch,
    parallel_gumbel_multinomial_triton,
)
from xopes.utils import get_threshold


def get_params():
    # shape = [(12, 4096, 2048), (12, 16, 16), (12, 16, 256), (12, 256, 384)]

    shape = [(12, 4096, 2048)]
    # shape = [(12, 16, 64)]  # fail
    shape = [(12, 16, 128)]
    # shape = [(12, 16, 256)]
    # shape = [(12, 256, 384)]

    return shape


@pytest.mark.parametrize("shape", get_params())
# @pytest.mark.parametrize("num_samples", [1, 8])
# @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_samples", [1])
@pytest.mark.parametrize("n", [8])  # test times
# @pytest.mark.parametrize("top_k", [-1, 1])
@pytest.mark.parametrize("top_k", [1])
@pytest.mark.parametrize("dtype", [torch.float32])
def test(shape, num_samples, n, top_k, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, d, V = shape
    x = torch.ones((b, d), dtype=dtype, device=device)
    value = -1e5

    if top_k == -1:
        v = V // n
        for i in range(n):
            W = torch.full((d, V), fill_value=value, dtype=dtype, device=device)
            index = torch.tensor([i * v], dtype=torch.int64, device=device)
            W[:, index] = 1

            # forward
            multinomial_torch(x, W, num_samples, top_k=top_k)
            sample_online_torch, lse_online_torch = online_multinomial_torch(
                x, W, num_samples
            )
            # sample_online_with_cache_torch = online_with_cache_multinomial_torch(
            #     x, W, num_samples
            # )
            # sample_online_triton = online_multinomial_triton(x, W, num_samples)
            # sample_parallel_triton = parallel_multinomial_triton(x, W, num_samples)
            (
                sample_parallel_gumbel_triton,
                lse_parallel_gumbel_triton,
            ) = parallel_gumbel_multinomial_triton(
                x,
                W,
                num_samples,
                output_lse=True,
                top_k=top_k,
            )

            atol, rtol = get_threshold(dtype)

            # assert torch.allclose(
            #     sample_torch, sample_online_torch, atol=atol, rtol=rtol
            # ), f"o diff: {torch.abs(sample_torch - sample_online_torch).max().item()}"

            # assert torch.allclose(
            #     sample_torch, sample_online_with_cache_torch, atol=atol, rtol=rtol
            # ), f"o diff: {torch.abs(sample_torch - sample_online_with_cache_torch).max().item()}"

            # assert torch.allclose(
            #     sample_torch, sample_online_triton, atol=atol, rtol=rtol
            # ), f"o diff: {torch.abs(sample_torch - sample_online_triton).max().item()}"

            # assert torch.allclose(
            #     sample_torch, sample_parallel_triton, atol=atol, rtol=rtol
            # ), f"o diff: {torch.abs(sample_torch - sample_parallel_triton).max().item()}"

            # assert torch.allclose(
            #     sample_torch, sample_parallel_gumbel_triton, atol=atol, rtol=rtol
            # ), f"o diff: {torch.abs(sample_torch - sample_parallel_gumbel_triton).max().item()}"
    else:
        x = torch.randn((b, d), dtype=dtype, device=device)
        W = torch.randn((d, V), dtype=dtype, device=device)

        sample_torch = multinomial_torch(x, W, num_samples, top_k=top_k)
        (
            sample_parallel_gumbel_triton,
            lse_parallel_gumbel_triton,
        ) = parallel_gumbel_multinomial_triton(
            x,
            W,
            num_samples,
            output_lse=True,
            top_k=top_k,
        )

        atol, rtol = get_threshold(dtype)

        print(sample_torch)
        print(sample_parallel_gumbel_triton)

        assert torch.allclose(
            sample_torch, sample_parallel_gumbel_triton, atol=atol, rtol=rtol
        ), f"o diff: {torch.abs(sample_torch - sample_parallel_gumbel_triton).max().item()}"

    # # check lse
    # x = torch.randn((b, d), dtype=dtype, device=device)
    # W = torch.randn((d, V), dtype=dtype, device=device)

    # # forward
    # multinomial_torch(x, W, num_samples)
    # sample_online_torch, lse_online_torch = online_multinomial_torch(x, W, num_samples)
    # # sample_online_with_cache_torch = online_with_cache_multinomial_torch(
    # #     x, W, num_samples
    # # )
    # # sample_online_triton = online_multinomial_triton(x, W, num_samples)
    # # sample_parallel_triton = parallel_multinomial_triton(x, W, num_samples)
    # (
    #     sample_parallel_gumbel_triton,
    #     lse_parallel_gumbel_triton,
    # ) = parallel_gumbel_multinomial_triton(
    #     x,
    #     W,
    #     num_samples,
    #     output_lse=True,
    # )

    # atol, rtol = THRESHOLD_DICT = {
    #     torch.float32: [5e-2, 5e-2],
    #     torch.float16: [5e-2, 5e-2],
    #     torch.bfloat16: [5e-2, 5e-2],
    # }[dtype]

    # # print(lse_online_torch.shape)
    # # print(lse_parallel_gumbel_triton.shape)

    # # print("aaa")
    # # print(lse_online_torch)
    # # print(lse_parallel_gumbel_triton)
    # # assert False

    # print(lse_online_torch.shape, lse_parallel_gumbel_triton.shape)
    # # check lse
    # assert torch.allclose(
    #     lse_online_torch, lse_parallel_gumbel_triton, atol=atol, rtol=rtol
    # ), f"lse diff: {torch.abs(lse_online_torch - lse_parallel_gumbel_triton).max().item()}"
