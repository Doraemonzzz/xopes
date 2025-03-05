import pytest
import torch
import torch.nn.functional as F
import triton

from xopes.ops.lightning_attn.scalar_data_dependent_decay.lasd3_parallel_triton import (
    lasd3_parallel_intra,
)
from xopes.ops.lightning_attn.scalar_data_dependent_decay.torch_utils import (
    lasd3_intra_torch,
)
from xopes.utils import get_threshold


def get_params():
    shapes = [
        (2, 128, 8, 64, 128),
        (4, 256, 12, 128, 256),
        (1, 512, 16, 256, 512),
        (2, 255, 7, 33, 63),
        (2, 65, 7, 33, 63),
        (2, 257, 7, 33, 63),
    ]

    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize(
    "reverse",
    [
        True,
        False,
    ],
)

# @pytest.mark.parametrize("reverse", [False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_lasd3_intra(shape, reverse, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")

    # Generate input tensors
    b, n, h, d, e = shape

    q = torch.randn(b, n, h, d, dtype=dtype, device=device)
    k = torch.randn(b, n, h, d, dtype=dtype, device=device)
    v = torch.randn(b, n, h, e, dtype=dtype, device=device)

    # Generate data-dependent decay factors (one per position)
    ld = F.logsigmoid(torch.randn(b, n, h, device=device))

    # Set block sizes for Triton kernel
    MAX_BLOCK_N = triton.next_power_of_2(n)
    MAX_BLOCK_C = MAX_BLOCK_N
    MAX_BLOCK_E = triton.next_power_of_2(e)
    MAX_BLOCK_D = triton.next_power_of_2(d)

    # Choose block size based on sequence length
    if n <= 512:
        BLOCK_N = min(MAX_BLOCK_N, 128)
    else:
        BLOCK_N = 256

    # Forward pass with PyTorch reference implementation
    o_torch = lasd3_intra_torch(q=q, k=k, v=v, ld=ld, reverse=reverse, BLOCK_N=BLOCK_N)

    # Forward pass with Triton parallel implementation
    o_triton = lasd3_parallel_intra(
        q=q,
        k=k,
        v=v,
        ld=ld,
        reverse=reverse,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    # Get thresholds based on dtype for numerical comparison
    atol, rtol = get_threshold(dtype)

    # Print test configuration and results
    print(f"\nShape: {shape}, E: {e}, reverse: {reverse}, dtype: {dtype}")
    print(torch.max(o_torch), torch.max(o_triton))
    print("o diff max: ", torch.abs(o_torch - o_triton).max().item())
    print("o diff norm: ", torch.norm(o_torch - o_triton).item())

    # Assert that the results are close enough
    assert torch.allclose(o_torch, o_triton, atol=atol, rtol=rtol)
