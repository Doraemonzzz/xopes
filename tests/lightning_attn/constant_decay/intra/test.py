import pytest
import torch
import torch.nn.functional as F
import triton

from xopes.ops.lightning_attn.constant_decay.lacd_parallel_triton import (
    lacd_parallel_intra,
)
from xopes.ops.lightning_attn.constant_decay.torch_utils import lacd_intra_torch
from xopes.utils import get_threshold


def get_params():
    shapes = [
        # standard shape
        (2, 256, 12, 128, 128),
        (2, 1024, 8, 32, 16),
        # BLOCK_N +- 1
        (2, 257, 8, 64, 32),
        (2, 255, 8, 64, 32),
        (2, 65, 7, 33, 63),
        # BLOCK_N +- C
        (2, 270, 8, 64, 32),
        (2, 270, 8, 33, 16),
        (2, 1125, 8, 43, 33),
    ]

    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_ld", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_lacd_intra(shape, use_ld, reverse, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")

    # Generate input tensors
    b, n, h, d, e = shape

    q = torch.randn(b, n, h, d, dtype=dtype, device=device)
    k = torch.randn(b, n, h, d, dtype=dtype, device=device)
    v = torch.randn(b, n, h, e, dtype=dtype, device=device)

    ld = None
    if use_ld:
        ld = F.logsigmoid(torch.randn(h, device=device))

    MAX_BLOCK_N = triton.next_power_of_2(n)
    MAX_BLOCK_C = MAX_BLOCK_N
    MAX_BLOCK_E = triton.next_power_of_2(e)
    MAX_BLOCK_D = triton.next_power_of_2(d)
    BLOCK_N = 64

    # Forward pass
    o_torch = lacd_intra_torch(q=q, k=k, v=v, ld=ld, reverse=reverse, BLOCK_N=BLOCK_N)

    o_triton = lacd_parallel_intra(
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

    # Get thresholds based on dtype
    atol, rtol = get_threshold(dtype)

    # Forward check
    print(
        f"\nShape: {shape}, E: {e}, use_ld: {use_ld}, reverse: {reverse}, dtype: {dtype}"
    )
    print("o diff max: ", torch.abs(o_torch - o_triton).max().item())
    print("o diff norm: ", torch.norm(o_torch - o_triton).item())

    assert torch.allclose(o_torch, o_triton, atol=atol, rtol=rtol)
