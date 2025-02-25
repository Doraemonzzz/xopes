import pytest
import torch
import torch.nn.functional as F
import triton

from xopes.ops.lightning_attn.scalar_decay.lasd_parallel_triton import (
    lasd_parallel_intra,
)
from xopes.ops.lightning_attn.scalar_decay.torch_utils import lasd_intra_torch
from xopes.utils import get_threshold


def get_params():
    shapes = [
        (2, 128, 8, 64, 128),
        (4, 256, 12, 128, 256),
        (1, 512, 16, 256, 512),
        (2, 255, 7, 33, 63),
        (2, 65, 7, 33, 63),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_ld", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_lasd_intra(shape, use_ld, reverse, dtype):
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
    BLOCK_N = MAX_BLOCK_N

    # Forward pass
    o_torch = lasd_intra_torch(q=q, k=k, v=v, ld=ld, reverse=reverse)

    o_triton = lasd_parallel_intra(
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
