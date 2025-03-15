import pytest
import torch
import torch.nn.functional as F
import triton

from xopes.ops.lightning_attn.scalar_decay.lasd_parallel_triton import (
    lasd_parallel_inter,
    lasd_parallel_state_parallel,
    lasd_parallel_state_reduce,
)
from xopes.ops.lightning_attn.scalar_decay.torch_utils import lasd_inter_torch
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
@pytest.mark.parametrize("use_initial_state", [True, False])
@pytest.mark.parametrize("trans", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_lasd_inter(shape, use_ld, use_initial_state, trans, dtype):
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

    initial_state = None
    if use_initial_state:
        initial_state = torch.randn(b, h, d, e, dtype=dtype, device=device)

    # Calculate block sizes
    MAX_BLOCK_N = triton.next_power_of_2(n)
    MAX_BLOCK_C = MAX_BLOCK_N
    MAX_BLOCK_E = triton.next_power_of_2(e)
    MAX_BLOCK_D = triton.next_power_of_2(d)
    BLOCK_N = 64
    MAX_BLOCK_C = BLOCK_N

    # Compute using reference implementation
    o_inter_torch = lasd_inter_torch(
        q=q,
        k=k,
        v=v,
        ld=ld,
        initial_state=initial_state,
        BLOCK_N=BLOCK_N,
    )

    # Compute using Triton implementation
    local_states = lasd_parallel_state_parallel(
        k=k,
        v=v,
        ld=ld,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    global_states = lasd_parallel_state_reduce(
        b=b,
        n=n,
        h=h,
        d=d,
        e=e,
        states=local_states,
        initial_state=initial_state,
        ld=ld,
        cu_seqlens=None,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    if trans:
        global_states = torch.transpose(global_states, -1, -2).contiguous()

    o_inter_triton = torch.zeros_like(o_inter_torch)

    o_inter_triton = lasd_parallel_inter(
        q=q,
        o=o_inter_triton,
        states=global_states,
        ld=ld,
        cu_seqlens=None,
        reverse=False,
        trans=trans,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    # Get thresholds based on dtype
    atol, rtol = get_threshold(dtype)

    # Check forward pass results
    print(f"\nShape: {shape}, E: {e}, use_ld: {use_ld}, trans: {trans}, dtype: {dtype}")
    print("o diff max: ", torch.abs(o_inter_torch - o_inter_triton).max().item())
    print("o diff norm: ", torch.norm(o_inter_torch - o_inter_triton).item())
    assert torch.allclose(o_inter_torch, o_inter_triton, atol=atol, rtol=rtol)
