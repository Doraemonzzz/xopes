import pytest
import torch
import torch.nn.functional as F
import triton

from xopes.ops.lightning_attn.scalar_decay.lasd_parallel_triton import (
    lasd_parallel_state_parallel,
    lasd_parallel_state_reduce,
)
from xopes.ops.lightning_attn.scalar_decay.torch_utils import compute_states
from xopes.utils import get_threshold


def get_params():
    shapes = [
        # (4, 256, 12, 128, 256),
        # (1, 512, 16, 256, 512),
        # (2, 255, 7, 33, 63),
        # (2, 65, 7, 33, 63),
        # (2, 193, 7, 33, 63),
        (2, 1125, 8, 43, 33),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
# @pytest.mark.parametrize("use_ld", [True, False])
# @pytest.mark.parametrize("use_initial_state", [True, False])
# @pytest.mark.parametrize("reverse", [True, False])


@pytest.mark.parametrize("use_ld", [True])
@pytest.mark.parametrize("use_initial_state", [True])
@pytest.mark.parametrize("reverse", [True])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_compute_states(shape, use_ld, use_initial_state, reverse, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")

    # Generate input tensors
    b, n, h, d, e = shape

    k = torch.randn(b, n, h, d, dtype=dtype, device=device)
    v = torch.randn(b, n, h, e, dtype=dtype, device=device)

    ld = None
    if use_ld:
        ld = F.logsigmoid(torch.randn(h, device=device))

    initial_state = None
    if use_initial_state:
        initial_state = torch.randn(b, h, d, e, dtype=dtype, device=device)

    MAX_BLOCK_N = triton.next_power_of_2(n)
    MAX_BLOCK_C = MAX_BLOCK_N
    MAX_BLOCK_E = triton.next_power_of_2(e)
    MAX_BLOCK_D = triton.next_power_of_2(d)
    BLOCK_N = 64
    MAX_BLOCK_C = BLOCK_N

    # Get thresholds based on dtype
    atol, rtol = get_threshold(dtype)

    # Reference implementation
    local_states_ref, global_states_ref = compute_states(
        k=k, v=v, ld=ld, initial_state=initial_state, BLOCK_N=BLOCK_N, reverse=reverse
    )

    # Parallel implementation
    local_states = lasd_parallel_state_parallel(
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

    m = local_states_ref.shape[2]
    for i in range(m):
        print(
            i,
            "states diff norm: ",
            torch.norm(local_states_ref[:, :, i] - local_states[:, :, i]).item(),
        )
    print(
        "local_states diff max: ",
        torch.abs(local_states_ref - local_states[:, :, :-1]).max().item(),
    )
    print(
        "local_states diff norm: ",
        torch.norm(local_states_ref - local_states[:, :, :-1]).item(),
    )
    assert torch.allclose(
        local_states_ref, local_states[:, :, :-1], atol=atol, rtol=rtol
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
        reverse=reverse,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    l = global_states_ref.shape[2]
    for i in range(l):
        print(
            i,
            "states diff norm: ",
            torch.norm(global_states_ref[:, :, i] - global_states[:, :, i]).item(),
        )
    print(
        "states diff max: ", torch.abs(global_states_ref - global_states).max().item()
    )
    print("states diff norm: ", torch.norm(global_states_ref - global_states).item())

    # Assert results match within tolerance
    assert torch.allclose(global_states_ref, global_states, atol=atol, rtol=rtol)
