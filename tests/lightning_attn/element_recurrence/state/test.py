import pytest
import torch
import torch.nn.functional as F

from xopes.ops.cumsum.chunk_cumsum_decay import chunk_cumsum_decay_triton
from xopes.ops.lightning_attn.element_recurrence.laer_parallel_triton import (
    laer_parallel_state_parallel,
)
from xopes.ops.lightning_attn.element_recurrence.torch_utils import compute_states
from xopes.utils import get_threshold, print_diff


def get_params():
    shapes = [
        # standard shape
        (2, 256, 128),
        (2, 1024, 32),
        # BLOCK_N +- 1
        (2, 257, 64),
        (2, 255, 64),
        (2, 65, 33),
        # BLOCK_N +- C
        (2, 270, 64),
        (2, 270, 33),
        (2, 1125, 43),
        # LARGE D
        (2, 512, 255),
        (2, 128, 257),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("c", [-10, 1, 10])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_laer_compute_states(shape, reverse, c, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")

    # Generate input tensors
    b, n, d = shape

    k = torch.randn(b, n, d, dtype=dtype, device=device)
    v = torch.randn(b, n, d, dtype=dtype, device=device)

    # Always use log decay
    ld = F.logsigmoid(torch.randn(b, n, d, device=device)) * c
    BLOCK_N = 64

    # Get thresholds based on dtype
    atol, rtol = get_threshold(dtype)

    # Reference implementation
    local_states_ref = compute_states(k=k, v=v, ld=ld, BLOCK_N=BLOCK_N, reverse=reverse)
    ld_cumsum_ref = chunk_cumsum_decay_triton(ld, reverse=reverse, chunk_size=BLOCK_N)

    # Parallel implementation
    local_states, ld_cumsum = laer_parallel_state_parallel(
        k=k,
        v=v,
        ld=ld,
        reverse=reverse,
        BLOCK_N=BLOCK_N,
    )

    print_diff(local_states_ref, local_states, n, BLOCK=BLOCK_N)
    print(
        "local_states diff max: ",
        torch.abs(local_states_ref - local_states).max().item(),
    )
    print(
        "local_states diff norm: ",
        torch.norm(local_states_ref - local_states).item(),
    )
    assert torch.allclose(local_states_ref, local_states, atol=atol, rtol=rtol)

    print_diff(ld_cumsum_ref, ld_cumsum, n, BLOCK=BLOCK_N)
    print(
        "ld_cumsum diff max: ",
        torch.abs(ld_cumsum_ref - ld_cumsum).max().item(),
    )
    print(
        "ld_cumsum diff norm: ",
        torch.norm(ld_cumsum_ref - ld_cumsum).item(),
    )
    assert torch.allclose(ld_cumsum_ref, ld_cumsum, atol=atol, rtol=rtol)
