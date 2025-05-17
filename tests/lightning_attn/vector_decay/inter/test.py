import pytest
import torch
import torch.nn.functional as F
import triton

from xopes.ops.lightning_attn.vector_decay.lavd_parallel_triton import (
    lavd_parallel_inter,
    lavd_parallel_state_parallel,
    lavd_parallel_state_reduce,
)
from xopes.ops.lightning_attn.vector_decay.torch_utils import lavd_inter_torch
from xopes.utils import assert_close, get_threshold


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
        # LARGE D, E
        (2, 1125, 8, 255, 257),
        (2, 1025, 8, 255, 257),
        # Train shape
        (8, 2048, 12, 64, 64),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_initial_state", [True, False])
@pytest.mark.parametrize("use_ldk", [True])
@pytest.mark.parametrize("use_ldv", [True])
@pytest.mark.parametrize("share_k", [True, False])
@pytest.mark.parametrize("share_v", [True, False])
@pytest.mark.parametrize("use_varlen", [False])
@pytest.mark.parametrize("trans", [True, False])
@pytest.mark.parametrize("c", [10])
@pytest.mark.parametrize("dtype", [torch.float32])
def test(
    shape,
    use_initial_state,
    use_ldk,
    use_ldv,
    share_k,
    share_v,
    use_varlen,
    trans,
    c,
    dtype,
):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    scale = 0.01

    # Generate input tensors
    b, n, h, d, e = shape

    if not use_ldk and not use_ldv:
        return

    reverse = False
    q = torch.randn(b, n, h, d, dtype=dtype, device=device)
    if share_k:
        use_ldk = True
        ldk = F.logsigmoid(
            (1 + scale * torch.randn(b, n, h, d, dtype=dtype, device=device)) * c
        ).requires_grad_()
        k = None
    else:
        k = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()

        if use_ldk:
            ldk = F.logsigmoid(
                (1 + scale * torch.randn(b, n, h, d, dtype=dtype, device=device)) * c
            ).requires_grad_()
        else:
            ldk = None

    if share_v:
        use_ldv = True
        ldv = F.logsigmoid(
            (1 + scale * torch.randn(b, n, h, e, dtype=dtype, device=device)) * c
        ).requires_grad_()
        v = None
    else:
        v = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()

        if use_ldv:
            ldv = F.logsigmoid(
                (1 + scale * torch.randn(b, n, h, e, dtype=dtype, device=device)) * c
            ).requires_grad_()
        else:
            ldv = None

    initial_state = None
    if use_initial_state:
        initial_state = torch.randn(b, h, d, e, dtype=dtype, device=device)

    # Calculate block sizes
    MAX_BLOCK_N = triton.next_power_of_2(n)
    MAX_BLOCK_C = MAX_BLOCK_N
    MAX_BLOCK_E = triton.next_power_of_2(e)
    MAX_BLOCK_D = triton.next_power_of_2(d)
    BLOCK_N = 128
    MAX_BLOCK_C = BLOCK_N

    # Compute using reference implementation
    o_inter_torch = lavd_inter_torch(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        initial_state=initial_state,
        cu_seqlens=None,
        BLOCK_N=BLOCK_N,
    )

    # Compute using Triton implementation
    local_states = lavd_parallel_state_parallel(
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        initial_state=initial_state,
        cu_seqlens=None,
        reverse=reverse,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    global_states = lavd_parallel_state_reduce(
        b=b,
        n=n,
        h=h,
        d=d,
        e=e,
        states=local_states.contiguous(),
        ldk=ldk,
        ldv=ldv,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        initial_state=initial_state,
        cu_seqlens=None,
        reverse=reverse,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    if trans:
        global_states = torch.transpose(global_states, -1, -2).contiguous()

    o_inter_triton = torch.zeros_like(o_inter_torch)

    o_inter_triton = lavd_parallel_inter(
        q=q,
        o=o_inter_triton,
        states=global_states,
        ldk=ldk,
        ldv=ldv,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        cu_seqlens=None,
        reverse=reverse,
        trans=trans,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    c = 16
    m = (n + c - 1) // c
    for i in range(m):
        start = i * c
        end = min(start + c, n)
        print(
            i,
            torch.norm(
                o_inter_torch[:, start:end, :, :] - o_inter_triton[:, start:end, :, :]
            ).item(),
        )

    # Get thresholds based on dtype
    atol, rtol = get_threshold(dtype)

    # Check forward pass results
    print(
        f"\nShape: {shape}, E: {e}, use_initial_state: {use_initial_state}, trans: {trans}, dtype: {dtype}"
    )
    print("o diff max: ", torch.abs(o_inter_torch - o_inter_triton).max().item())
    print("o diff norm: ", torch.norm(o_inter_torch - o_inter_triton).item())

    assert_close(o_inter_torch, o_inter_triton, atol, rtol)
