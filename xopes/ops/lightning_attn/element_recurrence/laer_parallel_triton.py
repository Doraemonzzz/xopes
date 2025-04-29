from typing import Optional, Tuple

import torch
import triton
from einops import repeat

from xopes.ops.lightning_attn.element_recurrence.utils import (
    _laer_parallel_intra_inter_bwd,
    _laer_parallel_intra_inter_fwd,
    _laer_parallel_state_parallel,
)
from xopes.utils import contiguous

BLOCK_N = 16


@contiguous
def laer_parallel_state_parallel(
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    BLOCK_N: int = 256,
):
    b, n, d = k.shape

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_pad = n % BLOCK_N != 0

    states = torch.empty((b, n, d), dtype=k.dtype, device=k.device)
    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)

    def grid(meta):
        return (
            b,
            triton.cdiv(n, BLOCK_N),
            triton.cdiv(d, meta["BLOCK_D"]),
        )

    ld_cumsum = torch.empty((b, n, d), dtype=torch.float32, device=k.device)

    _laer_parallel_state_parallel[grid](
        K=k,
        V=v,
        STATES=states,
        LOG_DECAY=ld,
        LOG_DECAY_CUMSUM=ld_cumsum,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        D=d,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_PAD=use_pad,
        REVERSE=reverse,
        BLOCK_N=BLOCK_N,
        NUM_BLOCK_N=NUM_BLOCK_N,
    )

    return states, ld_cumsum


@contiguous
def laer_parallel_intra_inter_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    states: torch.Tensor,
    ld: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    ld_cumsum: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    BLOCK_N: int = 256,
):
    b, n, d = q.shape

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)
    use_pad = n % BLOCK_N != 0
    use_initial_state = initial_state is not None

    if use_cu_seqlens:
        o = torch.empty((1, n, d), dtype=q.dtype, device=q.device)
    else:
        o = torch.empty((b, n, d), dtype=q.dtype, device=q.device)

    def grid(meta):
        return (
            b,
            triton.cdiv(d, meta["BLOCK_D"]),
        )

    _laer_parallel_intra_inter_fwd[grid](
        Q=q,
        K=k,
        V=v,
        O=o,
        STATE=initial_state,
        STATES=states,
        LOG_DECAY=ld,
        LOG_DECAY_CUMSUM=ld_cumsum,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        D=d,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        BLOCK_N=BLOCK_N,
        NUM_BLOCK_N=NUM_BLOCK_N,
    )

    return o, states


@contiguous
def laer_parallel_intra_inter_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    states: torch.Tensor,
    dstates: torch.Tensor,
    ld: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    dfinal_state: Optional[torch.Tensor] = None,
    ld_reverse_cumsum: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    BLOCK_N: int = 256,
):
    b, n, d = q.shape

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)
    use_pad = n % BLOCK_N != 0
    use_initial_state = initial_state is not None
    use_dfinal_state = dfinal_state is not None

    if use_cu_seqlens:
        dq = torch.empty((1, n, d), dtype=q.dtype, device=q.device)
        dk = torch.empty((1, n, d), dtype=k.dtype, device=k.device)
        dv = torch.empty((1, n, d), dtype=v.dtype, device=v.device)
        dld = torch.empty((1, n, d), dtype=ld.dtype, device=ld.device)
    else:
        dq = torch.empty((b, n, d), dtype=q.dtype, device=q.device)
        dk = torch.empty((b, n, d), dtype=k.dtype, device=k.device)
        dv = torch.empty((b, n, d), dtype=v.dtype, device=v.device)
        dld = torch.empty((b, n, d), dtype=ld.dtype, device=ld.device)

    def grid(meta):
        return (
            b,
            triton.cdiv(d, meta["BLOCK_D"]),
        )

    _laer_parallel_intra_inter_bwd[grid](
        Q=q,
        K=k,
        V=v,
        DO=do,
        DQ=dq,
        DK=dk,
        DV=dv,
        DLD=dld,
        STATE=initial_state,
        DSTATE=dfinal_state,
        STATES=states,
        DSTATES=dstates,
        LOG_DECAY=ld,
        LOG_DECAY_REVERSE_CUMSUM=ld_reverse_cumsum,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        D=d,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        USE_DFINAL_STATE=use_dfinal_state,
        BLOCK_N=BLOCK_N,
        NUM_BLOCK_N=NUM_BLOCK_N,
    )

    return dq, dk, dv, dld, dstates


########## Fwd start ##########
@contiguous
def laer_parallel_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
):
    """
    Forward pass for Lightning Attention with Data-Dependent Scalar Decay in parallel mode.

    Args:
        q: Query tensor of shape (B, N, D)
        k: Key tensor of shape (B, N, D)
        v: Value tensor of shape (B, N, D)
        ld: Log decay tensor of shape (B, N, D) - data dependent decay factors
        initial_state: Initial state tensor of shape (B, D)
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        reverse: Whether to process the sequence in reverse order
        trans: Whether to transpose the final output
        use_chunk_loop: Whether to use chunk loop

    Returns:
        o: Output tensor of shape (B, N, H, E)
        states: Final state tensor
    """
    b, n, d = q.shape

    # Step1: Compute states in parallel or chunk loop
    states, ld_cumsum = laer_parallel_state_parallel(
        k=k,
        v=v,
        ld=ld,
        cu_seqlens=cu_seqlens,
        reverse=reverse,
        BLOCK_N=BLOCK_N,
    )

    # Step2: Compute intra and inter in parallel, for each chunk, parallel over sub-chunk
    o, states = laer_parallel_intra_inter_fwd(
        q=q,
        k=k,
        v=v,
        states=states,
        initial_state=initial_state,
        ld=ld,
        ld_cumsum=ld_cumsum,
        cu_seqlens=cu_seqlens,
        BLOCK_N=BLOCK_N,
    )

    return o, states


@contiguous
def laer_parallel_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    dfinal_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    states: Optional[torch.Tensor] = None,
):
    """
    Backward pass for Lightning Attention with Data-Dependent Scalar Decay in parallel mode.

    Args:
        q: Query tensor of shape (B, N, D)
        k: Key tensor of shape (B, N, D)
        v: Value tensor of shape (B, N, D)
        do: Gradient of output tensor of shape (B, N, D)
        ld: Log decay tensor of shape (B, N, D) - data dependent decay factors
        initial_state: Initial state tensor of shape (B, D)
        dfinal_state: Gradient of final state tensor
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        states: Cached states from forward pass (optional)
        ld_cumsum: Cached ld_cumsum from forward pass (optional)

    Returns:
        dq: Gradient of query tensor
        dk: Gradient of key tensor
        dv: Gradient of value tensor
        dld: Gradient of log decay tensor
        dinitial_state: Gradient of initial state tensor
    """
    b, n, d = q.shape

    # Compute dstates for dk and dv
    dstates, ld_reverse_cumsum = laer_parallel_state_parallel(
        k=q,  # b n d
        v=do,  # b n d
        ld=ld,  # b n d
        cu_seqlens=cu_seqlens,
        reverse=True,
        BLOCK_N=BLOCK_N,
    )

    dq, dk, dv, dld, dstates = laer_parallel_intra_inter_bwd(
        q=q,  # b n d
        k=k,  # b n d
        v=v,  # b n d
        do=do,  # b n d
        states=states,  # b n d
        dstates=dstates,  # b n d
        ld=ld,
        ld_reverse_cumsum=ld_reverse_cumsum,
        initial_state=initial_state,
        dfinal_state=dfinal_state,
        cu_seqlens=cu_seqlens,
        BLOCK_N=BLOCK_N,
    )

    # Compute gradient for initial state if needed
    need_dfinal_state = (
        dfinal_state is not None
        and initial_state is not None
        and initial_state.requires_grad
    )

    return dq, dk, dv, dld, dstates[:, -1] if need_dfinal_state else None


class LaerParallelFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        ld,
        initial_state=None,
        cu_seqlens=None,
        save_states=True,
        use_chunk_loop=False,
    ):
        # Forward computation
        output, states = laer_parallel_fwd(
            q=q,
            k=k,
            v=v,
            ld=ld,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

        # Save tensors needed for backward
        final_state = states[:, -1]
        ctx.save_for_backward(q, k, v, ld, initial_state, cu_seqlens, states)
        del states

        return output, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dfinal_state):
        q, k, v, ld, initial_state, cu_seqlens, states = ctx.saved_tensors
        dq, dk, dv, dld, dinitial_state = laer_parallel_bwd(
            q=q,
            k=k,
            v=v,
            do=do,
            ld=ld,
            initial_state=initial_state,
            dfinal_state=dfinal_state,
            cu_seqlens=cu_seqlens,
            states=states,
        )

        return (
            dq,
            dk,
            dv,
            dld,
            dinitial_state,
            None,
            None,
            None,
        )


def laer_parallel_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    save_states: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Lightning Attention Parallel with Data-Dependent Scalar Decay in Triton.

    Args:
        q: Query tensor of shape (B, N, D)
        k: Key tensor of shape (B, N, D)
        v: Value tensor of shape (B, N, D)
        ld: Logarithmic decay tensor of shape (B, N, D) - data dependent decay factors
        initial_state: Initial state tensor of shape (B, D)
        save_states: Whether to save states for backward
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        output: Tensor of shape (B, N, D)
        state: Tensor of shape (B, D)
    """
    b = q.shape[0]
    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1
    if initial_state is not None:
        initial_state = initial_state.squeeze(0)
        # treat for varlen training
        if len(initial_state.shape) == 1:
            initial_state = repeat(initial_state, "d -> b d", b=b)

    return LaerParallelFunction.apply(
        q,
        k,
        v,
        ld,
        initial_state,
        cu_seqlens,
        save_states,
    )


if __name__ == "__main__":
    import torch.nn.functional as F

    b, n, d = 2, 16, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    q = torch.randn(b, n, d, device=device, dtype=dtype).requires_grad_(True)
    k = torch.randn(b, n, d, device=device, dtype=dtype).requires_grad_(True)
    v = torch.randn(b, n, d, device=device, dtype=dtype).requires_grad_(True)
    # Data-dependent decay factors
    ld = F.logsigmoid(torch.randn(b, n, d, device=device))
    initial_state = torch.randn(b, d, device=device, dtype=dtype).requires_grad_(True)
    output, final_state = laer_parallel_triton(q, k, v, ld, initial_state)
    loss = output.sum() + final_state.sum()
    loss.backward()
