from typing import Optional, Tuple

import torch
import triton
from einops import repeat

from xopes.ops.cumsum import cumsum_fn
from xopes.ops.lightning_attn.scalar_decay.utils import (
    _lasd_parallel_inter,
    _lasd_parallel_intra,
    _lasd_parallel_intra_inter,
    _lasd_parallel_state_parallel,
    _lasd_parallel_state_parallel_reduce,
    _lasd_parallel_state_reduce,
)
from xopes.utils import contiguous
from xopes.utils.constant import SM_COUNT


@contiguous
def lasd_parallel_state_parallel(
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    MAX_BLOCK_N: int = 256,
    MAX_BLOCK_C: int = 256,
    MAX_BLOCK_E: int = 128,
    MAX_BLOCK_D: int = 128,
    BLOCK_N: int = 256,
):
    b, n, h, d = k.shape
    e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)
    use_pad = n % BLOCK_N != 0
    use_ld = ld is not None

    states = torch.empty(
        (b, h, NUM_BLOCK_N + 1, d, e), dtype=torch.float32, device=k.device
    )

    def grid_partial(MAX_BLOCK_D, MAX_BLOCK_E):
        def grid(meta):
            meta["BLOCK_D"] = min(meta["BLOCK_D"], MAX_BLOCK_D)
            meta["BLOCK_E"] = min(meta["BLOCK_E"], MAX_BLOCK_E)
            return (
                b * h * NUM_BLOCK_N,
                triton.cdiv(d, meta["BLOCK_D"]),
                triton.cdiv(e, meta["BLOCK_E"]),
            )

        return grid

    grid = grid_partial(MAX_BLOCK_D, MAX_BLOCK_E)

    _lasd_parallel_state_parallel[grid](
        K=k,
        V=v,
        STATES=states,
        LOG_DECAY=ld,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_LOG_DECAY=use_ld,
        USE_PAD=use_pad,
        REVERSE=reverse,
        BLOCK_N=BLOCK_N,
    )

    return states


@contiguous
def lasd_parallel_state_reduce(
    b: int,
    n: int,
    h: int,
    d: int,
    e: int,
    states: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    ld: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    MAX_BLOCK_N: int = 256,
    MAX_BLOCK_C: int = 256,
    MAX_BLOCK_E: int = 128,
    MAX_BLOCK_D: int = 128,
    BLOCK_N: int = 256,
):
    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_initial_state = initial_state is not None
    use_ld = ld is not None

    def grid_partial(MAX_BLOCK_D, MAX_BLOCK_E):
        def grid(meta):
            meta["BLOCK_D"] = min(meta["BLOCK_D"], MAX_BLOCK_D)
            meta["BLOCK_E"] = min(meta["BLOCK_E"], MAX_BLOCK_E)
            return (
                b * h,
                triton.cdiv(d, meta["BLOCK_D"]),
                triton.cdiv(e, meta["BLOCK_E"]),
            )

        return grid

    grid = grid_partial(MAX_BLOCK_D, MAX_BLOCK_E)

    _lasd_parallel_state_reduce[grid](
        STATE=initial_state,
        STATES=states,
        LOG_DECAY=ld,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_LOG_DECAY=use_ld,
        USE_INITIAL_STATE=use_initial_state,
        REVERSE=reverse,
        BLOCK_N=BLOCK_N,
    )

    return states


@contiguous
def lasd_parallel_state_parallel_reduce(
    k: torch.Tensor,
    v: torch.Tensor,
    b: int,
    n: int,
    h: int,
    d: int,
    e: int,
    initial_state: Optional[torch.Tensor] = None,
    ld: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    MAX_BLOCK_N: int = 256,
    MAX_BLOCK_C: int = 256,
    MAX_BLOCK_E: int = 128,
    MAX_BLOCK_D: int = 128,
    BLOCK_N: int = 256,
):
    b, n, h, d = k.shape
    e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)
    use_pad = n % BLOCK_N != 0
    use_initial_state = initial_state is not None
    use_ld = ld is not None

    states = torch.empty(
        (b, h, NUM_BLOCK_N + 1, d, e), dtype=torch.float32, device=k.device
    )

    def grid_partial(MAX_BLOCK_D, MAX_BLOCK_E):
        def grid(meta):
            meta["BLOCK_D"] = min(meta["BLOCK_D"], MAX_BLOCK_D)
            meta["BLOCK_E"] = min(meta["BLOCK_E"], MAX_BLOCK_E)
            return (
                b * h,
                triton.cdiv(d, meta["BLOCK_D"]),
                triton.cdiv(e, meta["BLOCK_E"]),
            )

        return grid

    grid = grid_partial(MAX_BLOCK_D, MAX_BLOCK_E)

    _lasd_parallel_state_parallel_reduce[grid](
        K=k,
        V=v,
        STATE=initial_state,
        STATES=states,
        LOG_DECAY=ld,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_LOG_DECAY=use_ld,
        USE_INITIAL_STATE=use_initial_state,
        USE_PAD=use_pad,
        REVERSE=reverse,
        BLOCK_N=BLOCK_N,
    )

    return states


@contiguous
def lasd_parallel_state_parallel_reduce_sep(
    k: torch.Tensor,
    v: torch.Tensor,
    b: int,
    n: int,
    h: int,
    d: int,
    e: int,
    initial_state: Optional[torch.Tensor] = None,
    ld: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    MAX_BLOCK_N: int = 256,
    MAX_BLOCK_C: int = 256,
    MAX_BLOCK_E: int = 128,
    MAX_BLOCK_D: int = 128,
    BLOCK_N: int = 256,
):
    # Step0.5: Compute local states in parallel
    states = lasd_parallel_state_parallel(
        k=k,
        v=v,
        ld=ld,
        cu_seqlens=cu_seqlens,
        reverse=reverse,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    # Step1: Update local states to get global states
    states = lasd_parallel_state_reduce(
        b=b,
        n=n,
        h=h,
        d=d,
        e=e,
        states=states,
        initial_state=initial_state,
        ld=ld,
        cu_seqlens=cu_seqlens,
        reverse=reverse,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    return states


##### intra and inter #####
@contiguous
def lasd_parallel_inter(
    q: torch.Tensor,
    o: torch.Tensor,
    states: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    trans: bool = False,
    MAX_BLOCK_N: int = 256,
    MAX_BLOCK_C: int = 256,
    MAX_BLOCK_E: int = 128,
    MAX_BLOCK_D: int = 128,
    BLOCK_N: int = 256,
):
    b, n, h, d = q.shape
    e = o.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)
    use_pad = n % BLOCK_N != 0

    use_ld = ld is not None

    def grid_partial(MAX_BLOCK_C, MAX_BLOCK_D, MAX_BLOCK_E):
        def grid(meta):
            meta["BLOCK_C"] = min(meta["BLOCK_C"], MAX_BLOCK_C)
            meta["BLOCK_D"] = min(meta["BLOCK_D"], MAX_BLOCK_D)
            meta["BLOCK_E"] = min(meta["BLOCK_E"], MAX_BLOCK_E)
            return (
                b * h * NUM_BLOCK_N,
                triton.cdiv(BLOCK_N, meta["BLOCK_C"]),
                triton.cdiv(e, meta["BLOCK_E"]),
            )

        return grid

    grid = grid_partial(MAX_BLOCK_C, MAX_BLOCK_D, MAX_BLOCK_E)

    _lasd_parallel_inter[grid](
        Q=q,
        O=o,
        STATES=states,
        LOG_DECAY=ld,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_LOG_DECAY=use_ld,
        REVERSE=reverse,
        TRANS=trans,
        BLOCK_N=BLOCK_N,
    )

    return o


@contiguous
def lasd_parallel_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    MAX_BLOCK_N: int = 256,
    MAX_BLOCK_C: int = 256,
    MAX_BLOCK_E: int = 128,
    MAX_BLOCK_D: int = 128,
    BLOCK_N: int = 256,
):
    b, n, h, d = q.shape
    e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_ld = ld is not None

    if use_cu_seqlens:
        o = torch.empty((1, n, h, e), dtype=q.dtype, device=q.device)
    else:
        o = torch.empty((b, n, h, e), dtype=q.dtype, device=q.device)

    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)
    use_pad = n % BLOCK_N != 0

    def grid_partial(MAX_BLOCK_C, MAX_BLOCK_E):
        def grid(meta):
            meta["BLOCK_C"] = min(meta["BLOCK_C"], MAX_BLOCK_C)
            meta["BLOCK_E"] = min(meta["BLOCK_E"], MAX_BLOCK_E)
            return (
                b * h * NUM_BLOCK_N,
                triton.cdiv(BLOCK_N, meta["BLOCK_C"]),
                triton.cdiv(e, meta["BLOCK_E"]),
            )

        return grid

    grid = grid_partial(MAX_BLOCK_C, MAX_BLOCK_E)

    _lasd_parallel_intra[grid](
        Q=q,
        K=k,
        V=v,
        O=o,
        LOG_DECAY=ld,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_LOG_DECAY=use_ld,
        REVERSE=reverse,
        BLOCK_N=BLOCK_N,
    )

    return o


@contiguous
def lasd_parallel_intra_inter(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    states: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    trans: bool = False,
    MAX_BLOCK_N: int = 256,
    MAX_BLOCK_C: int = 256,
    MAX_BLOCK_E: int = 128,
    MAX_BLOCK_D: int = 128,
    BLOCK_N: int = 256,
):
    b, n, h, d = q.shape
    e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)
    use_pad = n % BLOCK_N != 0

    use_ld = ld is not None

    if use_cu_seqlens:
        o = torch.empty((1, n, h, e), dtype=q.dtype, device=q.device)
    else:
        o = torch.empty((b, n, h, e), dtype=q.dtype, device=q.device)

    def grid_partial(MAX_BLOCK_C, MAX_BLOCK_D, MAX_BLOCK_E):
        def grid(meta):
            meta["BLOCK_C"] = min(meta["BLOCK_C"], MAX_BLOCK_C)
            meta["BLOCK_D"] = min(meta["BLOCK_D"], MAX_BLOCK_D)
            meta["BLOCK_E"] = min(meta["BLOCK_E"], MAX_BLOCK_E)
            return (
                b * h * NUM_BLOCK_N,
                triton.cdiv(BLOCK_N, meta["BLOCK_C"]),
                triton.cdiv(e, meta["BLOCK_E"]),
            )

        return grid

    grid = grid_partial(MAX_BLOCK_C, MAX_BLOCK_D, MAX_BLOCK_E)

    _lasd_parallel_intra_inter[grid](
        Q=q,
        K=k,
        V=v,
        O=o,
        STATES=states,
        LOG_DECAY=ld,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_LOG_DECAY=use_ld,
        REVERSE=reverse,
        TRANS=trans,
        BLOCK_N=BLOCK_N,
    )

    return o


########## Fwd start ##########
@contiguous
def lasd_parallel_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    trans: bool = False,
    use_chunk_loop: bool = False,
):
    """
    Forward pass for Lightning Attention with Scalar Decay in parallel mode.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        ld: Log decay tensor of shape (H,)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        reverse: Whether to process the sequence in reverse order
        trans: Whether to transpose the final output

    Returns:
        o: Output tensor of shape (B, N, H, E)
        states: Final state tensor
    """
    b, n, h, d = q.shape
    e = v.shape[-1]

    MAX_BLOCK_N = triton.next_power_of_2(n)
    MAX_BLOCK_C = MAX_BLOCK_N
    MAX_BLOCK_E = triton.next_power_of_2(e)
    MAX_BLOCK_D = triton.next_power_of_2(d)
    NUM_PARALLEL_BLOCKS = b * h

    if n <= 512:
        BLOCK_N = min(MAX_BLOCK_N, 128)
    else:
        BLOCK_N = 256

    # step1: compute states in parallel or chunk loop
    if NUM_PARALLEL_BLOCKS >= SM_COUNT or use_chunk_loop:
        fn = lasd_parallel_state_parallel_reduce
    else:
        fn = lasd_parallel_state_parallel_reduce_sep

    states = fn(
        k=k,
        v=v,
        b=b,
        n=n,
        h=h,
        d=d,
        e=e,
        initial_state=initial_state,
        ld=ld,
        cu_seqlens=cu_seqlens,
        reverse=reverse,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    # Step2: Compute intra and inter in parallel, for each chunk, parallel over sub-chunk
    o = lasd_parallel_intra_inter(
        q=q,
        k=k,
        v=v,
        states=states,
        ld=ld,
        cu_seqlens=cu_seqlens,
        reverse=reverse,
        trans=trans,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    return o, states


@contiguous
def lasd_parallel_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    dfinal_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    states: Optional[torch.Tensor] = None,
    use_chunk_loop: bool = False,
):
    """
    Backward pass for Lightning Attention with Scalar Decay in parallel mode.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        do: Gradient of output tensor of shape (B, N, H, E)
        ld: Log decay tensor of shape (H,)
        initial_state: Initial state tensor of shape (B, H, D, E)
        dfinal_state: Gradient of final state tensor
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        states: Cached states from forward pass (optional)

    Returns:
        dq: Gradient of query tensor
        dk: Gradient of key tensor
        dv: Gradient of value tensor
        dld: Gradient of log decay tensor
        dinitial_state: Gradient of initial state tensor
    """
    b, n, h, d = q.shape
    e = v.shape[-1]

    MAX_BLOCK_N = triton.next_power_of_2(n)
    MAX_BLOCK_C = MAX_BLOCK_N
    MAX_BLOCK_E = triton.next_power_of_2(e)
    MAX_BLOCK_D = triton.next_power_of_2(d)
    NUM_PARALLEL_BLOCKS = b * h

    if n <= 512:
        BLOCK_N = min(MAX_BLOCK_N, 128)
    else:
        BLOCK_N = 256

    """
    The following code is equivalent to this simple code:

    dq, states = lasd_parallel_fwd(
        q=do,  # b n h e
        k=v,  # b n h e
        v=k,  # b n h d
        ld=ld,
        initial_state=initial_state.transpose(-1, -2).contiguous(),
        cu_seqlens=cu_seqlens,
        trans=True,
    )

    dk, dstates = lasd_parallel_fwd(
        q=v,  # b n h e
        k=do,  # b n h e
        v=q,  # b n h d
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        reverse=True,
        trans=True,
    )

    dv, dstates = lasd_parallel_fwd(
        q=k,  # b n h d
        k=q,  # b n h d
        v=do,  # b n h e
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        reverse=True,
        trans=False,
    )
    """
    if NUM_PARALLEL_BLOCKS >= SM_COUNT or use_chunk_loop:
        fn = lasd_parallel_state_parallel_reduce
    else:
        fn = lasd_parallel_state_parallel_reduce_sep

    if states is None:
        states = fn(
            k=k,
            v=v,
            b=b,
            n=n,
            h=h,
            d=d,
            e=e,
            initial_state=initial_state,
            ld=ld,
            cu_seqlens=cu_seqlens,
            reverse=False,
            MAX_BLOCK_N=MAX_BLOCK_N,
            MAX_BLOCK_C=MAX_BLOCK_C,
            MAX_BLOCK_E=MAX_BLOCK_E,
            MAX_BLOCK_D=MAX_BLOCK_D,
            BLOCK_N=BLOCK_N,
        )

    dq = lasd_parallel_intra_inter(
        q=do,  # b n h e
        k=v,  # b n h e
        v=k,  # b n h d
        states=states,  # b h (m + 1) d e
        ld=ld,
        cu_seqlens=cu_seqlens,
        reverse=False,
        trans=True,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    final_state = states[:, :, -1, :, :]
    del states

    dstates = fn(
        k=q,  # b n h d
        v=do,  # b n h e
        b=b,
        n=n,
        h=h,
        d=d,
        e=e,
        initial_state=dfinal_state,
        ld=ld,
        cu_seqlens=cu_seqlens,
        reverse=True,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    dk = lasd_parallel_intra_inter(
        q=v,  # b n h e
        k=do,  # b n h e
        v=q,  # b n h d
        states=dstates,  # b h (m + 1) d e
        ld=ld,
        cu_seqlens=cu_seqlens,
        reverse=True,
        trans=True,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    dv = lasd_parallel_intra_inter(
        q=k,  # b n h d
        k=q,  # b n h d
        v=do,  # b n h e
        states=dstates,  # b h (m + 1) d e
        ld=ld,
        cu_seqlens=cu_seqlens,
        reverse=True,
        trans=False,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    need_dfinal_state = (
        dfinal_state is not None
        and initial_state is not None
        and initial_state.requires_grad
    )

    return (
        dq,
        dk,
        dv,
        dstates[:, :, -1, :, :] if need_dfinal_state else None,
        final_state,
    )


class LasdParallelFunction(torch.autograd.Function):
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
        output, states = lasd_parallel_fwd(
            q=q,
            k=k,
            v=v,
            ld=ld,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            use_chunk_loop=use_chunk_loop,
        )

        # Save tensors needed for backward
        final_state = states[:, :, -1, :, :]
        if not save_states:
            states = None
        ctx.save_for_backward(q, k, v, ld, initial_state, cu_seqlens, states)
        ctx.use_chunk_loop = use_chunk_loop
        del states

        return output, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dfinal_state):
        q, k, v, ld, initial_state, cu_seqlens, states = ctx.saved_tensors
        use_chunk_loop = ctx.use_chunk_loop
        dq, dk, dv, dinitial_state, final_state = lasd_parallel_bwd(
            q=q,
            k=k,
            v=v,
            do=do,
            ld=ld,
            initial_state=initial_state,
            dfinal_state=dfinal_state,
            cu_seqlens=cu_seqlens,
            states=states,
            use_chunk_loop=use_chunk_loop,
        )

        if ld is not None and ld.requires_grad:
            # b n h d -> n h
            dld = (q * dq - k * dk).sum(-1).sum(0)
            dld = cumsum_fn(dld, dim=0, reverse=True).sum(0)

            if dfinal_state is not None:
                n = q.shape[1]
                # !!! important, the following line is equivalent to the following line
                # dld = cumsum_fn(dld, dim=0, reverse=True)
                # dld.add_((final_state * dfinal_state).sum(-1).sum(-1).sum(0))
                # dld = dld.sum(0)
                dld.add_((final_state * dfinal_state).sum(-1).sum(-1).sum(0) * n)
        else:
            dld = None

        return (dq, dk, dv, dld, dinitial_state, None, None, None)


def lasd_parallel_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    save_states: bool = True,
    use_chunk_loop: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Lightning Attention Parallel with Scalar Decay in Triton.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (H,)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training
        save_states: Whether to save the states
        use_chunk_loop: Whether to use chunk loop

    Returns:
        output: Tensor of shape (B, N, H, E)
        state: Tensor of shape (B, H, D, E)
    """
    b = q.shape[0]
    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1
    if initial_state is not None:
        initial_state = initial_state.squeeze(0)
        # treat for varlen training
        if len(initial_state.shape) == 3:
            initial_state = repeat(initial_state, "h d e -> b h d e", b=b).contiguous()

    return LasdParallelFunction.apply(
        q,
        k,
        v,
        ld,
        initial_state,
        cu_seqlens,
        save_states,
        use_chunk_loop,
    )


if __name__ == "__main__":
    import torch.nn.functional as F

    b, n, h, d = 2, 16, 12, 64
    e = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    q = torch.randn(b, n, h, d, device=device, dtype=dtype).requires_grad_(True)
    k = torch.randn(b, n, h, d, device=device, dtype=dtype).requires_grad_(True)
    v = torch.randn(b, n, h, e, device=device, dtype=dtype).requires_grad_(True)
    ld = F.logsigmoid(torch.randn(h, device=device))
    initial_state = torch.randn(b, h, d, e, device=device, dtype=dtype).requires_grad_(
        True
    )
    output, final_state = lasd_parallel_triton(q, k, v, ld, initial_state)
    loss = output.sum() + final_state.sum()
    loss.backward()
