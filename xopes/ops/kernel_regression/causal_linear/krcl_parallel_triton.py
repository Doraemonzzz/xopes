# krcl: kernel regression with causal linear
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton

from xopes.ops.cumsum import chunk_cumsum_decay_fn
from xopes.ops.kernel_regression.causal_linear.utils import (
    _krcl_parallel_chunk_loop,
    _krcl_parallel_intra_inter,
    _krcl_parallel_inverse,
)
from xopes.utils import contiguous
from xopes.utils.constant import XOPES_DEBUG


@contiguous
def krcl_parallel_inverse(
    q: torch.Tensor,
    k: torch.Tensor,
    ld: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    ld_cumsum: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    BLOCK_N: int = 128,
    **kwargs,
):
    b, n, h, d = k.shape
    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_q = q is not None
    use_alpha = alpha is not None
    use_beta = beta is not None

    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)
    inv = torch.empty(
        (b, h, NUM_BLOCK_N, BLOCK_N, BLOCK_N), device=k.device, dtype=k.dtype
    )

    def grid(meta):
        return (b, h, NUM_BLOCK_N)

    if ld_cumsum is None:
        ld_cumsum = chunk_cumsum_decay_fn(
            ld, reverse=reverse, chunk_size=BLOCK_N, use_offset=False
        )

    _krcl_parallel_inverse[grid](
        Q=q,
        K=k,
        INV=inv,
        LOG_DECAY=ld_cumsum,
        ALPHA=alpha,
        BETA=beta,
        USE_Q=use_q,
        USE_ALPHA=use_alpha,
        USE_BETA=use_beta,
        B=b,
        N=n,
        H=h,
        D=d,
        CU_SEQLENS=cu_seqlens,
        USE_CU_SEQLENS=use_cu_seqlens,
        REVERSE=reverse,
        BLOCK_N=BLOCK_N,
        NUM_BLOCK_N=NUM_BLOCK_N,
    )

    return inv


@contiguous
def krcl_parallel_chunk_loop(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    inv: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    ld_cumsum: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    save_states: bool = False,
    BLOCK_N: int = 128,
    **kwargs,
):
    b, n, h, d = k.shape
    e = v.shape[-1]
    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_q = q is not None
    use_alpha = alpha is not None
    use_beta = beta is not None
    use_initial_state = initial_state is not None
    use_pad = n % BLOCK_N != 0
    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)

    o = torch.empty((b, n, h, e), device=k.device, dtype=k.dtype)
    if save_states:
        states = torch.empty((b, h, NUM_BLOCK_N, d, e), dtype=k.dtype, device=k.device)
    else:
        states = None
    final_state = torch.empty((b, h, d, e), device=k.device, dtype=k.dtype)

    def grid(meta):
        return (b * h, triton.cdiv(e, meta["BLOCK_E"]))

    if ld_cumsum is None:
        ld_cumsum = chunk_cumsum_decay_fn(
            ld, reverse=reverse, chunk_size=BLOCK_N, use_offset=False
        )

    _krcl_parallel_chunk_loop[grid](
        Q=q,
        K=k,
        V=v,
        O=o,
        INV=inv,
        LOG_DECAY=ld_cumsum,
        ALPHA=alpha,
        BETA=beta,
        INITIAL_STATE=initial_state,
        FINAL_STATE=final_state,
        STATES=states,
        USE_Q=use_q,
        USE_ALPHA=use_alpha,
        USE_BETA=use_beta,
        USE_INITIAL_STATE=use_initial_state,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_PAD=use_pad,
        REVERSE=reverse,
        SAVE_STATES=save_states,
        BLOCK_N=BLOCK_N,
        NUM_BLOCK_N=NUM_BLOCK_N,
    )

    return o, states, final_state


@contiguous
def krcl_parallel_intra_inter(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    states: torch.Tensor,
    ld: torch.Tensor,
    ld_cumsum: Optional[torch.Tensor] = None,
    ld_reverse_cumsum: Optional[torch.Tensor] = None,
    x: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_q: bool = True,
    compute_dq: bool = False,
    share_qk: bool = False,
    reverse: bool = False,
    trans: bool = False,
    BLOCK_N: int = 128,
    **kwargs,
):
    b, n, h, d = k.shape
    e = v.shape[-1]
    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_alpha = alpha is not None
    use_beta = beta is not None
    initial_state is not None
    use_pad = n % BLOCK_N != 0
    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)

    if XOPES_DEBUG:
        BLOCK_D = 32
        BLOCK_E = 32
    else:
        BLOCK_D = min(MAX_BLOCK_D, 128)
        BLOCK_E = min(MAX_BLOCK_E, 128)

    if use_cu_seqlens:
        o = torch.empty((1, n, h, e), dtype=q.dtype, device=q.device)
    else:
        o = torch.empty((b, n, h, e), dtype=q.dtype, device=q.device)

    NUM_BLOCK_D = triton.cdiv(d, BLOCK_D)
    NUM_BLOCK_E = triton.cdiv(e, BLOCK_E)
    dld = torch.empty((b, n, h, NUM_BLOCK_E), dtype=torch.float32, device=q.device)

    def grid(meta):
        return (
            b * h,
            NUM_BLOCK_N,
            NUM_BLOCK_E,
        )

    if ld_cumsum is None:
        ld_cumsum = chunk_cumsum_decay_fn(ld, reverse=False, chunk_size=BLOCK_N)

    if ld_reverse_cumsum is None and reverse:
        ld_reverse_cumsum = chunk_cumsum_decay_fn(
            ld, reverse=True, chunk_size=BLOCK_N, use_offset=False
        )

    _krcl_parallel_intra_inter[grid](
        Q=q,
        K=k,
        V=v,
        O=o,
        ALPHA=alpha,
        BETA=beta,
        STATES=states,
        LOG_DECAY=ld_cumsum,
        LOG_DECAY_REVERSE=ld_reverse_cumsum,
        X=x,
        DLOG_DECAY=dld,
        CU_SEQLENS=cu_seqlens,
        USE_Q=use_q,
        USE_ALPHA=use_alpha,
        USE_BETA=use_beta,
        COMPUTE_DQ=compute_dq,
        SHARE_QK=share_qk,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_CU_SEQLENS=use_cu_seqlens,
        REVERSE=reverse,
        TRANS=trans,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_E=BLOCK_E,
        NUM_BLOCK_N=NUM_BLOCK_N,
        NUM_BLOCK_D=NUM_BLOCK_D,
        NUM_BLOCK_E=NUM_BLOCK_E,
    )

    return o, dld


########## Fwd start ##########
@contiguous
def krcl_parallel_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    BLOCK_N: int = 128,
    **kwargs,
):
    ld_cumsum = chunk_cumsum_decay_fn(ld, reverse=False, chunk_size=BLOCK_N)

    inv = krcl_parallel_inverse(
        q=q,
        k=k,
        ld=ld,
        alpha=alpha,
        beta=beta,
        ld_cumsum=ld_cumsum,
        cu_seqlens=cu_seqlens,
        reverse=False,
        BLOCK_N=BLOCK_N,
    )

    o, _, final_state = krcl_parallel_chunk_loop(
        q=q,
        k=k,
        v=v,
        ld=ld,
        inv=inv,
        alpha=alpha,
        beta=beta,
        initial_state=initial_state,
        ld_cumsum=ld_cumsum,
        cu_seqlens=cu_seqlens,
        reverse=False,
        save_states=False,
        BLOCK_N=BLOCK_N,
    )

    return inv, o, _, final_state


########## Bwd start ##########
def krcl_parallel_bwd(
    inv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    ld: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    dfinal_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    BLOCK_N: int = 128,
    **kwargs,
):
    ld_cumsum = chunk_cumsum_decay_fn(ld, reverse=False, chunk_size=BLOCK_N)
    ld_reverse_cumsum = chunk_cumsum_decay_fn(
        ld, reverse=True, chunk_size=BLOCK_N, use_offset=True
    )

    o, states, _ = krcl_parallel_chunk_loop(
        q=q,
        k=k,
        v=v,
        ld=ld,
        inv=inv,
        alpha=alpha,
        beta=beta,
        initial_state=initial_state,
        ld_cumsum=ld_cumsum,
        cu_seqlens=cu_seqlens,
        reverse=False,
        save_states=True,
        BLOCK_N=BLOCK_N,
    )

    dv, dstates, _ = krcl_parallel_chunk_loop(
        q=k if q is not None else None,
        k=q if q is not None else k,
        v=do,
        ld=ld,
        inv=inv,
        alpha=alpha,
        beta=beta,
        initial_state=dfinal_state,
        ld_cumsum=ld_reverse_cumsum,
        cu_seqlens=cu_seqlens,
        reverse=True,
        save_states=True,
        BLOCK_N=BLOCK_N,
    )

    use_q = q is not None
    dk, dld_k = krcl_parallel_intra_inter(
        q=o,
        k=dv,
        v=q,
        states=dstates,
        ld=ld,
        ld_cumsum=ld_cumsum,
        ld_reverse_cumsum=ld_reverse_cumsum,
        x=k,
        alpha=alpha,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        use_q=use_q,
        compute_dq=False,
        share_qk=False,
        reverse=True,
        trans=True,
        BLOCK_N=BLOCK_N,
    )

    dq, dld_q = krcl_parallel_intra_inter(
        q=dv,
        k=o,
        v=k,
        o=dk if q is None else None,
        states=states,
        ld=ld,
        ld_cumsum=ld_cumsum,
        ld_reverse_cumsum=ld_reverse_cumsum,
        x=q,
        alpha=alpha,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        use_q=use_q,
        compute_dq=True,
        share_qk=q is None,
        reverse=False,
        trans=True,
        BLOCK_N=BLOCK_N,
    )

    if not use_q:
        dk = dq
        dq = None

    dld = torch.empty_like(ld)
    dalpha = torch.empty_like(alpha) if alpha is not None else None
    dbeta = torch.empty_like(beta) if beta is not None else None
    dinitial_state = (
        torch.empty_like(initial_state) if initial_state is not None else None
    )

    return dq, dk, dv, dld, dalpha, dbeta, dinitial_state


class KrclParallelFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        ld,
        alpha,
        beta,
        initial_state=None,
        cu_seqlens=None,
        BLOCK_N: int = 128,
    ):
        # Forward computation
        inv, o, _, final_state = krcl_parallel_fwd(
            q=q,
            k=k,
            v=v,
            ld=ld,
            alpha=alpha,
            beta=beta,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            BLOCK_N=BLOCK_N,
        )

        # Save tensors needed for backward
        ctx.save_for_backward(
            q, k, v, ld, alpha, beta, inv, initial_state, final_state, cu_seqlens
        )
        ctx.BLOCK_N = BLOCK_N

        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dfinal_state):
        (
            q,
            k,
            v,
            ld,
            alpha,
            beta,
            inv,
            initial_state,
            final_state,
            cu_seqlens,
        ) = ctx.saved_tensors
        BLOCK_N = ctx.BLOCK_N

        dq, dk, dv, dld, dalpha, dbeta, dinitial_state = krcl_parallel_bwd(
            inv=inv,
            q=q,
            k=k,
            v=v,
            do=do,
            ld=ld,
            alpha=alpha,
            beta=beta,
            initial_state=initial_state,
            final_state=final_state,
            dfinal_state=dfinal_state,
            cu_seqlens=cu_seqlens,
            BLOCK_N=BLOCK_N,
        )

        return (dq, dk, dv, dld, dalpha, dbeta, dinitial_state, None, None, None)


def krcl_parallel_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    BLOCK_N: int = 128,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Kernel Regression with Causal Linear in Pytorch.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (B, N, H)
        alpha: Alpha tensor of shape (B, N, H)
        beta: Beta tensor of shape (B, N, H)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training
        BLOCK_N: Block size for parallelization

    Returns:
        o: Tensor of shape (B, N, H, E)
        state: Tensor of shape (B, H, D, E)
    """
    return KrclParallelFunction.apply(
        q, k, v, ld, alpha, beta, initial_state, cu_seqlens, BLOCK_N
    )


if __name__ == "__main__":
    b, n, h, d = 2, 16, 12, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    q = torch.randn(b, n, h, d, device=device, dtype=dtype)
    k = torch.randn(b, n, h, d, device=device, dtype=dtype)
    v = torch.randn(b, n, h, d, device=device, dtype=dtype)
    ld = F.logsigmoid(torch.randn(b, n, h, device=device))
    o, state = krcl_parallel_triton(q, k, v, ld)
