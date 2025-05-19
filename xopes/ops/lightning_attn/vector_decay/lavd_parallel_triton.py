from typing import Optional, Tuple

import torch
import triton
from einops import repeat

from xopes.ops.cumsum import chunk_cumsum_decay_fn
from xopes.ops.lightning_attn.log_decay import compute_dld_with_cumsum_fn
from xopes.ops.lightning_attn.vector_decay.utils import (
    BLOCK_C,
    _lavd_parallel_inter,
    _lavd_parallel_intra,
    _lavd_parallel_intra_inter_no_loop,
    _lavd_parallel_state_parallel,
    _lavd_parallel_state_parallel_reduce,
    _lavd_parallel_state_reduce,
    _lavd_parallel_sub_intra,
    _lavd_parallel_sub_intra_attn,
    _lavd_parallel_sub_intra_o,
)
from xopes.utils import contiguous
from xopes.utils.constant import MIN_BLOCK, SM_COUNT, XOPES_DEBUG


@contiguous
def lavd_parallel_state_parallel(
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    ldk_cumsum: Optional[torch.Tensor] = None,
    ldv_cumsum: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    MAX_BLOCK_N: int = 256,
    MAX_BLOCK_C: int = 256,
    MAX_BLOCK_E: int = 128,
    MAX_BLOCK_D: int = 128,
    BLOCK_N: int = 256,
):
    if k is None:
        b, n, h, d = ldk.shape
        dtype = ldk.dtype
        device = ldk.device
    else:
        b, n, h, d = k.shape
        dtype = k.dtype
        device = k.device

    if v is None:
        e = ldv.shape[-1]
    else:
        e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)
    use_pad = n % BLOCK_N != 0

    states = torch.empty((b, h, NUM_BLOCK_N + 1, d, e), dtype=dtype, device=device)

    def grid(meta):
        return (
            b * h * NUM_BLOCK_N,
            triton.cdiv(d, meta["BLOCK_D"]),
            triton.cdiv(e, meta["BLOCK_E"]),
        )

    if use_ldk and ldk_cumsum is None:
        ldk_cumsum = chunk_cumsum_decay_fn(ldk, reverse=reverse, chunk_size=BLOCK_N)

    if use_ldv and ldv_cumsum is None:
        ldv_cumsum = chunk_cumsum_decay_fn(ldv, reverse=reverse, chunk_size=BLOCK_N)

    share_k = k is None
    share_v = v is None

    _lavd_parallel_state_parallel[grid](
        K=k,
        V=v,
        STATES=states,
        LOG_DECAY_K=ldk,
        LOG_DECAY_V=ldv,
        LOG_DECAY_K_CUMSUM=ldk_cumsum,
        LOG_DECAY_V_CUMSUM=ldv_cumsum,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_DECAY_K=use_ldk,
        USE_DECAY_V=use_ldv,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_PAD=use_pad,
        REVERSE=reverse,
        SHARE_K=share_k,
        SHARE_V=share_v,
        BLOCK_N=BLOCK_N,
    )

    return states


@contiguous
def lavd_parallel_state_reduce(
    b: int,
    n: int,
    h: int,
    d: int,
    e: int,
    states: torch.Tensor,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    ldk_cumsum: Optional[torch.Tensor] = None,
    ldv_cumsum: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    initial_state: Optional[torch.Tensor] = None,
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
    use_pad = n % BLOCK_N != 0

    def grid(meta):
        return (
            b * h,
            triton.cdiv(d, meta["BLOCK_D"]),
            triton.cdiv(e, meta["BLOCK_E"]),
        )

    if use_ldk and ldk_cumsum is None:
        ldk_cumsum = chunk_cumsum_decay_fn(ldk, reverse=reverse, chunk_size=BLOCK_N)

    if use_ldv and ldv_cumsum is None:
        ldv_cumsum = chunk_cumsum_decay_fn(ldv, reverse=reverse, chunk_size=BLOCK_N)

    _lavd_parallel_state_reduce[grid](
        STATE=initial_state,
        STATES=states,
        LOG_DECAY_K=ldk,
        LOG_DECAY_V=ldv,
        LOG_DECAY_K_CUMSUM=ldk_cumsum,
        LOG_DECAY_V_CUMSUM=ldv_cumsum,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_DECAY_K=use_ldk,
        USE_DECAY_V=use_ldv,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        USE_PAD=use_pad,
        REVERSE=reverse,
        BLOCK_N=BLOCK_N,
    )

    return states


@contiguous
def lavd_parallel_state_parallel_reduce(
    b: int,
    n: int,
    h: int,
    d: int,
    e: int,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    ldk_cumsum: Optional[torch.Tensor] = None,
    ldv_cumsum: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    MAX_BLOCK_N: int = 256,
    MAX_BLOCK_C: int = 256,
    MAX_BLOCK_E: int = 128,
    MAX_BLOCK_D: int = 128,
    BLOCK_N: int = 256,
):
    if k is None:
        b, n, h, d = ldk.shape
        dtype = ldk.dtype
        device = ldk.device
    else:
        b, n, h, d = k.shape
        dtype = k.dtype
        device = k.device

    if v is None:
        e = ldv.shape[-1]
    else:
        e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)
    use_pad = n % BLOCK_N != 0
    use_initial_state = initial_state is not None

    states = torch.empty((b, h, NUM_BLOCK_N + 1, d, e), dtype=dtype, device=device)

    def grid(meta):
        return (
            b * h,
            triton.cdiv(d, meta["BLOCK_D"]),
            triton.cdiv(e, meta["BLOCK_E"]),
        )

    if use_ldk and ldk_cumsum is None:
        ldk_cumsum = chunk_cumsum_decay_fn(ldk, reverse=reverse, chunk_size=BLOCK_N)

    if use_ldv and ldv_cumsum is None:
        ldv_cumsum = chunk_cumsum_decay_fn(ldv, reverse=reverse, chunk_size=BLOCK_N)

    share_k = k is None
    share_v = v is None

    _lavd_parallel_state_parallel_reduce[grid](
        K=k,
        V=v,
        STATE=initial_state,
        STATES=states,
        LOG_DECAY_K=ldk,
        LOG_DECAY_V=ldv,
        LOG_DECAY_K_CUMSUM=ldk_cumsum,
        LOG_DECAY_V_CUMSUM=ldv_cumsum,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_DECAY_K=use_ldk,
        USE_DECAY_V=use_ldv,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        USE_PAD=use_pad,
        REVERSE=reverse,
        SHARE_K=share_k,
        SHARE_V=share_v,
        BLOCK_N=BLOCK_N,
    )

    return states


@contiguous
def lavd_parallel_state_parallel_reduce_sep(
    b: int,
    n: int,
    h: int,
    d: int,
    e: int,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    ldk_cumsum: Optional[torch.Tensor] = None,
    ldv_cumsum: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    MAX_BLOCK_N: int = 256,
    MAX_BLOCK_C: int = 256,
    MAX_BLOCK_E: int = 128,
    MAX_BLOCK_D: int = 128,
    BLOCK_N: int = 256,
):
    # Step0.5: Compute local states in parallel
    states = lavd_parallel_state_parallel(
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        ldk_cumsum=ldk_cumsum,
        ldv_cumsum=ldv_cumsum,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        reverse=reverse,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    # Step1: Update local states to get global states
    states = lavd_parallel_state_reduce(
        b=b,
        n=n,
        h=h,
        d=d,
        e=e,
        states=states,
        ldk=ldk,
        ldv=ldv,
        ldk_cumsum=ldk_cumsum,
        ldv_cumsum=ldv_cumsum,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        initial_state=initial_state,
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
def lavd_parallel_sub_intra(
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    ldk_cumsum: Optional[torch.Tensor] = None,
    ldv_cumsum: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    MAX_BLOCK_N: int = 256,
    MAX_BLOCK_C: int = 256,
    MAX_BLOCK_E: int = 128,
    MAX_BLOCK_D: int = 128,
    BLOCK_N: int = 256,
):
    if k is None:
        b, n, h, d = ldk.shape
        dtype = ldk.dtype
        device = ldk.device
    else:
        b, n, h, d = k.shape
        dtype = k.dtype
        device = k.device

    if v is None:
        e = ldv.shape[-1]
    else:
        e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    if use_cu_seqlens:
        o = torch.empty((1, n, h, e), dtype=dtype, device=device)
    else:
        o = torch.empty((b, n, h, e), dtype=dtype, device=device)

    triton.cdiv(n, BLOCK_N)
    use_pad = n % BLOCK_N != 0

    def grid(meta):
        return (
            b * h,
            triton.cdiv(n, meta["BLOCK_C"]),
            triton.cdiv(e, meta["BLOCK_E"]),
        )

    if use_ldk and ldk_cumsum is None:
        ldk_cumsum = chunk_cumsum_decay_fn(ldk, reverse=reverse, chunk_size=BLOCK_N)

    if use_ldv and ldv_cumsum is None:
        ldv_cumsum = chunk_cumsum_decay_fn(ldv, reverse=reverse, chunk_size=BLOCK_N)

    share_k = k is None
    share_v = v is None

    _lavd_parallel_sub_intra[grid](
        Q=q,
        K=k,
        V=v,
        O=o,
        LOG_DECAY_K=ldk,
        LOG_DECAY_V=ldv,
        LOG_DECAY_K_CUMSUM=ldk_cumsum,
        LOG_DECAY_V_CUMSUM=ldv_cumsum,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_DECAY_K=use_ldk,
        USE_DECAY_V=use_ldv,
        USE_CU_SEQLENS=use_cu_seqlens,
        REVERSE=reverse,
        USE_PAD=use_pad,
        SHARE_K=share_k,
        SHARE_V=share_v,
        BLOCK_N=BLOCK_N,
    )

    return o


@contiguous
def lavd_parallel_sub_intra_sep(
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    ldk_cumsum: Optional[torch.Tensor] = None,
    ldv_cumsum: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    MAX_BLOCK_N: int = 256,
    MAX_BLOCK_C: int = 256,
    MAX_BLOCK_E: int = 128,
    MAX_BLOCK_D: int = 128,
    BLOCK_N: int = 256,
    BLOCK_C: int = 16,
):
    if k is None:
        b, n, h, d = ldk.shape
        dtype = ldk.dtype
        device = ldk.device
    else:
        b, n, h, d = k.shape
        dtype = k.dtype
        device = k.device

    if v is None:
        e = ldv.shape[-1]
    else:
        e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    # attention matrix
    NUM_ATTN_MATRIX = triton.cdiv(n, BLOCK_C)
    a = torch.empty(
        (b, h, NUM_ATTN_MATRIX, BLOCK_C, BLOCK_C), dtype=dtype, device=device
    )

    if use_cu_seqlens:
        o = torch.empty((1, n, h, e), dtype=dtype, device=device)
    else:
        o = torch.empty((b, n, h, e), dtype=dtype, device=device)

    triton.cdiv(n, BLOCK_N)
    use_pad = n % BLOCK_N != 0

    if use_ldk and ldk_cumsum is None:
        ldk_cumsum = chunk_cumsum_decay_fn(ldk, reverse=reverse, chunk_size=BLOCK_N)

    if use_ldv and ldv_cumsum is None:
        ldv_cumsum = chunk_cumsum_decay_fn(ldv, reverse=reverse, chunk_size=BLOCK_N)

    share_q = q is None
    share_k = k is None
    share_v = v is None

    def grid(meta):
        return (b * h, NUM_ATTN_MATRIX)

    _lavd_parallel_sub_intra_attn[grid](
        Q=q,
        K=k,
        V=v,
        A=a,
        LOG_DECAY_K=ldk,
        LOG_DECAY_V=ldv,
        LOG_DECAY_K_CUMSUM=ldk_cumsum,
        LOG_DECAY_V_CUMSUM=ldv_cumsum,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_DECAY_K=use_ldk,
        USE_DECAY_V=use_ldv,
        USE_CU_SEQLENS=use_cu_seqlens,
        REVERSE=reverse,
        USE_PAD=use_pad,
        SHARE_Q=share_q,
        SHARE_K=share_k,
        SHARE_V=share_v,
        BLOCK_N=BLOCK_N,
        BLOCK_C=BLOCK_C,
    )

    def grid(meta):
        return (
            b * h,
            triton.cdiv(n, meta["BLOCK_C"]),
            triton.cdiv(e, meta["BLOCK_E"]),
        )

    _lavd_parallel_sub_intra_o[grid](
        Q=q,
        K=k,
        V=v,
        O=o,
        A=a,
        LOG_DECAY_K=ldk,
        LOG_DECAY_V=ldv,
        LOG_DECAY_K_CUMSUM=ldk_cumsum,
        LOG_DECAY_V_CUMSUM=ldv_cumsum,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_DECAY_K=use_ldk,
        USE_DECAY_V=use_ldv,
        USE_CU_SEQLENS=use_cu_seqlens,
        REVERSE=reverse,
        USE_PAD=use_pad,
        SHARE_K=share_k,
        SHARE_V=share_v,
        BLOCK_N=BLOCK_N,
        BLOCK_C=BLOCK_C,
    )

    return o


@contiguous
def lavd_parallel_intra(
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    ldk_cumsum: Optional[torch.Tensor] = None,
    ldv_cumsum: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    MAX_BLOCK_N: int = 256,
    MAX_BLOCK_C: int = 256,
    MAX_BLOCK_E: int = 128,
    MAX_BLOCK_D: int = 128,
    BLOCK_N: int = 256,
):
    if k is None:
        b, n, h, d = ldk.shape
        ldk.dtype
        ldk.device
    else:
        b, n, h, d = k.shape
        k.dtype
        k.device

    if v is None:
        e = ldv.shape[-1]
    else:
        e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    if use_cu_seqlens:
        o = torch.empty((1, n, h, e), dtype=q.dtype, device=q.device)
    else:
        o = torch.empty((b, n, h, e), dtype=q.dtype, device=q.device)

    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)
    use_pad = n % BLOCK_N != 0

    def grid(meta):
        return (
            b * h * NUM_BLOCK_N,
            triton.cdiv(BLOCK_N, meta["BLOCK_C"]),
            triton.cdiv(e, meta["BLOCK_E"]),
        )

    if use_ldk and ldk_cumsum is None:
        ldk_cumsum = chunk_cumsum_decay_fn(ldk, reverse=reverse, chunk_size=BLOCK_N)

    if use_ldv and ldv_cumsum is None:
        ldv_cumsum = chunk_cumsum_decay_fn(ldv, reverse=reverse, chunk_size=BLOCK_N)

    share_k = k is None
    share_v = v is None

    _lavd_parallel_intra[grid](
        Q=q,
        K=k,
        V=v,
        O=o,
        LOG_DECAY_K=ldk,
        LOG_DECAY_V=ldv,
        LOG_DECAY_K_CUMSUM=ldk_cumsum,
        LOG_DECAY_V_CUMSUM=ldv_cumsum,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_DECAY_K=use_ldk,
        USE_DECAY_V=use_ldv,
        USE_CU_SEQLENS=use_cu_seqlens,
        REVERSE=reverse,
        USE_PAD=use_pad,
        SHARE_K=share_k,
        SHARE_V=share_v,
        BLOCK_N=BLOCK_N,
        MAX_BLOCK_N=MAX_BLOCK_N,
    )

    return o


@contiguous
def lavd_parallel_inter(
    q: torch.Tensor,
    o: torch.Tensor,
    states: torch.Tensor,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    ldk_cumsum: Optional[torch.Tensor] = None,
    ldv_cumsum: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
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

    def grid(meta):
        return (
            b * h * NUM_BLOCK_N,
            triton.cdiv(BLOCK_N, meta["BLOCK_C"]),
            triton.cdiv(e, meta["BLOCK_E"]),
        )

    if use_ldk and ldk_cumsum is None:
        ldk_cumsum = chunk_cumsum_decay_fn(ldk, reverse=reverse, chunk_size=BLOCK_N)

    if use_ldv and ldv_cumsum is None:
        ldv_cumsum = chunk_cumsum_decay_fn(ldv, reverse=reverse, chunk_size=BLOCK_N)

    _lavd_parallel_inter[grid](
        Q=q,
        O=o,
        STATES=states,
        LOG_DECAY_K_CUMSUM=ldk_cumsum,
        LOG_DECAY_V_CUMSUM=ldv_cumsum,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_DECAY_K=use_ldk,
        USE_DECAY_V=use_ldv,
        USE_CU_SEQLENS=use_cu_seqlens,
        REVERSE=reverse,
        TRANS=trans,
        BLOCK_N=BLOCK_N,
    )

    return o


@contiguous
def lavd_parallel_intra_inter(
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    states: Optional[torch.Tensor] = None,
    a: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    ldk_cumsum: Optional[torch.Tensor] = None,
    ldv_cumsum: Optional[torch.Tensor] = None,
    ldk_reverse_cumsum: Optional[torch.Tensor] = None,
    ldv_reverse_cumsum: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    x: Optional[torch.Tensor] = None,  # use for dldv compute
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    trans_state: bool = False,
    trans_a: bool = False,
    share_x: bool = False,
    MAX_BLOCK_N: int = 256,
    MAX_BLOCK_C: int = 256,
    MAX_BLOCK_E: int = 128,
    MAX_BLOCK_D: int = 128,
    BLOCK_N: int = 256,
):
    if k is None:
        b, n, h, d = ldk.shape
        dtype = ldk.dtype
        device = ldk.device
    else:
        b, n, h, d = k.shape
        dtype = k.dtype
        device = k.device

    if v is None:
        e = ldv.shape[-1]
    else:
        e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    if use_cu_seqlens:
        o = torch.empty((1, n, h, e), dtype=dtype, device=device)
    else:
        o = torch.empty((b, n, h, e), dtype=dtype, device=device)

    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)
    use_pad = n % BLOCK_N != 0

    if use_ldk and ldk_cumsum is None:
        ldk_cumsum = chunk_cumsum_decay_fn(ldk, reverse=reverse, chunk_size=BLOCK_N)

    if use_ldv and ldv_cumsum is None:
        ldv_cumsum = chunk_cumsum_decay_fn(ldv, reverse=reverse, chunk_size=BLOCK_N)

    compute_dld = use_ldv and ((x is not None) or share_x)
    if compute_dld:
        dld = torch.empty((b, n, h, e), dtype=dtype, device=device)
    else:
        dld = None

    share_q = q is None
    share_k = k is None
    share_v = v is None

    def grid(meta):
        return (
            b * h,
            triton.cdiv(n, meta["BLOCK_C"]),
            triton.cdiv(e, meta["BLOCK_E"]),
        )

    if a is None:
        NUM_ATTN_MATRIX = triton.cdiv(n, BLOCK_C)
        # a_dtype = (
        #     dtype if (share_q or share_k or share_v) else torch.float32
        # )  # !!! important, if not share, a must be float32
        # !!! important, a must be float32
        a = torch.empty(
            (b, h, NUM_ATTN_MATRIX, BLOCK_C, BLOCK_C),
            dtype=torch.float32,
            device=device,
        )

        def grid(meta):
            return (b * h, NUM_ATTN_MATRIX)

        _lavd_parallel_sub_intra_attn[grid](
            Q=q,
            K=k,
            V=v,
            A=a,
            LOG_DECAY_K=ldk,
            LOG_DECAY_V=ldv,
            LOG_DECAY_K_CUMSUM=ldk_cumsum,
            LOG_DECAY_V_CUMSUM=ldv_cumsum,
            CU_SEQLENS=cu_seqlens,
            B=b,
            N=n,
            H=h,
            D=d,
            E=e,
            USE_DECAY_K=use_ldk,
            USE_DECAY_V=use_ldv,
            USE_CU_SEQLENS=use_cu_seqlens,
            REVERSE=reverse,
            USE_PAD=use_pad,
            SHARE_Q=share_q,
            SHARE_K=share_k,
            SHARE_V=share_v,
            BLOCK_N=BLOCK_N,
            BLOCK_C=BLOCK_C,
        )

    def grid(meta):
        return (
            b * h * NUM_BLOCK_N,
            triton.cdiv(BLOCK_N, meta["BLOCK_C"]),
            triton.cdiv(e, meta["BLOCK_E"]),
        )

    _lavd_parallel_intra_inter_no_loop[grid](
        Q=q,
        K=k,
        V=v,
        O=o,
        A=a,
        STATES=states,
        LOG_DECAY_K=ldk,
        LOG_DECAY_V=ldv,
        LOG_DECAY_K_CUMSUM=ldk_cumsum,
        LOG_DECAY_V_CUMSUM=ldv_cumsum,
        X=x,
        DLOG_DECAY=dld,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_DECAY_K=use_ldk,
        USE_DECAY_V=use_ldv,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_PAD=use_pad,
        COMPUTE_DLD=compute_dld,
        REVERSE=reverse,
        TRANS_STATE=trans_state,
        TRANS_A=trans_a,
        SHARE_Q=share_q,
        SHARE_K=share_k,
        SHARE_V=share_v,
        SHARE_X=share_x,
        BLOCK_N=BLOCK_N,
        BLOCK_C_=BLOCK_C,
    )

    return o, dld, a


########## Fwd start ##########
@contiguous
def lavd_parallel_fwd(
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    trans: bool = False,
    use_chunk_loop: bool = False,
):
    """
    Forward pass for Lightning Attention with Data-Dependent Scalar Decay in parallel mode.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        ldk: Log decay tensor for key of shape (B, N, H, D)
        ldv: Log decay tensor for value of shape (B, N, H, E)
        initial_state: Initial state tensor of shape (B, H, D, E)
        use_ldk: Whether to use log decay for key
        use_ldv: Whether to use log decay for value
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        reverse: Whether to process the sequence in reverse order
        trans: Whether to transpose the final output
        use_chunk_loop: Whether to use chunk loop

    Returns:
        o: Output tensor of shape (B, N, H, E)
        states: Final state tensor
    """
    b, n, h, d = q.shape
    if v is None:
        e = ldv.shape[-1]
    else:
        e = v.shape[-1]

    MAX_BLOCK_N = max(MIN_BLOCK, triton.next_power_of_2(n))
    MAX_BLOCK_E = max(MIN_BLOCK, triton.next_power_of_2(e))
    MAX_BLOCK_D = max(MIN_BLOCK, triton.next_power_of_2(d))
    NUM_PARALLEL_BLOCKS = b * h
    if XOPES_DEBUG:
        USE_CHUNK_LOOP = use_chunk_loop
    else:
        USE_CHUNK_LOOP = NUM_PARALLEL_BLOCKS >= SM_COUNT or use_chunk_loop

    BLOCK_N = min(MAX_BLOCK_N, 128)

    MAX_BLOCK_C = MAX_BLOCK_N

    # Step1: Compute states in parallel or chunk loop
    if USE_CHUNK_LOOP:
        fn = lavd_parallel_state_parallel_reduce
    else:
        fn = lavd_parallel_state_parallel_reduce_sep

    ldk_cumsum = None
    if use_ldk:
        ldk_cumsum = chunk_cumsum_decay_fn(ldk, reverse=reverse, chunk_size=BLOCK_N)

    ldv_cumsum = None
    if use_ldv:
        ldv_cumsum = chunk_cumsum_decay_fn(ldv, reverse=reverse, chunk_size=BLOCK_N)

    states = fn(
        b=b,
        n=n,
        h=h,
        d=d,
        e=e,
        k=k,
        v=v,
        initial_state=initial_state,
        ldk=ldk,
        ldv=ldv,
        ldk_cumsum=ldk_cumsum,
        ldv_cumsum=ldv_cumsum,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        cu_seqlens=cu_seqlens,
        reverse=reverse,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    # Step2: Compute intra and inter in parallel, for each chunk, parallel over sub-chunk
    o, _, a = lavd_parallel_intra_inter(
        q=q,
        k=k,
        v=v,
        states=states,
        ldk=ldk,
        ldv=ldv,
        ldk_cumsum=ldk_cumsum,
        ldv_cumsum=ldv_cumsum,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        cu_seqlens=cu_seqlens,
        reverse=reverse,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    return o, states, ldk_cumsum, ldv_cumsum, a


@contiguous
def lavd_parallel_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    ldk_cumsum: Optional[torch.Tensor] = None,
    ldv_cumsum: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    dfinal_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    states: Optional[torch.Tensor] = None,
    a: Optional[torch.Tensor] = None,
    use_chunk_loop: bool = False,
):
    """
    Backward pass for Lightning Attention with Data-Dependent Scalar Decay in parallel mode.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        do: Gradient of output tensor of shape (B, N, H, E)
        ldk: Log decay for key tensor of shape (B, N, H, D)
        ldv: Log decay for value tensor of shape (B, N, H, E)
        ldk_cumsum: Cumulative log decay for key tensor of shape (B, N, H)
        ldv_cumsum: Cumulative log decay for value tensor of shape (B, N, H)
        use_ldk: Whether to use log decay for key
        use_ldv: Whether to use log decay for value
        initial_state: Initial state tensor of shape (B, H, D, E)
        dfinal_state: Gradient of final state tensor
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        states: Cached states from forward pass (optional)
        a: Cached attention matrix from forward pass (optional)
        use_chunk_loop: Whether to use chunk loop

    Returns:
        dq: Gradient of query tensor
        dk: Gradient of key tensor
        dv: Gradient of value tensor
        dldk: Gradient of log decay tensor for key
        dldv: Gradient of log decay tensor for value
        dinitial_state: Gradient of initial state tensor
    """
    b, n, h, d = q.shape
    if v is None:
        e = ldv.shape[-1]
    else:
        e = v.shape[-1]

    MAX_BLOCK_N = max(MIN_BLOCK, triton.next_power_of_2(n))
    MAX_BLOCK_E = max(MIN_BLOCK, triton.next_power_of_2(e))
    MAX_BLOCK_D = max(MIN_BLOCK, triton.next_power_of_2(d))
    NUM_PARALLEL_BLOCKS = b * h
    if XOPES_DEBUG:
        USE_CHUNK_LOOP = use_chunk_loop
    else:
        USE_CHUNK_LOOP = NUM_PARALLEL_BLOCKS >= SM_COUNT or use_chunk_loop

    share_k = k is None
    share_v = v is None

    BLOCK_N = min(MAX_BLOCK_N, 128)

    MAX_BLOCK_C = MAX_BLOCK_N

    if USE_CHUNK_LOOP:
        fn = lavd_parallel_state_parallel_reduce
    else:
        fn = lavd_parallel_state_parallel_reduce_sep

    # Recompute states if not provided
    if states is None:
        states = fn(
            b=b,
            n=n,
            h=h,
            d=d,
            e=e,
            k=k,
            v=v,
            initial_state=initial_state,
            ldk=ldk,
            ldv=ldv,
            ldk_cumsum=ldk_cumsum,
            ldv_cumsum=ldv_cumsum,
            use_ldk=use_ldk,
            use_ldv=use_ldv,
            cu_seqlens=cu_seqlens,
            reverse=reverse,
            MAX_BLOCK_N=MAX_BLOCK_N,
            MAX_BLOCK_C=MAX_BLOCK_C,
            MAX_BLOCK_E=MAX_BLOCK_E,
            MAX_BLOCK_D=MAX_BLOCK_D,
            BLOCK_N=BLOCK_N,
        )

    ldk_cumsum = None
    if ldk is not None and ldk_cumsum is None:
        ldk_cumsum = chunk_cumsum_decay_fn(ldk, reverse=False, chunk_size=BLOCK_N)

    ldv_cumsum = None
    if ldv is not None and ldv_cumsum is None:
        ldv_cumsum = chunk_cumsum_decay_fn(ldv, reverse=False, chunk_size=BLOCK_N)

    if ldv is not None:
        _, dldv_o, _ = lavd_parallel_intra_inter(
            q=q,
            k=k,
            v=v,
            states=states,
            a=a,
            ldk=ldk,
            ldv=ldv,
            ldk_cumsum=ldk_cumsum,
            ldv_cumsum=ldv_cumsum,
            use_ldk=use_ldk,
            use_ldv=use_ldv,
            x=do,
            cu_seqlens=cu_seqlens,
            reverse=False,
            trans_state=False,
            trans_a=False,
            MAX_BLOCK_N=MAX_BLOCK_N,
            MAX_BLOCK_C=MAX_BLOCK_C,
            MAX_BLOCK_E=MAX_BLOCK_E,
            MAX_BLOCK_D=MAX_BLOCK_D,
            BLOCK_N=BLOCK_N,
        )
    else:
        dldv_o = None

    dq, dldk_q, da = lavd_parallel_intra_inter(
        q=do,  # b n h e
        k=v,  # b n h e
        v=k,  # b n h d
        states=states,  # b h (m + 1) d e
        ldk=ldv,  # b n h e
        ldv=ldk,  # b n h d
        ldk_cumsum=ldv_cumsum,
        ldv_cumsum=ldk_cumsum,
        use_ldk=use_ldv,
        use_ldv=use_ldk,
        x=q,  # b n h d
        cu_seqlens=cu_seqlens,
        reverse=False,
        trans_state=True,
        trans_a=False,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    final_state = states[:, :, -1, :, :]
    del states

    ldk_reverse_cumsum = None
    if use_ldk:
        ldk_reverse_cumsum = chunk_cumsum_decay_fn(
            ldk, reverse=True, chunk_size=BLOCK_N
        )

    ldv_reverse_cumsum = None
    if use_ldv:
        ldv_reverse_cumsum = chunk_cumsum_decay_fn(
            ldv, reverse=True, chunk_size=BLOCK_N
        )

    # Compute dstates for dk and dv
    dstates = fn(
        b=b,
        n=n,
        h=h,
        d=d,
        e=e,
        k=q,  # b n h d
        v=do,  # b n h e
        initial_state=dfinal_state,
        ldk=ldk,
        ldv=ldv,
        ldk_cumsum=ldk_reverse_cumsum,
        ldv_cumsum=ldv_reverse_cumsum,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        cu_seqlens=cu_seqlens,
        reverse=True,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    dk, dldk_k, _ = lavd_parallel_intra_inter(
        q=v,  # b n h e
        k=do,  # b n h e
        v=q,  # b n h d
        states=dstates,  # b h (m + 1) d e
        a=da,
        ldk=ldv,  # b n h e
        ldv=ldk,  # b n h d
        ldk_cumsum=ldv_reverse_cumsum,
        ldv_cumsum=ldk_reverse_cumsum,
        use_ldk=use_ldv,
        use_ldv=use_ldk,
        x=k,  # b n h d
        cu_seqlens=cu_seqlens,
        reverse=True,
        trans_state=True,
        trans_a=True,
        share_x=share_k,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    dv, dldv_v, _ = lavd_parallel_intra_inter(
        q=k,  # b n h d
        k=q,  # b n h d
        v=do,  # b n h e
        states=dstates,  # b h (m + 1) d e
        a=a,
        ldk=ldk,  # b n h d
        ldv=ldv,  # b n h e
        ldk_cumsum=ldk_reverse_cumsum,
        ldv_cumsum=ldv_reverse_cumsum,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        x=v if ldv is not None else None,  # b n h e
        cu_seqlens=cu_seqlens,
        reverse=True,
        trans_state=False,
        trans_a=True,
        share_x=share_v,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    # Compute gradient for initial state if needed
    need_dfinal_state = (
        dfinal_state is not None
        and initial_state is not None
        and initial_state.requires_grad
    )

    if ldk is not None and use_ldk and ldk.requires_grad:
        dldk = compute_dld_with_cumsum_fn(
            dld_q=dldk_q,  # B N H D
            dld_k=dldk_k,  # B N H D
            dfinal_state=dfinal_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            sum_option=0,
        )
    else:
        dldk = None

    if share_k:
        dldk += dk * (-torch.exp(ldk))
        dk = None

    if ldv is not None and use_ldv and ldv.requires_grad:
        dldv = compute_dld_with_cumsum_fn(
            dld_q=dldv_o,  # B N H E
            dld_k=dldv_v,  # B N H E
            dfinal_state=dfinal_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            sum_option=1,
        )
    else:
        dldv = None

    if share_v:
        dldv += dv * (-torch.exp(ldv))
        dv = None

    return (
        dq,
        dk,
        dv,
        dldk,
        dldv,
        dstates[:, :, -1, :, :] if need_dfinal_state else None,
    )


class LavdParallelFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        ldk,
        ldv,
        use_ldk,
        use_ldv,
        initial_state=None,
        cu_seqlens=None,
        save_states=True,
        save_ld=True,
        save_a=True,
        use_chunk_loop=False,
    ):
        # Forward computation
        output, states, ldk_cumsum, ldv_cumsum, a = lavd_parallel_fwd(
            q=q,
            k=k,
            v=v,
            ldk=ldk,
            ldv=ldv,
            use_ldk=use_ldk,
            use_ldv=use_ldv,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            use_chunk_loop=use_chunk_loop,
        )

        # Save tensors needed for backward
        final_state = states[:, :, -1, :, :]
        if not save_states:
            states = None
        if not save_ld:
            ldk_cumsum = None
            ldv_cumsum = None
        if not save_a:
            a = None

        ctx.save_for_backward(
            q,
            k,
            v,
            ldk,
            ldv,
            ldk_cumsum,
            ldv_cumsum,
            initial_state,
            cu_seqlens,
            states,
            a,
        )

        ctx.use_chunk_loop = use_chunk_loop
        ctx.use_ldk = use_ldk
        ctx.use_ldv = use_ldv
        del states
        del ldk_cumsum
        del ldv_cumsum

        return output, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dfinal_state):
        (
            q,
            k,
            v,
            ldk,
            ldv,
            ldk_cumsum,
            ldv_cumsum,
            initial_state,
            cu_seqlens,
            states,
            a,
        ) = ctx.saved_tensors

        use_chunk_loop = ctx.use_chunk_loop
        use_ldk = ctx.use_ldk
        use_ldv = ctx.use_ldv
        dq, dk, dv, dldk, dldv, dinitial_state = lavd_parallel_bwd(
            q=q,
            k=k,
            v=v,
            do=do,
            ldk=ldk,
            ldv=ldv,
            ldk_cumsum=ldk_cumsum,
            ldv_cumsum=ldv_cumsum,
            use_ldk=use_ldk,
            use_ldv=use_ldv,
            initial_state=initial_state,
            dfinal_state=dfinal_state,
            cu_seqlens=cu_seqlens,
            states=states,
            a=a,
            use_chunk_loop=use_chunk_loop,
        )

        return (
            dq,
            dk,
            dv,
            dldk,
            dldv,
            None,
            None,
            dinitial_state,
            None,
            None,
            None,
            None,
            None,
        )


def lavd_parallel_triton(
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    save_states: bool = True,
    save_ld: bool = True,
    save_a: bool = True,
    use_chunk_loop: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Lightning Attention with Vector Decay in Pytorch.

    Args:
        q: Query tensor, shape (B, N, H, D)
        k: Key tensor, shape (B, N, H, D)
        v: Value tensor, shape (B, N, H, E)
        ldk: Log Decay vector for key, shape (B, N, H, D)
        ldv: Log Decay vector for value, shape (B, N, H, E)
        use_ldk: Whether to use log decay for key
        use_ldv: Whether to use log decay for value
        initial_state: Initial state tensor, shape (B, H, D, E) or (H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training
        save_states: Whether to save states for backward
        save_ld: Whether to save log decay for backward
        save_a: Whether to save a for backward
        use_chunk_loop: Whether to use chunk loop for forward

    Returns:
        Output tensor, shape (B, N, H, E)
        State tensor, shape (B, H, D, E)
    """
    b = q.shape[0]
    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1
    if initial_state is not None:
        initial_state = initial_state.squeeze(0)
        # treat for varlen training
        if len(initial_state.shape) == 3:
            initial_state = repeat(initial_state, "h d e -> b h d e", b=b)
    if ldk is not None:
        use_ldk = True
    else:
        use_ldk = False

    if ldv is not None:
        use_ldv = True
    else:
        use_ldv = False

    return LavdParallelFunction.apply(
        q,
        k,
        v,
        ldk,
        ldv,
        use_ldk,
        use_ldv,
        initial_state,
        cu_seqlens,
        save_states,
        save_ld,
        save_a,
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
    # Data-dependent decay factors
    ldk = F.logsigmoid(torch.randn(b, n, h, d, device=device)).requires_grad_(True)
    ldv = F.logsigmoid(torch.randn(b, n, h, e, device=device)).requires_grad_(True)
    output, final_state = lavd_parallel_triton(q, k, v, ldk, ldv)
    loss = output.sum() + final_state.sum()
    loss.backward()
