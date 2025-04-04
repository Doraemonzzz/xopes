from typing import Optional

import torch
import triton

from xopes.ops.cumsum import chunk_cumsum_decay_fn
from xopes.ops.lightning_attn.vector_decay.utils import (
    _lavd_parallel_state_parallel,
    _lavd_parallel_state_parallel_reduce,
    _lavd_parallel_state_reduce,
)
from xopes.utils import contiguous


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
