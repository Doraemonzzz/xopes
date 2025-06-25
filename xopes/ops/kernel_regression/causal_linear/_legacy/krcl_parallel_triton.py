# krcl: kernel regression with causal linear
from typing import Optional

import torch
import triton

from xopes.ops.cumsum import chunk_cumsum_decay_fn
from xopes.ops.kernel_regression.causal_linear.utils import (
    _krcl_parallel_inverse_diag,
    _krcl_parallel_inverse_merge,
)
from xopes.utils import contiguous


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
        BLOCK_N1=BLOCK_N // 2,
        BLOCK_N2=BLOCK_N // 4,
        NUM_BLOCK_N=NUM_BLOCK_N,
    )

    return inv


#### v2 #####
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
    inv = torch.zeros(
        (b, h, NUM_BLOCK_N, BLOCK_N, BLOCK_N), device=k.device, dtype=k.dtype
    )
    m = 4
    BLOCK_M = BLOCK_N // m

    def grid(meta):
        return (b * h, NUM_BLOCK_N, m)

    if ld_cumsum is None:
        ld_cumsum = chunk_cumsum_decay_fn(
            ld, reverse=reverse, chunk_size=BLOCK_N, use_offset=False
        )

    _krcl_parallel_inverse_diag[grid](
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
        BLOCK_M=BLOCK_M,
        NUM_BLOCK_N=NUM_BLOCK_N,
    )

    def grid(meta):
        return (b, h, NUM_BLOCK_N)

    _krcl_parallel_inverse_merge[grid](
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
        BLOCK_M=BLOCK_M,
        NUM_BLOCK_N=NUM_BLOCK_N,
    )

    return inv
