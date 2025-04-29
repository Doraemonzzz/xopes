from typing import Optional

import torch
import triton

from xopes.ops.lightning_attn.element_recurrence.utils import (
    _laer_parallel_state_parallel,
)
from xopes.utils import contiguous


@contiguous
def laer_parallel_state_parallel(
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    ld_cumsum: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    compute_ld_cumsum: bool = True,
    BLOCK_N: int = 256,
):
    b, n, d = k.shape

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_pad = n % BLOCK_N != 0

    states = torch.empty((b, n, d), dtype=k.dtype, device=k.device)
    num_block_n = triton.cdiv(n, BLOCK_N)

    def grid(meta):
        return (
            b,
            triton.cdiv(n, BLOCK_N),
            triton.cdiv(d, meta["BLOCK_D"]),
        )

    if ld_cumsum is None:
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
        COMPUTE_LOG_DECAY_CUMSUM=compute_ld_cumsum,
        REVERSE=reverse,
        BLOCK_N=BLOCK_N,
        NUM_BLOCK_N=num_block_n,
    )

    return states, ld_cumsum
