from typing import Optional, Tuple

import torch

from .laer_parallel_triton import laer_parallel_triton
from .laer_recurrence_triton import laer_recurrence_triton
from .laer_torch import laer_torch


def laer_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Lightning Attention Parallel with Constant Decay in Triton.

    Args:
        q: Query tensor of shape (B, N, D)
        k: Key tensor of shape (B, N, D)
        v: Value tensor of shape (B, N, D)
        ld: Logarithmic decay tensor of shape (B, N, D)
        initial_state: Initial state tensor of shape (B, D)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        output: Tensor of shape (B, N, D)
        state: Tensor of shape (B, D)
    """
    if v.shape[1] > 1:
        fn = laer_parallel_triton
    else:
        fn = laer_recurrence_triton

    return fn(q=q, k=k, v=v, ld=ld, initial_state=initial_state, cu_seqlens=cu_seqlens)
