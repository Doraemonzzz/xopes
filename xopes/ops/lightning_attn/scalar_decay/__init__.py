from typing import Optional, Tuple

import torch

from .lasd_parallel_triton import lasd_parallel_triton
from .lasd_recurrence_triton import lasd_recurrence_triton
from .lasd_torch import lasd_torch


def lasd_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    save_states: bool = True,
    use_chunk_loop: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Lightning Attention Parallel with Data-Dependent Scalar Decay in Triton.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (B, N, H) - data dependent decay factors
        initial_state: Initial state tensor of shape (B, H, D, E)
        save_states: Whether to save states for backward
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training
        use_chunk_loop: Whether to use chunk loop

    Returns:
        output: Tensor of shape (B, N, H, E)
        state: Tensor of shape (B, H, D, E)
    """
    if v.shape[1] > 1:
        fn = lasd_parallel_triton
    else:
        fn = lasd_recurrence_triton

    return fn(
        q=q,
        k=k,
        v=v,
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        save_states=save_states,
        use_chunk_loop=use_chunk_loop,
    )
