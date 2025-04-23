from typing import Optional, Tuple

import torch

from .lacd_parallel_torch import lacd_parallel_torch
from .lacd_parallel_triton import lacd_parallel_triton
from .lacd_recurrence_triton import lacd_recurrence_triton
from .lacd_torch import lacd_torch


def lacd_fn(
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
    Apply Lightning Attention Parallel with Constant Decay in Triton.

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
    if v.requires_grad and v.shape[1] > 1:  # TODO: update this later
        fn = lacd_parallel_triton
    else:
        fn = lacd_recurrence_triton

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
