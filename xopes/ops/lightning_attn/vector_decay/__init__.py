from typing import Optional, Tuple

import torch

from .lavd_parallel_triton import lavd_parallel_triton
from .lavd_recurrence_triton import lavd_recurrence_triton
from .lavd_torch import lavd_torch


def lavd_fn(
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    save_states: bool = True,
    save_ld: bool = True,
    save_a: bool = True,
    use_chunk_loop: bool = True,
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
        initial_state: Initial state tensor, shape (B, H, D, E) or (H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training
        save_states: Whether to save states for backward
        save_ld: Whether to save log decay for backward
        save_a: Whether to save a for backward
        use_chunk_loop: Whether to use chunk loop

    Returns:
        Output tensor, shape (B, N, H, E)
        State tensor, shape (B, H, D, E)
    """
    if q.shape[1] > 1:
        fn = lavd_parallel_triton
    else:
        fn = lavd_recurrence_triton

    use_ldk = ldk is not None
    use_ldv = ldv is not None

    return fn(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        save_states=save_states,
        save_ld=save_ld,
        save_a=save_a,
        use_chunk_loop=use_chunk_loop,
    )
