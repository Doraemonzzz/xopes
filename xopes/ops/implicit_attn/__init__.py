from typing import Optional, Tuple

import torch

from .inverse_attn import ilav_recurrence_triton


def implicit_attn_func(
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    o: Optional[torch.Tensor] = None,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    save_states: bool = True,
    save_ld: bool = True,
    use_chunk_loop: bool = False,
    implicit_type: str = "inverse_v",
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Implicit Attention in Pytorch.

    Args:
        q: Query tensor, shape (B, N, H, D)
        k: Key tensor, shape (B, N, H, D)
        v: Value tensor, shape (B, N, H, E)
        o: Output tensor, shape (B, N, H, E)
        ld: Log Decay vector, shape (B, N, H) or (H,)
        initial_state: Initial state tensor, shape (B, H, D, E) or (H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training
        save_states: Whether to save states for backward
        save_ld: Whether to save log decay for backward
        use_chunk_loop: Whether to use chunk loop
        implicit_type: Decay type, one of "inverse_v"

    Returns:
        Output tensor, shape (B, N, H, E)
        State tensor, shape (B, H, D, E)
    """

    if implicit_type == "inverse_v":
        fn = ilav_recurrence_triton
    else:
        raise ValueError(f"Invalid implicit type: {implicit_type}")

    return fn(
        q=q,
        k=k,
        v=v,
        o=o,
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        save_states=save_states,
        save_ld=save_ld,
        use_chunk_loop=use_chunk_loop,
    )
