from typing import Optional, Tuple

import torch

from .constant_decay import lacd_fn
from .element_recurrence import laer_fn
from .positional_encoding import lape_fn
from .scalar_decay import lasd_fn
from .vector_decay import lavd_fn


def lightning_attn_func(
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    ld: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    save_states: bool = True,
    save_ld: bool = True,
    save_a: bool = True,
    use_chunk_loop: bool = True,
    decay_type: str = "constant",
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Lightning Attention with Constant/Vector/Scalar Decay in Pytorch.

    Args:
        q: Query tensor, shape (B, N, H, D)
        k: Key tensor, shape (B, N, H, D)
        v: Value tensor, shape (B, N, H, E)
        ld: Log Decay vector, shape (B, N, H) or (H,)
        ldk: Log Decay vector for key, shape (B, N, H, D)
        ldv: Log Decay vector for value, shape (B, N, H, E)
        initial_state: Initial state tensor, shape (B, H, D, E) or (H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training
        save_states: Whether to save states for backward
        save_ld: Whether to save log decay for backward
        save_a: Whether to save a for backward
        use_chunk_loop: Whether to use chunk loop
        decay_type: Decay type, one of "constant", "vector", "scalar", "positional", "element"

    Returns:
        Output tensor, shape (B, N, H, E)
        State tensor, shape (B, H, D, E)
    """

    if decay_type == "constant":
        fn = lacd_fn
    elif decay_type == "vector":
        fn = lavd_fn
    elif decay_type == "scalar":
        fn = lasd_fn
    elif decay_type == "positional":
        fn = lape_fn
    elif decay_type == "element":
        fn = laer_fn
    else:
        raise ValueError(f"Invalid decay type: {decay_type}")

    return fn(
        q=q,
        k=k,
        v=v,
        ld=ld,
        ldk=ldk,
        ldv=ldv,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        save_states=save_states,
        save_ld=save_ld,
        save_a=save_a,
        use_chunk_loop=use_chunk_loop,
    )
