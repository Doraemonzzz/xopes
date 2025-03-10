from typing import Optional, Tuple

import torch

from .positional_encoding import lape_fn
from .scalar_decay import lasd_fn


def lightning_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(q.shape) == 2:
        fn = lape_fn
    else:
        fn = lasd_fn

    return fn(
        q=q,
        k=k,
        v=v,
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )
