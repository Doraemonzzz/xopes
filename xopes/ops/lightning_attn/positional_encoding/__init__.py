from typing import Optional, Tuple

import torch

from .lape_parallel_triton import lape_parallel_triton
from .lape_recurrence_triton import lape_recurrence_triton
from .lape_torch import lape_torch


def lape_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if q.requires_grad:
        fn = lape_parallel_triton
    else:
        fn = lape_recurrence_triton

    return fn(q=q, k=k, v=v, ld=ld, initial_state=initial_state, cu_seqlens=cu_seqlens)
