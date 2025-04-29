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
    return laer_recurrence_triton(
        q=q, k=k, v=v, ld=ld, initial_state=initial_state, cu_seqlens=cu_seqlens
    )
