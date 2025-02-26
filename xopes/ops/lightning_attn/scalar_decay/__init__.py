from typing import Optional, Tuple

import torch

from .lasd_parallel_torch import lasd_parallel_torch
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
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if q.requires_grad:
        fn = lasd_parallel_triton
    else:
        fn = lasd_recurrence_triton

    return fn(q=q, k=k, v=v, ld=ld, initial_state=initial_state, cu_seqlens=cu_seqlens)
