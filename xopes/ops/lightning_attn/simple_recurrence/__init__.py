from typing import Optional, Tuple

import torch

from .lasr_recurrence_triton import lasr_recurrence_triton
from .lasr_torch import lasr_torch


def lasr_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return lasr_recurrence_triton(
        q=q, k=k, v=v, ld=ld, initial_state=initial_state, cu_seqlens=cu_seqlens
    )
