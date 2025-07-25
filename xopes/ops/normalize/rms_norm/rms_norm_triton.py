from typing import Optional, Tuple

import torch

from ..normalize import normalize_triton


def rms_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    dim: int,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
    return_residual: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    return normalize_triton(
        x=x,
        weight=weight,
        bias=None,
        residual=residual,
        c=dim**0.5,
        eps=eps,
        use_mean=False,
        num_groups=1,
        return_residual=return_residual,
    )
