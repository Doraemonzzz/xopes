from typing import Optional, Tuple

import torch

from ..normalize import normalize_triton


def grouprmsnorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    dim: int,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
    return_residual: bool = False,
    num_groups: int = 1,  # Added parameter for group normalization
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    group_size = dim // num_groups
    return normalize_triton(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        c=group_size**0.5,
        eps=eps,
        use_mean=False,
        num_groups=num_groups,
        return_residual=return_residual,
    )
