from typing import Optional, Tuple

import torch
from einops import rearrange


def group_srms_norm_torch(
    x: torch.Tensor,
    dim: int,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
    return_residual: bool = False,
    num_groups: int = 1,  # Added parameter for group normalization
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    dtype = x.dtype
    x = x.float()

    # Handle residual connection
    if residual is not None:
        x = x + residual.float()
        residual = x.to(dtype)
    else:
        if return_residual:
            residual = x.to(dtype)

    x_ = rearrange(x, "... (g e) -> ... g e", g=num_groups)

    # Calculate mean and variance across last two dimensions
    x_ = x_ * torch.rsqrt(x_.pow(2).mean(-1, keepdim=True) + eps)

    # Apply weight and bias
    x_ = rearrange(x_, "... g e -> ... (g e)")

    return x_.to(dtype), residual
