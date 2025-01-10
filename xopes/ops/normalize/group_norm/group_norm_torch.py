from typing import Optional, Tuple

import torch
from einops import rearrange


def group_norm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
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

    x.shape
    x_ = rearrange(x, "... (g e) -> ... g e", g=num_groups)

    x_ = x_ - x_.mean(-1, keepdim=True)
    o = x_ * torch.rsqrt(x_.pow(2).mean(-1, keepdim=True) + eps)
    # sigma = torch.sqrt(torch.sum(x_ * x_, dim=-1, keepdim=True) + eps)
    # c = x_.shape[-1] ** 0.5
    # o = c * x_ / sigma

    # Apply weight and bias
    o = rearrange(o, "... g e -> ... (g e)")
    o = o * weight + bias

    return o.to(dtype), residual
