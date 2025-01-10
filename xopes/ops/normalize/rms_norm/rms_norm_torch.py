from typing import Optional, Tuple

import torch


def rms_norm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    dim: int,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
    return_residual: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    dtype = x.dtype
    x = x.float()

    if residual is not None:
        x = x + residual.float()
        residual = x.to(dtype)
    else:
        if return_residual:
            residual = x.to(dtype)

    o = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    return o.to(dtype), residual
