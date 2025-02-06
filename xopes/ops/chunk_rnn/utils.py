from typing import Optional

import torch


def ln_fused_l2_bwd(
    x: torch.Tensor,
    l2_target: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Batch backward for SRMSNorm fused with L2 loss.

    Args:
        x: Input tensor, shape (B, N, H, D)
        l2_target: L2 target tensor, shape (B, N, H, D)
        scale: Scale tensor, shape (H, D)
        eps: Epsilon for numerical stability

    Returns:
        Gradient tensor, shape (B, N, H, D)
    """
    d = x.shape[-1]

    # Mean and variance computation
    var = (x * x).mean(dim=-1, keepdim=True)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = x / std

    # Scale
    if gamma is not None:
        y = x_hat * gamma
    else:
        y = x_hat

    grad_x_hat = y - l2_target
    z = (
        (1.0 / d)
        * (
            d * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )

    return z
