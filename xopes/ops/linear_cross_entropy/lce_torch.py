from typing import Optional

import torch
import torch.nn.functional as F


def linear_cross_entropy_torch(
    x: torch.Tensor,  # (n d)
    y: torch.Tensor,  # (n)
    W: torch.Tensor,  # (e d)
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    logits = torch.matmul(x, W.T)
    return F.cross_entropy(
        input=logits,
        target=y,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
