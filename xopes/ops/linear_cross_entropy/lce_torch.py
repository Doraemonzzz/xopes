from typing import Optional

import torch
import torch.nn.functional as F


def linear_cross_entropy_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    W: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    logits = F.linear(x, W, bias)
    return F.cross_entropy(
        input=logits,
        target=y,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
