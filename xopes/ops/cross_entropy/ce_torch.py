import torch
import torch.nn.functional as F


def cross_entropy_torch(
    z: torch.Tensor,  # (b v)
    y: torch.Tensor,  # (b)
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    return F.cross_entropy(
        input=z,
        target=y,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
