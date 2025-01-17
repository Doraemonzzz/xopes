import torch
import torch.nn.functional as F


def cross_entropy_torch(
    z: torch.Tensor,
    y: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Applies cross entropy loss using PyTorch.

    Args:
        z: Input logits tensor of shape (B, V)
        y: Target indices tensor of shape (B)
        ignore_index: Index to ignore in loss calculation
        reduction: Reduction method
        label_smoothing: Label smoothing factor

    Returns:
        Cross entropy loss tensor
    """
    return F.cross_entropy(
        input=z,
        target=y,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
