import torch

try:
    from fla.modules import FusedCrossEntropyLoss
except:
    FusedCrossEntropyLoss = lambda x: None


def cross_entropy_fla_wrapper(
    x: torch.Tensor,  # (b v)
    y: torch.Tensor,  # (b)
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
):
    loss_fn = FusedCrossEntropyLoss(
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
        inplace_backward=True,
    )
    return loss_fn(
        input=x,
        target=y,
    )
