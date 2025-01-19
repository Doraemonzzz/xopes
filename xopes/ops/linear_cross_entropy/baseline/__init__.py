from typing import Optional

import torch
import torch.nn.functional as F

from .linear_cross_entropy import linear_cross_entropy_jg

try:
    from fla.modules import FusedLinearCrossEntropyLoss
except:
    FusedLinearCrossEntropyLoss = None

try:
    from fla.modules import FusedCrossEntropyLoss
except:
    FusedCrossEntropyLoss = lambda x: None

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except:
    LigerFusedLinearCrossEntropyLoss = lambda x: None

try:
    from cut_cross_entropy import linear_cross_entropy
except:
    linear_cross_entropy = None

from xopes.ops.cross_entropy import cross_entropy_fn


def linear_cross_entropy_fla_wrapper(
    x: torch.Tensor,  # (b d)
    y: torch.Tensor,  # (b)
    W: torch.Tensor,  # (v d)
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
):
    loss_fn = FusedLinearCrossEntropyLoss()
    return loss_fn(
        x=x,
        target=y,
        weight=W,
    )


def linear_cross_entropy_liger_wrapper(
    x: torch.Tensor,  # (b d)
    y: torch.Tensor,  # (b)
    W: torch.Tensor,  # (v d)
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
):
    loss_fn = LigerFusedLinearCrossEntropyLoss()
    return loss_fn(
        lin_weight=W,
        _input=x,
        target=y,
    )


def linear_cross_entropy_cut_wrapper(
    x: torch.Tensor,  # (b d)
    y: torch.Tensor,  # (b)
    W: torch.Tensor,  # (v d)
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
):
    loss_fn = linear_cross_entropy
    return loss_fn(
        e=x,
        targets=y,
        c=W,
    )


def linear_cross_entropy_jg_wrapper(
    x: torch.Tensor,  # (b d)
    y: torch.Tensor,  # (b)
    W: torch.Tensor,  # (v d)
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
):
    loss_fn = linear_cross_entropy_jg
    return loss_fn(
        x=x,
        y=y,
        At=W.T,
        ignore_index=ignore_index,
    )[0]


def linear_cross_entropy_xopes_wrapper(
    x: torch.Tensor,  # (b d)
    y: torch.Tensor,  # (b)
    W: torch.Tensor,  # (v d)
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
):
    z = F.linear(x, W)
    loss_fn = cross_entropy_fn
    return loss_fn(
        z=z,
        y=y,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
