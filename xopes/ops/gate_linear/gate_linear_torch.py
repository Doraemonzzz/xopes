import torch

from xopes.ops.act.act_torch import act_torch


def gate_linear_torch(
    x1: torch.Tensor,
    x2: torch.Tensor,
    W: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    act: str = "none",
) -> torch.Tensor:
    """
    Apply gate linear to the input tensor x1, x2, W, bias, residual.

    Args:
        x1: Input tensor, ... d1
        x2: Input tensor, ... d2
        W: Weight tensor, d2 d1
        bias: Bias tensor, d2
        residual: Residual tensor, ... d2
        act: Activation function
    """
    x1 = act_torch(x1, act)
    y = x1 * x2
    o = F.linear(y, W, bias)
    if residual is not None:
        o = o + residual

    return o
