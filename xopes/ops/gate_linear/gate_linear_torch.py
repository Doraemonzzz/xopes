from typing import Optional

import torch
import torch.nn.functional as F

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


if __name__ == "__main__":
    b, d1, d2 = 4, 64, 32
    device = "cuda"
    x1 = torch.randn(b, d1).to(device).requires_grad_(True)
    x2 = torch.randn(b, d1).to(device).requires_grad_(True)
    weight = torch.randn(d2, d1).to(device).requires_grad_(True)
    bias = torch.randn(d2).to(device).requires_grad_(True)
    residual = torch.randn(b, d2).to(device).requires_grad_(True)
    act = "relu"
    o = gate_linear_torch(x1, x2, weight, bias, residual, act)
    o.sum().backward()
