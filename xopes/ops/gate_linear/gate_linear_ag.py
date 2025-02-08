from typing import Optional

import torch
import torch.nn.functional as F

from xopes.ops.act import act_fn, act_grad_fn
from xopes.utils import contiguous


class GateLinearAutograd(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x1, x2, weight, bias=None, residual=None, act="none"):
        with torch.no_grad():
            y = act_fn(x1, act) * x2
            o = F.linear(y, weight, bias)
            if residual is not None:
                o = o + residual

        # Save for backward
        ctx.save_for_backward(x1, x2, weight, bias, residual)
        ctx.act = act

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x1, x2, weight, bias, residual = ctx.saved_tensors
        act = ctx.act
        use_bias = bias is not None
        use_residual = residual is not None

        # Prepare shapes and inputs
        x1_ = x1.reshape(-1, x1.shape[-1]).contiguous()
        x2_ = x2.reshape(-1, x2.shape[-1]).contiguous()
        do_ = do.reshape(-1, do.shape[-1]).contiguous()

        if use_bias:
            db = do_.sum(dim=0)
        else:
            db = None

        if use_residual:
            dr = do
        else:
            dr = None

        with torch.no_grad():
            f_x1 = act_fn(x1_, act)
            y = f_x1 * x2_
            # b d2, b d1 -> d2 d1
            dw = torch.matmul(do_.T, y)
            # b d2, d2 d1 -> b d2
            dy = torch.matmul(do_, weight.to(f_x1.dtype))
            df_x1 = act_grad_fn(x1_, act)
            dx1 = df_x1 * x2_ * dy
            dx2 = f_x1 * dy

        return dx1.reshape_as(x1), dx2.reshape_as(x2), dw, db, dr, None, None


def gate_linear_ag(
    x1: torch.Tensor,
    x2: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    act: str = "none",
) -> torch.Tensor:
    """
    Apply gate linear using Autograd with torch and Triton.

    Args:
        x1: Input tensor, ... d1
        x2: Input tensor, ... d1
        weight: Weight tensor, d2 d1
        bias: Bias tensor, d2
        residual: Residual tensor, ... d2
        act: Activation function

    Returns:
        Output tensor, ... d2
    """
    return GateLinearAutograd.apply(x1, x2, weight, bias, residual, act)


if __name__ == "__main__":
    b, d1, d2 = 4, 64, 32
    device = "cuda"
    x1 = torch.randn(b, d1).to(device).requires_grad_(True)
    x2 = torch.randn(b, d1).to(device).requires_grad_(True)
    weight = torch.randn(d2, d1).to(device).requires_grad_(True)
    bias = torch.randn(d2).to(device).requires_grad_(True)
    residual = torch.randn(b, d2).to(device).requires_grad_(True)
    act = "relu"
    o = gate_linear_ag(x1, x2, weight, bias, residual, act)
    o.sum().backward()
