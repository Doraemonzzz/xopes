from typing import Optional

import torch
import torch.nn.functional as F
import triton

from xopes.ops.act.act_torch import act_torch
from xopes.ops.gate_linear.gate_linear_triton import _gate_fn, _gate_linear_bwd
from xopes.utils import contiguous


class GateLinearAutograd(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x1, x2, weight, bias=None, residual=None, act="none"):
        with torch.no_grad():
            y = act_torch(x1, act) * x2
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

        # Prepare shapes and inputs
        x1_ = x1.reshape(-1, x1.shape[-1]).contiguous()
        x2_ = x2.reshape(-1, x2.shape[-1]).contiguous()
        b, d1 = x1_.shape
        d2 = weight.shape[0]
        use_bias = bias is not None
        use_residual = residual is not None
        output_shape = list(x1.shape[:-1]) + [d2]

        # Allocate output
        dx1 = torch.empty_like(x1_)
        dx2 = torch.empty_like(x2_)
        y = torch.empty_like(x1_)
        dw = torch.empty_like(weight)
        do_ = do.reshape(-1, do.shape[-1]).contiguous()
        if use_bias:
            db = do_.sum(dim=0)
        else:
            db = None

        if use_residual:
            dr = do
        else:
            dr = None

        # b d1
        # implement f(x1) * x2
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x1.element_size()
        BLOCK_D = min(MAX_FUSED_SIZE, triton.next_power_of_2(d1))
        if d1 > BLOCK_D:
            raise RuntimeError("Normalize doesn't support feature dim >= 64KB.")

        def grid(meta):
            return (triton.cdiv(b, meta["BLOCK_B"]),)

        grid = (b,)
        _gate_fn[grid](
            X1=x1_,
            X2=x2_,
            O=y,
            B=b,
            D=d1,
            ACT=act,
            BLOCK_D=BLOCK_D,
        )

        def f(do_, y, weight):
            # b d2, b d1 -> d2 d1
            dw = torch.matmul(do_.T, y)
            # b d2, d2 d1 -> b d2
            dy = torch.matmul(do_, weight.to(y.dtype))

            return dw, dy

        dw, dy = f(do_, y, weight)

        _gate_linear_bwd[grid](
            X1=x1_,
            X2=x2_,
            DY=dy,
            DX1=dx1,
            DX2=dx2,
            B=b,
            D=d1,
            ACT=act,
            BLOCK_D=BLOCK_D,
        )

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
