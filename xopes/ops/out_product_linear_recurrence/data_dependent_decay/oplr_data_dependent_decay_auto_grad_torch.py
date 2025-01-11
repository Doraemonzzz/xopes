from typing import Optional

import torch

from xopes.utils import contiguous


class OPLRDataDependentDecayTorch(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, xk, xv, log_decay=None):
        if log_decay is None:
            assert torch.all(xk <= 1), "xk must be all negative when decay is None"

        # Get dimensions
        b, n, d = xk.shape
        e = xv.shape[-1]

        # Initialize output
        o = torch.empty((b, n, d, e), dtype=xk.dtype, device=xk.device)

        # Forward pass loop
        for i in range(n):
            if log_decay is not None:
                decay = torch.exp(log_decay[:, i, :])  # [b, d]
            else:
                decay = 1 - xk[:, i, :]  # [b, d]

            o[:, i] = decay.unsqueeze(-1) * (o[:, i - 1] if i > 0 else 0) + xk[
                :, i, :
            ].unsqueeze(-1) * xv[:, i, :].unsqueeze(1)

        ctx.save_for_backward(xk, xv, log_decay, o)
        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        xk, xv, log_decay, o = ctx.saved_tensors
        b, n, d = xk.shape
        e = xv.shape[-1]

        dxk = torch.empty_like(xk)
        dxv = torch.empty_like(xv)
        d_decay = torch.empty_like(xk)

        # Initialize accumulated gradients
        dkv = torch.zeros((b, d, e), device=xk.device, dtype=torch.float32)

        # Backward pass loop
        for i in range(n - 1, -1, -1):
            if i < n - 1:
                if log_decay is not None:
                    decay = torch.exp(log_decay[:, i + 1, :])
                else:
                    decay = 1 - xk[:, i + 1, :]
                dkv = decay.unsqueeze(-1) * dkv + do[:, i]
            else:
                dkv = do[:, i]

            if i > 0:
                d_decay[:, i] = torch.sum(dkv * o[:, i - 1], dim=-1)
            else:
                d_decay[:, i] = torch.zeros_like(d_decay[:, i])

            dxk[:, i] = torch.sum(dkv * xv[:, i].unsqueeze(1), dim=-1)
            dxv[:, i] = torch.sum(dkv * xk[:, i].unsqueeze(-1), dim=1)

        if log_decay is not None:
            dlog_decay = d_decay * torch.exp(log_decay)
        else:
            dlog_decay = None
            dxk = dxk - d_decay

        return dxk, dxv, dlog_decay


def oplr_data_dependent_decay_auto_grad_torch(
    xk: torch.Tensor,
    xv: torch.Tensor,
    log_decay: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies Out Product Linear Recurrence with data-dependent decay using PyTorch.

    Args:
        xk: Input tensor (B, N, D)
        xv: Expansion vector (B, N, E)
        log_decay: Optional decay tensor (B, N, D)

    Returns:
        Output tensor (B, N, D, E)
    """
    return OPLRDataDependentDecayTorch.apply(xk, xv, log_decay)
