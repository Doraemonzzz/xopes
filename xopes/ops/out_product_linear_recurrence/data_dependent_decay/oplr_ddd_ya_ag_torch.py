from typing import Optional

import torch

from xopes.utils import contiguous


class OplrDddYaAgTorch(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, xk, xv, log_decay=None):
        if log_decay is None:
            assert torch.all(xk <= 1), "xk must be all negative when decay is None"

        dtype = xk.dtype
        xk = xk.float()
        xv = xv.float()
        if log_decay is not None:
            log_decay = log_decay.float()

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
        ctx.dtype = dtype
        return o.to(dtype)

    @staticmethod
    @contiguous
    def backward(ctx, do):
        xk, xv, log_decay, o = ctx.saved_tensors
        dtype = ctx.dtype
        b, n, d = xk.shape
        e = xv.shape[-1]
        do = do.float()

        dxk = torch.empty_like(xk)
        dxv = torch.empty_like(xv)

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

            dxk[:, i] = torch.sum(dkv * xv[:, i].unsqueeze(1), dim=-1)
            dxv[:, i] = torch.sum(dkv * xk[:, i].unsqueeze(-1), dim=1)

        dbeta = torch.sum(o * do, dim=-1) - xk * dxk
        dlog_decay_ = torch.flip(
            torch.cumsum(torch.flip(dbeta, dims=(1,)), dim=1), dims=(1,)
        )

        if log_decay is not None:
            dlog_decay = dlog_decay_
            dlog_decay = dlog_decay.to(dtype)
        else:
            dlog_decay = None
            d_decay = dlog_decay_ / (1 - xk)
            dxk = dxk - d_decay

        return dxk.to(dtype), dxv.to(dtype), dlog_decay


def oplr_ddd_ya_ag_torch(
    xk: torch.Tensor,
    xv: torch.Tensor,
    log_decay: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies Out Product Linear Recurrence with data-dependent decay using Yet Another AutoGrad PyTorch.

    Args:
        xk: Expansion vector of shape (B, N, D)
        xv: Input tensor of shape (B, N, E)
        log_decay: Optional decay tensor of shape (B, N, D)

    Returns:
        Output tensor of shape (B, N, D, E)
    """
    return OplrDddYaAgTorch.apply(xk, xv, log_decay)


if __name__ == "__main__":
    import torch.nn.functional as F

    b, n, d, e = 2, 512, 128, 128
    dtype = torch.bfloat16
    xv = torch.randn((b, n, e), dtype=dtype).cuda()
    xk = torch.randn((b, n, d), dtype=dtype).cuda()
    log_decay = F.logsigmoid(torch.randn((b, n, d), dtype=dtype).cuda())
    o = oplr_ddd_ya_ag_torch(xk, xv, log_decay)
    print(o.shape)
