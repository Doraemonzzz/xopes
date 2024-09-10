import torch
import torch.nn.functional as F

from xopes.utils import contiguous


def _silu_fwd(x):
    return F.silu(x)


def _silu_bwd(x, do):
    return do * F.sigmoid(x) * (1 + x * (1 - F.sigmoid(x)))


class Silu(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x):
        o = silu_fwd(x)

        ctx.save_for_backward(x)

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x = ctx.saved_tensors

        dx = silu_bwd(x, do)

        return dx


def silu_fwd(x):
    o = _silu_fwd(x)

    return o


def silu_bwd(x, do):
    dx = _silu_bwd(x, do)

    return dx


def silu(
    x,
    **kwargs,
):
    return Silu.apply(x)
