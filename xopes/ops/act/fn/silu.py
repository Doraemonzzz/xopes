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
        o = _silu_fwd(x)

        ctx.save_for_backward(x)

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x = ctx.saved_tensors

        _silu_bwd(x, do)

        return x


def silu(
    x,
    **kwargs,
):
    return Silu.apply(x)
