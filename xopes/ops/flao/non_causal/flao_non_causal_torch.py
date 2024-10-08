import torch

from xopes.utils import contiguous


class FusedLinearAttentionOutputGateTorch(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, g):
        kv = torch.einsum("... n d, ... n e -> ... d e", k, v)
        qkv = torch.matmul(q, kv)
        o = g * qkv

        ctx.save_for_backward(q, k, v, g, kv, qkv)

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, v, g, kv, qkv = ctx.saved_tensors

        dg = do * qkv
        dqkv = do * g
        dq = torch.einsum("... n e, ... d e -> ... n d", dqkv, kv)
        dkv = torch.einsum("... n d, ... n e -> ... d e", q, dqkv)
        dk = torch.einsum("... n e, ... d e -> ... n d", v, dkv)
        dv = torch.einsum("... n d, ... d e -> ... n e", k, dkv)

        return dq, dk, dv, dg


def flao_non_causal_torch(q, k, v, g):
    return FusedLinearAttentionOutputGateTorch.apply(q, k, v, g)
