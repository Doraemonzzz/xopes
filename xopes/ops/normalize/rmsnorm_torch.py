import torch


def rmsnorm_torch(x, weight, dim, eps=1e-6, residual=None):
    if residual is not None:
        x = x + residual

    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
