import torch


def rmsnorm_torch(x, weight, dim, eps=1e-6, residual=None, return_residual=False):
    dtype = x.dtype
    x = x.float()

    if residual is not None:
        x = x + residual.float()
        residual = x.to(dtype)
    else:
        if return_residual:
            residual = x.to(dtype)

    o = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    return o.to(dtype), residual
