import torch


def layernorm_torch(
    x, weight, bias, dim, eps=1e-6, residual=None, return_residual=False
):
    dtype = x.dtype
    x = x.float()

    if residual is not None:
        x = x + residual.float()
        residual = x.to(dtype)
    else:
        if return_residual:
            residual = x.to(dtype)

    x_ = x - x.mean(-1, keepdim=True)
    o = x_ * torch.rsqrt(x_.pow(2).mean(-1, keepdim=True) + eps) * weight + bias

    return o.to(dtype), residual
