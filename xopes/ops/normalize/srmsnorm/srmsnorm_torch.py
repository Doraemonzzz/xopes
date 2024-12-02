import torch


def srmsnorm_torch(x, eps=1e-8, scale=False):
    def _norm(x, scale=False):
        c = x.shape[-1] ** (-1.0 / 2) if scale else 1
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * c

    output = _norm(x.float()).type_as(x)

    return output
