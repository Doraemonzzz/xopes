import torch


def logcumsumexp_torch(x, dim=-2):
    # x: ...., n, d
    return torch.logcumsumexp(x, dim=dim)
