import torch


def lse_torch(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim, keepdim=keepdim)
