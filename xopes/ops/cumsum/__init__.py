import torch

from .cumsum_torch import cumsum_torch
from .cumsum_triton import cumsum_triton


def cumsum_fn(x: torch.Tensor, dim: int = -1, reverse: bool = False):
    return cumsum_triton(x=x, dim=dim, reverse=reverse)
