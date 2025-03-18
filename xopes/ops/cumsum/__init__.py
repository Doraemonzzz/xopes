from typing import Optional

import torch

from .chunk_cumsum import chunk_cumsum_torch, chunk_cumsum_triton
from .chunk_cumsum_reduce import chunk_cumsum_reduce_torch, chunk_cumsum_reduce_triton
from .cumsum_torch import cumsum_torch
from .cumsum_triton import cumsum_triton


def cumsum_fn(
    x: torch.Tensor,
    dim: int = -1,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    return cumsum_triton(x=x, dim=dim, reverse=reverse, cu_seqlens=cu_seqlens)


def chunk_cumsum_fn(
    x: torch.Tensor,
    dim: int = -1,
    reverse: bool = False,
    chunk_size: int = 128,
):
    return chunk_cumsum_triton(x=x, dim=dim, reverse=reverse, chunk_size=chunk_size)


def chunk_cumsum_reduce_fn(
    x: torch.Tensor,
    dim: int = -1,
    reverse: bool = False,
    chunk_size: int = 128,
):
    return chunk_cumsum_reduce_triton(
        x=x, dim=dim, reverse=reverse, chunk_size=chunk_size
    )
