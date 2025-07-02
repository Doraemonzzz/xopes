from typing import Optional, Tuple

import torch

from .krcl_parallel_triton import krcl_parallel_triton
from .krcl_recurrence_triton import krcl_recurrence_triton
from .krcl_torch import krcl_torch
from .torch_utils import krcl_inverse_torch


def krcl_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    BLOCK_N: int = 64,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Kernel Regression with Causal Linear in Pytorch.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (B, N, H)
        alpha: Alpha tensor of shape (B, N, H)
        beta: Beta tensor of shape (B, N, H)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training
        BLOCK_N: Block size for parallelization

    Returns:
        o: Tensor of shape (B, N, H, E)
        state: Tensor of shape (B, H, D, E)
    """
    if k.shape[1] > 1:
        fn = krcl_parallel_triton
    else:
        fn = krcl_recurrence_triton

    return fn(
        q=q,
        k=k,
        v=v,
        ld=ld,
        alpha=alpha,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        BLOCK_N=BLOCK_N,
        **kwargs,
    )
