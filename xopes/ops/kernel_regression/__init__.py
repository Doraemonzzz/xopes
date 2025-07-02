from typing import Optional, Tuple

import torch

from .causal_linear import krcl_fn


def kernel_regression_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    BLOCK_N: int = 64,
    kernel_type: str = "causal_linear",
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Kernel Regressionin Pytorch.

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
        kernel_type: Kernel type, one of "causal_linear"

    Returns:
        o: Tensor of shape (B, N, H, E)
        state: Tensor of shape (B, H, D, E)
    """
    if kernel_type == "causal_linear":
        fn = krcl_fn
    else:
        raise ValueError(f"Kernel type {kernel_type} not supported")

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
