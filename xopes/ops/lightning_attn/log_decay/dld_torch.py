from typing import Optional

import torch


def compute_dld_torch(
    dld_q: torch.Tensor,  # B N H
    dld_k: torch.Tensor,  # B N H
    final_state: Optional[torch.Tensor] = None,  # B H D E
    dfinal_state: Optional[torch.Tensor] = None,  # B H D E
    cu_seqlens: Optional[torch.Tensor] = None,  # M
):
    """
    Compute the derivative of the log decay.

    Args:
        dld_q: The derivative of the log decay with respect to the query of shape (B, N, H).
        dld_k: The derivative of the log decay with respect to the key of shape (B, N, H).
        final_state: The final state of the recurrence of shape (B, H, D, E).
        dfinal_state: The derivative of the final state of the recurrence of shape (B, H, D, E).
        cu_seqlens: The cumulative sequence lengths of the query of shape (M,).

    Returns:
        dld: The derivative of the log decay.
    """
    dld = dld_q - dld_k

    if dfinal_state is not None:
        # B 1 H
        dld_state = (final_state * dfinal_state).sum(dim=-1).sum(dim=-1).unsqueeze(1)
        dld = dld + dld_state

    return dld
