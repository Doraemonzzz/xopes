from typing import Optional

import torch


def compute_dld_with_cumsum_torch(
    dld_q: torch.Tensor,  # B N H F
    dld_k: torch.Tensor,  # B N H F
    final_state: Optional[torch.Tensor] = None,  # B H D E
    dfinal_state: Optional[torch.Tensor] = None,  # B H D E
    cu_seqlens: Optional[torch.Tensor] = None,  # M
    sum_option: Optional[int] = -1,
):
    """
    Compute the derivative of the log decay with cumsum.

    Args:
        dld_q: The derivative of the log decay with respect to the query of shape (B, N, H, F), F could be D or E or NUM_FEATURES.
        dld_k: The derivative of the log decay with respect to the key of shape (B, N, H, F), F could be D or E or NUM_FEATURES.
        final_state: The final state of the recurrence of shape (B, H, D, E).
        dfinal_state: The derivative of the final state of the recurrence of shape (B, H, D, E).
        cu_seqlens: The cumulative sequence lengths of the query of shape (M,).
        sum_option: The option to sum the derivative of the log decay over the dimension,
                    -1: for dld_q and dld_k, sum over the last dimension,
                    0: for final_state and dfinal_state, sum over the e dimension,
                    1: for dfinal_state and dld_k, sum over the d dimension.

    Returns:
        dld: The derivative of the log decay.
    """
    dtype = dld_q.dtype
    dld_q = dld_q.to(torch.float32)
    dld_k = dld_k.to(torch.float32)
    if final_state is not None:
        final_state = final_state.to(torch.float32)
    if dfinal_state is not None:
        dfinal_state = dfinal_state.to(torch.float32)

    dld = dld_q - dld_k
    if sum_option == -1:
        dld = dld.sum(dim=-1)
    dld = torch.flip(dld, dims=[1])
    dld = torch.cumsum(dld, dim=1)
    dld = torch.flip(dld, dims=[1])
    dld = dld.to(dtype)

    if dfinal_state is not None:
        # B 1 H D E
        dld_state = (final_state * dfinal_state).unsqueeze(1)
        if sum_option == -1:  # B 1 H
            dld_state = dld_state.sum(dim=-1).sum(dim=-1)
        elif sum_option == 0:  # B 1 H D
            dld_state = dld_state.sum(dim=-1)
        else:  # B 1 H E
            dld_state = dld_state.sum(dim=-2)

        dld = dld + dld_state

    return dld.to(dtype)
