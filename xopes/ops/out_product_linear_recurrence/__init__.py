from typing import Optional

import torch

from .data_dependent_decay import (
    oplr_ddd_ag_torch,
    oplr_ddd_torch,
    oplr_ddd_triton,
    oplr_ddd_ya_ag_torch,
)
from .oplr_no_decay_torch import oplr_no_decay_torch

oplr_ddd_fn = torch.compile(oplr_ddd_torch)
oplr_no_decay_fn = torch.compile(oplr_no_decay_torch)


def oplr_fn(
    xk: torch.Tensor,  # b n d
    xv: torch.Tensor,  # b n e
    log_decay: Optional[torch.Tensor],  # b n d
    decay_type: str = "no_decay",
) -> torch.Tensor:  # b n d e
    """
    Applies Out Product Linear Recurrence with data-dependent decay.

    Args:
        xv: Input tensor
        xk: Expansion vector

    Returns:
        Output tensor
    """
    assert decay_type in ["no_decay", "data_dependent_decay"], "Invalid decay type"
    if decay_type == "no_decay":
        return oplr_no_decay_fn(
            xk=xk,
            xv=xv,
        )
    else:
        return oplr_ddd_fn(
            xk=xk,
            xv=xv,
            log_decay=log_decay,
        )
