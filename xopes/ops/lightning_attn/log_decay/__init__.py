from .dld_torch import compute_dld_torch
from .dld_triton import compute_dld_triton
from .log_decay_with_cumsum import (
    compute_dld_with_cumsum_torch,
    compute_dld_with_cumsum_triton,
)

compute_dld_fn = compute_dld_triton
compute_dld_with_cumsum_fn = compute_dld_with_cumsum_triton
