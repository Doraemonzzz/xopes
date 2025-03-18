from .dld_torch import compute_dld_torch
from .dld_triton import compute_dld_triton

compute_dld_fn = compute_dld_triton
