from .lce_torch import linear_cross_entropy_torch
from .lce_triton import linear_cross_entropy_triton

linear_cross_entropy_fn = linear_cross_entropy_triton
