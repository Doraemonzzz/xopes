import torch

from .ewbo_torch import ewbo_torch
from .ewbo_triton import ewbo_triton

ewbo_fn = ewbo_triton
ewbo_fwd_fn = torch.compile(ewbo_torch)
