import torch

from .act_torch import act_torch
from .act_triton import act_bwd_triton, act_fwd_triton, act_triton

act_bwd_fn = act_bwd_triton
act_fwd_fn = act_fwd_triton
act_fn = torch.compile(act_torch)
