from .act_torch import act_torch
from .act_triton import act_bwd_triton, act_fwd_triton, act_triton

act_bwd = act_bwd_triton
act_fwd = act_fwd_triton
act = act_triton
