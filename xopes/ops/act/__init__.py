import torch

from .act_torch import act_bwd_torch, act_fwd_torch, act_grad_torch, act_torch
from .act_triton import act_bwd_triton, act_fwd_triton, act_triton

act_bwd_fn = act_bwd_torch
act_fwd_fn = act_fwd_torch
act_fn = act_torch
act_grad_fn = act_grad_torch
