import torch

from .gate_linear_ag import gate_linear_ag
from .gate_linear_torch import gate_linear_torch
from .gate_linear_triton import gate_linear_triton

fn = torch.compile(gate_linear_ag)


def gate_linear_fn(x1, x2, W, bias=None, residual=None, act="none"):
    return fn(x1, x2, W, bias, residual, act)
