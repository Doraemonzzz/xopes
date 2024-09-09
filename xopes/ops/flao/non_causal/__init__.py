from .flao_non_causal_torch import flao_non_causal_torch
from .flao_non_causal_triton import flao_non_causal_triton
from .lao_non_causal_torch import lao_non_causal_torch


def flao_non_causal_fn(q, k, v, g):
    return flao_non_causal_torch(q, k, v, g)
