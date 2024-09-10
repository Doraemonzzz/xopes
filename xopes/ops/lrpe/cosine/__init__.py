from .lrpe_cosine_torch import lrpe_cosine_torch
from .lrpe_cosine_triton import (
    lrpe_cosine_bwd_triton,
    lrpe_cosine_fwd_triton,
    lrpe_cosine_triton,
)


def lrpe_cosine_fn(x, theta, offset=0):
    return lrpe_cosine_triton(x, theta, offset)


def lrpe_cosine_fwd(x, theta, offset=0):
    return lrpe_cosine_fwd_triton(x, theta, offset)


def lrpe_cosine_bwd(x, theta, offset=0):
    return lrpe_cosine_bwd_triton(x, theta, offset)
