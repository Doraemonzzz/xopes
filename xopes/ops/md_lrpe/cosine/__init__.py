from .md_lrpe_cosine_cache_triton import (
    md_lrpe_cosine_cache_bwd_triton,
    md_lrpe_cosine_cache_fwd_triton,
    md_lrpe_cosine_cache_triton,
)
from .md_lrpe_cosine_torch import md_lrpe_cosine_torch
from .md_lrpe_cosine_triton import (
    md_lrpe_cosine_bwd_triton,
    md_lrpe_cosine_fwd_triton,
    md_lrpe_cosine_triton,
)


def md_lrpe_cosine_fn(x, theta, shape, l=0):
    return md_lrpe_cosine_cache_triton(x, theta, shape, l)


def md_lrpe_cosine_fwd(x, theta, shape, l=0):
    return md_lrpe_cosine_fwd_triton(x, theta, shape, l)


def md_lrpe_cosine_bwd(x, theta, do, shape, l=0):
    return md_lrpe_cosine_bwd_triton(x, theta, do, shape, l)
