from .md_lrpe_cosine_cache_triton import md_lrpe_cosine_cache_triton
from .md_lrpe_cosine_parallel_triton import md_lrpe_cosine_parallel_triton
from .md_lrpe_cosine_torch import md_lrpe_cosine_torch
from .md_lrpe_cosine_triton import md_lrpe_cosine_triton


def md_lrpe_cosine_fn(x, theta, shape=None):
    return md_lrpe_cosine_triton(x, theta, shape)
