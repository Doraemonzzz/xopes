from ._1d import (
    lrpe_cosine_1d_bp_bwd_triton,
    lrpe_cosine_1d_bp_fwd_triton,
    lrpe_cosine_1d_bp_triton,
    lrpe_cosine_1d_sp_bwd_triton,
    lrpe_cosine_1d_sp_fwd_triton,
    lrpe_cosine_1d_sp_triton,
    lrpe_cosine_1d_torch,
)


def lrpe_cosine_fn(x, theta, offset=0):
    return lrpe_cosine_1d_sp_triton(x, theta, offset)


def lrpe_cosine_fwd(x, theta, offset=0):
    return lrpe_cosine_1d_sp_fwd_triton(x, theta, offset)


def lrpe_cosine_bwd(x, theta, do, offset=0):
    return lrpe_cosine_1d_sp_bwd_triton(x, theta, do, offset)
