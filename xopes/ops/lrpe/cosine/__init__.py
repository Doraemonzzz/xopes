from ._1d import (
    lrpe_cosine_1d_bp_bwd_triton,
    lrpe_cosine_1d_bp_fwd_triton,
    lrpe_cosine_1d_bp_triton,
    lrpe_cosine_1d_sp_bwd_triton,
    lrpe_cosine_1d_sp_fwd_triton,
    lrpe_cosine_1d_sp_triton,
    lrpe_cosine_1d_torch,
)


def lrpe_cosine_fn(x, theta, offset=0, act="none", dim=None):
    if dim == -2:
        fn = lrpe_cosine_1d_bp_triton
    else:
        fn = lrpe_cosine_1d_sp_triton
    return fn(x=x, theta=theta, offset=offset, act=act, dim=dim)


def lrpe_cosine_fwd(x, theta, offset=0, act="none", dim=None):
    if dim == -2:
        fn = lrpe_cosine_1d_bp_fwd_triton
    else:
        fn = lrpe_cosine_1d_sp_fwd_triton
    return fn(x=x, theta=theta, offset=offset, act=act, dim=dim)


def lrpe_cosine_bwd(x, theta, do, offset=0, act="none", dim=None):
    if dim == -2:
        fn = lrpe_cosine_1d_bp_bwd_triton
    else:
        fn = lrpe_cosine_1d_sp_bwd_triton
    return fn(x=x, theta=theta, do=do, offset=offset, act=act, dim=dim)
