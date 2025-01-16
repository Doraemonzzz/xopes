from ._1d import (
    lrpe_rotate_1d_sp_bwd_triton,
    lrpe_rotate_1d_sp_fwd_triton,
    lrpe_rotate_1d_sp_triton,
)


def lrpe_rotate_fn(x, theta, offset=0, act="none", dim=None, **kwargs):
    return lrpe_rotate_1d_sp_triton(x=x, theta=theta, offset=offset, act=act, dim=dim)


def lrpe_rotate_fwd_fn(x, theta, offset=0, act="none", dim=None, **kwargs):
    return lrpe_rotate_1d_sp_fwd_triton(
        x=x, theta=theta, offset=offset, act=act, dim=dim
    )


def lrpe_rotate_bwd_fn(x, theta, do, offset=0, act="none", dim=None, **kwargs):
    return lrpe_rotate_1d_sp_bwd_triton(
        x=x, theta=theta, do=do, offset=offset, act=act, dim=dim
    )
