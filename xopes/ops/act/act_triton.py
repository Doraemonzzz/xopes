from .triton import (
    act_no_dim_fwd_triton,
    act_no_dim_triton,
    softmax_bwd_triton,
    softmax_fwd_triton,
    softmax_triton,
)


def act_triton(x, act="none", dim=None):
    if dim != None:
        return softmax_triton(x, dim)
    else:
        return act_no_dim_triton(x, act)


def act_fwd_triton(x, act="none", dim=None):
    if dim != None:
        return softmax_fwd_triton(x, dim)
    else:
        return act_no_dim_fwd_triton(x, act)


def act_bwd_triton(x, do, act="none", dim=None):
    if dim != None:
        return softmax_bwd_triton(x, do, dim)
    else:
        return act_no_dim_fwd_triton(x, act)
