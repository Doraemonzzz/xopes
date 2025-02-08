from xopes.utils import is_act_valid

from ._act_triton import (
    act_no_dim_bwd_triton,
    act_no_dim_fwd_triton,
    act_no_dim_triton,
    softmax_bwd_triton,
    softmax_fwd_triton,
    softmax_triton,
)


def act_triton(x, act="none", dim=None):
    is_act_valid(act)
    if dim != None:
        if act in ["softmax", "softmax_no_cache"]:
            fn = softmax_triton
        return fn(x, dim)
    else:
        return act_no_dim_triton(x, act)


def act_fwd_triton(x, act="none", dim=None):
    is_act_valid(act)
    if dim != None:
        if act in ["softmax"]:
            fn = softmax_fwd_triton
        return fn(x, dim)
    else:
        return act_no_dim_fwd_triton(x, act)


def act_bwd_triton(x, do, act="none", dim=None):
    is_act_valid(act)
    if dim != None:
        if act in ["softmax"]:
            fn = softmax_bwd_triton
        return fn(x, do, dim)
    else:
        return act_no_dim_bwd_triton(x, do, act)
