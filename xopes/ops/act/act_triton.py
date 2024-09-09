
from .fn import _silu_bwd, _silu_fwd, silu

ACT_DICT_TRITON = {
    "silu": silu,
}

ACT_DICT_FWD_TRITON = {
    "silu": _silu_fwd,
}

ACT_DICT_BWD_TRITON = {
    "silu": _silu_bwd,
}


def act_triton(act, x, dim=None):
    if dim is None:
        fn = ACT_DICT_TRITON[act]

    return fn(x)


def act_fwd_triton(act, x, dim=None):
    if dim is None:
        fn = ACT_DICT_FWD_TRITON[act]

    return fn(x)


def act_bwd_triton(act, x, dim=None):
    if dim is None:
        fn = ACT_DICT_FWD_TRITON[act]

    return fn(x)
