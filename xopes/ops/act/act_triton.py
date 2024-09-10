from xopes.utils import identity

from .fn import _silu_bwd, _silu_fwd, silu

ACT_TRITON_DICT = {
    "silu": silu,
    "none": identity,
}

ACT_FWD_TRITON_DICT = {
    "silu": _silu_fwd,
    "none": identity,
}

ACT_BWD_TRITON_DICT = {
    "silu": _silu_bwd,
    "none": identity,
}


def act_triton(x, act, dim=None):
    if dim is None:
        fn = ACT_TRITON_DICT[act]

    return fn(x)


def act_fwd_triton(x, act, dim=None):
    if dim is None:
        fn = ACT_FWD_TRITON_DICT[act]

        return fn(x)


def act_bwd_triton(x, do, act, dim=None):
    if dim is None:
        fn = ACT_BWD_TRITON_DICT[act]

        return fn(x)
