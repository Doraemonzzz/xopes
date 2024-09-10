from xopes.utils import identity, identity_bwd, identity_fwd

from .fn import silu, silu_bwd, silu_fwd

ACT_TRITON_DICT = {
    "silu": silu,
    "none": identity,
}

ACT_FWD_TRITON_DICT = {
    "silu": silu_fwd,
    "none": identity_fwd,
}

ACT_BWD_TRITON_DICT = {
    "silu": silu_bwd,
    "none": identity_bwd,
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

        return fn(x, do)
