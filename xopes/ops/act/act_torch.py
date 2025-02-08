from xopes.utils import is_act_valid

from ._act_torch import (
    ACT_BWD_TORCH_DICT,
    ACT_DIM_TORCH_DICT,
    ACT_GRAD_TORCH_DICT,
    ACT_TORCH_DICT,
)


def act_torch(x, act, dim=None):
    is_act_valid(act)
    if dim is None:
        fn = ACT_TORCH_DICT[act]

        return fn(x)
    else:
        fn = ACT_DIM_TORCH_DICT[act]

        return fn(x, dim=dim)


def act_fwd_torch(x, act, dim=None):
    is_act_valid(act)
    if dim is None:
        fn = ACT_TORCH_DICT[act]

        return fn(x)
    else:
        fn = ACT_DIM_TORCH_DICT[act]

        return fn(x, dim=dim)


def act_bwd_torch(x, do, act, dim=None):
    is_act_valid(act)
    if dim is None:
        fn = ACT_BWD_TORCH_DICT[act]
        return fn(x, do)
    else:
        fn = ACT_DIM_BWD_TORCH_DICT[act]

        return fn(x, do, dim=dim)


def act_grad_torch(x, act):
    is_act_valid(act)
    fn = ACT_GRAD_TORCH_DICT[act]
    return fn(x)
