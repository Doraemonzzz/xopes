import torch
import torch.nn.functional as F

from xopes.utils import identity, is_act_valid

ACT_TORCH_DICT = {
    "relu": F.relu,
    "sigmoid": F.sigmoid,
    "silu": F.silu,
    "none": identity,
}

ACT_DIM_TORCH_DICT = {
    "softmax": lambda x, dim: F.softmax(x, dim=dim, dtype=torch.float32).to(x.dtype),
}


def act_torch(x, act, dim=None):
    is_act_valid(act)
    if dim is None:
        fn = ACT_TORCH_DICT[act]

        return fn(x)
    else:
        fn = ACT_DIM_TORCH_DICT[act]

        return fn(x, dim=dim)
