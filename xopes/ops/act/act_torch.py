import torch.nn.functional as F

from xopes.utils import identity

ACT_TORCH_DICT = {
    "relu": F.relu,
    "sigmoid": F.sigmoid,
    "silu": F.silu,
    "none": identity,
}

ACT_DIM_TORCH_DICT = {
    "softmax": F.softmax,
    "softmax_no_cache": F.softmax,
}


def act_torch(x, act, dim=None):
    if dim is None:
        fn = ACT_TORCH_DICT[act]

        return fn(x)
    else:
        fn = ACT_DIM_TORCH_DICT[act]

        return fn(x, dim=dim)
