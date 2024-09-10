import torch.nn.functional as F

from xopes.utils import identity

ACT_TORCH_DICT = {
    "relu": F.relu,
    "sigmoid": F.sigmoid,
    "silu": F.silu,
    "none": identity,
}


def act_torch(x, act, dim=None):
    if dim is None:
        fn = ACT_TORCH_DICT[act]

        return fn(x)
