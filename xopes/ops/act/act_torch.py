import torch.nn.functional as F

ACT_DICT_TORCH = {
    "relu": F.relu,
    "sigmoid": F.sigmoid,
    "silu": F.silu,
    "none": lambda x: x,
}


def act_torch(act, x, dim=None):
    if dim is None:
        fn = ACT_DICT_TORCH[act]

    return fn(x)
