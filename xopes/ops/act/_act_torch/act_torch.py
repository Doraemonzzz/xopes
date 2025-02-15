import torch
import torch.nn.functional as F


def relu(x, dim=None):
    return F.relu(x)


def relu_bwd(x, do, dim=None):
    return torch.where(x >= 0, do, 0)


def grad_relu(x, dim=None):
    return torch.where(x >= 0, 1, 0)


def sigmoid(x, dim=None):
    return F.sigmoid(x)


def sigmoid_bwd(x, do, dim=None):
    sigmoid = F.sigmoid(x)
    return do * sigmoid * (1 - sigmoid)


def grad_sigmoid(x, dim=None):
    sigmoid = F.sigmoid(x)
    return sigmoid * (1 - sigmoid)


def silu(x, dim=None):
    return F.silu(x)


def silu_bwd(x, do, dim=None):
    sigmoid = F.sigmoid(x)
    return do * sigmoid * (1 + x * (1 - sigmoid))


def grad_silu(x):
    sigmoid = F.sigmoid(x)
    return sigmoid * (1 + x * (1 - sigmoid))


def none(x):
    return x


def none_bwd(x, do):
    return do


def grad_none(x):
    return 1


ACT_TORCH_DICT = {
    "relu": relu,
    "sigmoid": sigmoid,
    "silu": silu,
    "none": none,
}

ACT_BWD_TORCH_DICT = {
    "relu": relu_bwd,
    "sigmoid": sigmoid_bwd,
    "silu": silu_bwd,
    "none": none_bwd,
}

ACT_DIM_TORCH_DICT = {
    "softmax": lambda x, dim: F.softmax(x, dim=dim, dtype=torch.float32).to(x.dtype),
}

ACT_GRAD_TORCH_DICT = {
    "relu": grad_relu,
    "sigmoid": grad_sigmoid,
    "silu": grad_silu,
    "none": grad_none,
}
