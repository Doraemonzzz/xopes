import torch
import torch.nn.functional as F


@torch.jit.script
def relu(x):
    return F.relu(x)


@torch.jit.script
def relu_bwd(x, do):
    return torch.where(x >= 0, do, 0)


@torch.jit.script
def sigmoid(x):
    return F.sigmoid(x)


@torch.jit.script
def sigmoid_bwd(x, do):
    sigmoid = F.sigmoid(x)
    return do * sigmoid * (1 - sigmoid)


@torch.jit.script
def silu(x):
    return F.silu(x)


@torch.jit.script
def silu_bwd(x, do):
    sigmoid = F.sigmoid(x)
    return do * sigmoid * (1 + x * (1 - sigmoid))


@torch.jit.script
def none(x):
    return x


@torch.jit.script
def none_bwd(x, do):
    return do


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
