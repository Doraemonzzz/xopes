import torch


@torch.compile
def transpose(x, dim0, dim1):
    return x.transpose(dim0, dim1).contiguous()
