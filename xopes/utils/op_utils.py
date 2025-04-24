import torch
import triton
import triton.language as tl


@torch.compile
def transpose(x, dim0, dim1):
    return x.transpose(dim0, dim1).contiguous()


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))
