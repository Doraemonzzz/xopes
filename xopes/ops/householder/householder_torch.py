import torch


def householder_torch(x, y, eps=1e-5):
    """
    Applies Householder transformation using PyTorch.

    Args:
        x: Input tensor
        y: Direction vector

    Returns:
        Transformed tensor
    """
    dtype = x.dtype
    x = x.float()
    y = y.float()
    sigma = torch.sqrt(torch.sum(y * y, dim=-1, keepdim=True) + eps)
    y_ = y / sigma
    c = (x * y_).sum(dim=-1, keepdim=True)
    o = x - 2 * c * y_

    return o.to(dtype)
