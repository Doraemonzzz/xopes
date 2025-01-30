import torch


def cumsum_torch(x, dim=-1, reverse=False):
    """
    Compute the cumulative sum of a tensor along a specified dimension.

    Args:
        x: The input tensor.
        dim: The dimension along which to compute the cumulative sum.
        reverse: If True, compute the cumulative sum in reverse order.

    Returns:
        The cumulative sum of the input tensor.
    """
    if reverse:
        x = torch.flip(x, [dim])
    o = torch.cumsum(x, dim)
    if reverse:
        o = torch.flip(o, [dim])
    return o
