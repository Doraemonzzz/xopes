import torch


def chunk_cumsum_torch(
    x: torch.Tensor,
    dim: int = -1,
    reverse: bool = False,
    chunk_size: int = 128,
):
    """
    Compute the cumulative sum of a tensor along a specified dimension.

    Args:
        x: The input tensor.
        dim: The dimension along which to compute the cumulative sum.
        reverse: If True, compute the cumulative sum in reverse order.
        chunk_size: The size of the chunks to use for the cumulative sum.

    Returns:
        The cumulative sum of the input tensor.
    """
    dtype = x.dtype
    x = x.float()

    if dim != -1:
        x = x.transpose(dim, -1)

    n = x.shape[-1]
    l = (n + chunk_size - 1) // chunk_size
    o = []
    for i in range(l):
        start = i * chunk_size
        end = min(start + chunk_size, n)
        x_ = x[..., start:end]
        if reverse:
            x_ = torch.flip(x_, [-1])
        o_ = torch.cumsum(x_, dim=-1)
        if reverse:
            o_ = torch.flip(o_, [-1])
        o.append(o_)
    o = torch.cat(o, dim=-1)

    if dim != -1:
        o = o.transpose(dim, -1)

    return o.to(dtype)
