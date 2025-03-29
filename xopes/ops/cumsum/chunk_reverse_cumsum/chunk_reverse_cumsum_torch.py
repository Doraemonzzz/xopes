import torch


def chunk_reverse_cumsum_torch(
    x: torch.Tensor,
    dim: int = -1,
    chunk_size: int = 128,
    **kwargs,
):
    """
    Compute the chunk reverse cumulative sum of a tensor along a specified dimension.
    if the input is x1, ... , xn, we first pad zero to the last position and drop the first position-> x2, ... , xn, 0,
    then do reverse chunk cumsum, this function is used for linear attention grad computation.

    Args:
        x: The input tensor.
        dim: The dimension along which to compute the cumulative sum.
        chunk_size: The size of the chunks to use for the cumulative sum.

    Returns:
        The cumulative sum of the input tensor.
    """
    dtype = x.dtype
    x = x.float()

    if dim != -1:
        x = x.transpose(dim, -1)

    shape = x.shape
    zero = torch.zeros(list(shape[:-1]) + [1], dtype=dtype, device=x.device)
    x = torch.cat([x, zero], dim=-1)[..., 1:]

    n = x.shape[-1]
    l = (n + chunk_size - 1) // chunk_size
    o = []
    for i in range(l):
        start = i * chunk_size
        end = min(start + chunk_size, n)
        x_ = torch.flip(x[..., start:end], [-1])
        o_ = torch.flip(torch.cumsum(x_, dim=-1), [-1])
        o.append(o_)
    o = torch.cat(o, dim=-1)

    if dim != -1:
        o = o.transpose(dim, -1)

    return o.to(dtype)
