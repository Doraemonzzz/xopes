import torch


def chunk_cumsum_reduce_torch(
    x: torch.Tensor,
    dim: int = -1,
    reverse: bool = False,
    chunk_size: int = 128,
    **kwargs,
):
    """
    Convert chunked cumulative sums into a complete cumulative sum result.

    This function takes a tensor that has already been processed by chunk_cumsum
    (where local cumulative sums within each chunk have been calculated), and combines
    these local cumulative sums into a complete cumulative sum result.

    Args:
        x: Input tensor that has already been processed by chunk_cumsum (with local cumulative sums within chunks)
        dim: The dimension along which to compute the cumulative sum
        reverse: If True, compute the cumulative sum in reverse order
        chunk_size: The size of the chunks used for the cumulative sum

    Returns:
        The complete cumulative sum of the input tensor
    """
    dtype = x.dtype
    x = x.float()

    if dim != -1:
        x = x.transpose(dim, -1)

    n = x.shape[-1]
    l = (n + chunk_size - 1) // chunk_size
    array = []
    for i in range(l):
        start = i * chunk_size
        end = min(start + chunk_size, n)
        array.append([start, end])
    if reverse:
        array = array[::-1]

    s = 0
    for index in array:
        start, end = index
        x_ = x[..., start:end]

        if reverse:
            s_ = s + x_[..., 0].unsqueeze(-1)
        else:
            s_ = s + x_[..., -1].unsqueeze(-1)

        x[..., start:end] = x_ + s
        s = s_

    o = x

    if dim != -1:
        o = o.transpose(dim, -1)

    return o.to(dtype)
