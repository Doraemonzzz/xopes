from typing import Optional

import torch


def cumsum_torch(
    x: torch.Tensor,
    dim: int = -1,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    """
    Compute the cumulative sum of a tensor along a specified dimension.

    Args:
        x: The input tensor.
        dim: The dimension along which to compute the cumulative sum.
        reverse: If True, compute the cumulative sum in reverse order.
        cu_seqlens: The cumulative sequence lengths of the input tensor.

    Returns:
        The cumulative sum of the input tensor.
    """
    dtype = x.dtype
    x = x.float()

    if cu_seqlens is None:
        if reverse:
            x = torch.flip(x, [dim])
        o = torch.cumsum(x, dim)
        if reverse:
            o = torch.flip(o, [dim])
    else:
        b = cu_seqlens.shape[0] - 1
        o = []
        for i in range(b):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            end - start
            x_ = x[start:end]
            if reverse:
                x_ = torch.flip(x_, [dim])
            o_ = torch.cumsum(x_, dim)
            if reverse:
                o_ = torch.flip(o_, [dim])
            o.append(o_)
        o = torch.cat(o, dim=0)
    return o.to(dtype)
