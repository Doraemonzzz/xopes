from typing import Optional

import torch
from einops import repeat


def lcse_torch(
    x: torch.Tensor,
    dim: int = -1,
    initial_state: Optional[torch.Tensor] = None,
):
    """
    Apply logcumsumexp on the dim dimension of x.

    Args:
        x: Input tensor of shape (...)
        dim: Dimension to apply the operation on
        initial_state: Initial state, the same shape as x, except the dim dimension, which is 1

    Returns:
        output: Tensor of shape (...)
    """
    if dim != -1:
        x = x.transpose(dim, -1)
        if initial_state is not None and len(initial_state.shape) > 1:
            initial_state = initial_state.transpose(dim, -1)

    # reshape input data into 2D tensor
    shape = list(x.shape)
    x = x.reshape(-1, x.shape[-1]).contiguous()

    if initial_state is not None and len(initial_state.shape) == 1:
        initial_state = repeat(initial_state, "n -> b n", b=x.shape[0])

    offset = 0
    if initial_state is not None:
        x = torch.cat([initial_state, x], dim=-1)
        offset = 1

    o = torch.logcumsumexp(x, dim=-1)[..., offset:]
    state = x[..., -1:]
    o = o.reshape(shape)
    state = state.reshape(shape[:-1] + [1])

    if dim != -1:
        x = x.transpose(dim, -1)
        state = state.transpose(dim, -1)

    return x, state


if __name__ == "__main__":
    x = torch.randn(1, 2, 3).cuda().requires_grad_()
    initial_state = torch.randn(1).cuda().requires_grad_()
    o, state = lcse_torch(x, initial_state=initial_state)
    print(o)
    print(state)
