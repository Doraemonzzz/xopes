import torch
from einops import rearrange


def normalize_torch(
    x,
    weight=None,
    bias=None,
    residual=None,
    c=1.0,
    eps=1e-5,
    use_mean=False,
    num_groups=1,
):
    assert (
        x.shape[-1] % num_groups == 0
    ), "The last dimension of x must be divisible by num_groups"
    dtype = x.dtype
    x = x.float()

    if weight is not None:
        weight = weight.float()
    if bias is not None:
        bias = bias.float()
    if residual is not None:
        residual = residual.float()
        x = x + residual

    x_ = rearrange(x, "... (g e) -> ... g e", g=num_groups)

    if use_mean:
        x_ = x_ - x_.mean(dim=-1, keepdim=True)

    sigma = torch.sqrt(torch.sum(x_ * x_, dim=-1, keepdim=True) + eps)
    o = c * x_ / sigma

    if weight is not None:
        weight = rearrange(weight, "... (g e) -> ... g e", g=num_groups)
        o = o * weight
    if bias is not None:
        bias = rearrange(bias, "... (g e) -> ... g e", g=num_groups)
        o = o + bias

    o = o.reshape_as(x).to(dtype)

    return o
