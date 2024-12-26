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

    x = rearrange(x, "... (g d) -> ... g d", g=num_groups)

    if use_mean:
        x = x - x.mean(dim=-1, keepdim=True)

    sigma = torch.sqrt(
        torch.einsum("... g d, ... g d -> ... g", x, x) / num_groups + eps
    )
    x = c * x / sigma

    if weight is not None:
        x = x * weight
    if bias is not None:
        x = x + bias

    return x.to(dtype)
