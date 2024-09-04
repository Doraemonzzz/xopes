import torch


def lrpe_cosine_torch(x, theta):
    # x: b, h, n, d
    # theta: h, d
    n = x.shape[-2]
    index = torch.arange(n, device=torch.cuda.current_device(), dtype=torch.int64)
    theta = torch.einsum("h d, n -> h n d", theta.float(), index)
    cos = theta.cos()
    sin = theta.sin()

    output = torch.cat([x * cos, x * sin], dim=-1)

    return output.to(x.dtype)
