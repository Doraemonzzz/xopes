import torch
import torch.nn.functional as F
from einops import pack


def md_lrpe_cosine_torch(x, theta, shape, l=0):
    # x: b, h, n, d; n = l + prod(shape)
    # theta: h, next_power_of_two((d + len(shape) - 1) // len(shape))
    # shape: n1, ... , nm
    # l: we do not do lrpe cosine on the first l tokens
    shape = torch.tensor(shape, dtype=torch.int32, device=x.device)
    d = x.shape[-1]
    m = len(shape)

    array = [
        torch.arange(n, dtype=torch.int64, device=torch.cuda.current_device())
        for n in shape
    ]
    grid = torch.meshgrid(array)
    index = torch.stack(grid, dim=-1)

    # h, d -> h, ..., d
    for _ in range(m):
        theta = theta.unsqueeze(1)

    theta_list = []
    for i in range(m):
        theta_list.append(index[..., i : i + 1] * theta.float())

    theta = torch.cat(theta_list, dim=-1)[..., :d]
    theta, ps = pack([theta], "h * d")

    x_no_lrpe = x[:, :, :l]
    x = x[:, :, l:]

    cos = theta.cos()
    sin = theta.sin()

    output = torch.cat([x * cos, x * sin], dim=-1)
    if l > 0:
        output = torch.cat([F.pad(x_no_lrpe, (0, d)), output], dim=-2)

    return output.to(x.dtype)


if __name__ == "__main__":
    # unit test
    from xopes.utils import next_power_of_two

    shape = tuple([2, 8, 32, 32, 64])
    l = 2
    b = shape[0]
    h = shape[1]
    d = shape[-1]
    m = len(shape) - 3
    e = next_power_of_two((d + m - 1) // m)
    dtype = torch.float32
    device = torch.cuda.current_device()

    x = torch.randn(shape, dtype=dtype, device=device)
    x, ps_x = pack([x], "b h * d")
    if l > 0:
        token = torch.randn((b, h, l, d), dtype=dtype, device=device)
        x = torch.cat([token, x], dim=-2)
    x = x.requires_grad_()

    theta = torch.randn((h, e), dtype=dtype, device=device)
    shape = shape[:-1] + (shape[-1] * 2,)

    do = torch.randn(shape, dtype=dtype, device=device)
    do, ps_do = pack([do], "b h * d")
    if l > 0:
        do_token = torch.randn((b, h, l, 2 * d), dtype=dtype, device=device)
        do = torch.cat([do_token, do], dim=-2)

    o = md_lrpe_cosine_torch(x, theta, shape=shape[2:-1], l=l)

    o.backward(do)
