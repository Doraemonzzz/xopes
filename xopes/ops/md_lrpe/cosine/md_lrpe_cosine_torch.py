import torch
from einops import pack


def md_lrpe_cosine_torch(x, theta, shape=None):
    # x: b, h, ..., d
    # theta: h, next_power_of_two((d + len(shape) - 1) // len(shape))
    if shape is None:
        shape = x.shape[2:-1]
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

    # # when x is flatten, we need to pack theta
    # if len(x.shape) == 4:  # b h n d
    #     theta, ps = pack([theta], "h * d")

    cos = theta.cos()
    sin = theta.sin()

    output = torch.cat([x * cos, x * sin], dim=-1)

    return output.to(x.dtype)


if __name__ == "__main__":
    # unit test
    from xopes.utils import next_power_of_two

    def test(use_pack=False):
        shape = tuple([2, 8, 32, 32, 64])
        h = shape[1]
        d = shape[-1]
        m = len(shape) - 3
        e = next_power_of_two((d + m - 1) // m)
        dtype = torch.float32
        device = torch.cuda.current_device()
        x = (torch.randn(shape, dtype=dtype, device=device)).requires_grad_()
        if use_pack:
            x, ps = pack([x], "b h * d")

        theta = torch.randn((h, e), dtype=dtype, device=device)
        shape = shape[:-1] + (shape[-1] * 2,)
        do = torch.randn(shape, dtype=dtype, device=device)
        if use_pack:
            do, ps = pack([do], "b h * d")

        o = md_lrpe_cosine_torch(x, theta, shape=shape[2:-1])
        o.backward(do)

    test(False)
    test(True)
