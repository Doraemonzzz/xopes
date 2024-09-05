import torch


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

    array = shape.tolist()[::-1]
    _basis = []
    c = 1
    for a in array:
        _basis.append(c)
        c *= a
    _basis = _basis[::-1]
    # _basis = torch.cumprod(
    #     torch.tensor([1] + array[:-1]), dim=0, dtype=torch.int32
    # )
    print(_basis)
    _basis = torch.tensor(_basis)
    print(index.shape, _basis.shape)

    indice = (index * _basis.to(x.device)).sum(dim=-1).to(torch.int32)
    print(index)
    print(indice)

    # h, d -> h, ..., d
    for _ in range(m):
        theta = theta.unsqueeze(1)

    theta_list = []
    for i in range(m):
        theta_list.append(index[..., i : i + 1] * theta.float())

    theta = torch.cat(theta_list, dim=-1)[..., :d]

    cos = theta.cos()
    sin = theta.sin()

    print(x.shape, cos.shape, theta.shape)
    output = torch.cat([x * cos, x * sin], dim=-1)

    return output.to(x.dtype)
