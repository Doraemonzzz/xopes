def householder_torch(x, y):
    dtype = x.dtype
    x = x.float()
    y = y.float()
    y_ = F.normalize(y, dim=-1)
    c = (x * y_).sum(dim=-1, keepdim=True)
    o = x - 2 * c * y_

    return o.to(dtype)
