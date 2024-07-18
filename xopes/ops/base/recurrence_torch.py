import torch


def base_rule_recurrence_torch(q, k, v, s=None):
    b, h, n, d = q.shape
    e = v.shape[-1]
    if s is None:
        s = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)

    o = []
    for i in range(n):
        ki = k[:, :, i : i + 1].contiguous().to(torch.float32)
        vi = v[:, :, i : i + 1].contiguous().to(torch.float32)
        qi = q[:, :, i : i + 1].contiguous().to(torch.float32)
        s = s + ki.transpose(-1, -2) * vi
        oi = torch.matmul(qi.to(s.dtype), s)
        o.append(oi)

    return torch.cat(o, dim=-2).to(q.dtype)
