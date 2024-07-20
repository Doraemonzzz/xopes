import torch
from einops import rearrange


# only support no initial state
def additive_rule_recurrence_torch(
    q, k, v, g, initial_state=None, output_final_state=False
):
    b, h, n, d = q.shape
    e = v.shape[-1]
    s = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
    denom = torch.zeros((b, h, 1, d)).to(q.device)
    with torch.no_grad():
        g_max = torch.max(g, dim=-2).values.unsqueeze(-2)
    g = g - g_max
    # p = torch.exp(g)
    # print(torch.min(p).item(), torch.max(p).item())
    o = []
    for i in range(n):
        ki = k[:, :, i : i + 1].contiguous().to(torch.float32)
        vi = v[:, :, i : i + 1].contiguous().to(torch.float32)
        gi = g[:, :, i : i + 1].contiguous().to(torch.float32)
        qi = q[:, :, i : i + 1].contiguous().to(torch.float32)
        k_ = torch.exp(gi) * ki
        s = s + k_.transpose(-1, -2) * vi

        denom = denom + torch.exp(gi)
        oi = torch.matmul(qi.to(s.dtype), s / rearrange(denom, "b h n d -> b h d n"))
        # oi = torch.matmul(qi.to(s.dtype) / denom, s)
        o.append(oi)

    final_state = None
    if output_final_state:
        final_state = s

    return torch.cat(o, dim=-2).to(q.dtype), final_state


def additive_rule_recurrence_stable_torch(
    q, k, v, g, initial_state=None, output_final_state=False
):
    b, h, n, d = q.shape
    e = v.shape[-1]

    if initial_state is not None:
        s = initial_state[0]
        denom = initial_state[1]
        m = initial_state[2]
    else:
        s = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
        denom = torch.zeros((b, h, d, 1)).to(q.device)
        m = torch.ones(b, h, d, 1).to(torch.float32).to(q.device) * (-1e5)

    o = []
    for i in range(n):
        ki = k[:, :, i].contiguous().to(torch.float32).unsqueeze(-1)
        vi = v[:, :, i].contiguous().to(torch.float32).unsqueeze(-1)
        gi = g[:, :, i].contiguous().to(torch.float32).unsqueeze(-1)
        qi = q[:, :, i].contiguous().to(torch.float32).unsqueeze(-1)
        mi = torch.max(m, gi)
        with torch.no_grad():
            gi = gi - mi
        # p = torch.exp(gi)
        # print(torch.min(p).item(), torch.max(p).item())
        # b h d 1
        lambdai = torch.exp(m - mi)

        k_ = torch.exp(gi) * ki
        s = lambdai * s + k_ * vi.transpose(-1, -2)
        # print(lambdai.shape, denom.shape, gi.shape)
        denom = lambdai * denom + torch.exp(gi)
        # print(qi.shape, s.shape, denom.shape)
        oi = torch.matmul(qi.transpose(-1, -2), s / denom)
        # oi = torch.matmul(qi.to(s.dtype) / denom, s)
        o.append(oi)
        m = mi

    final_state = None
    if output_final_state:
        final_state = (s, denom, m)

    return torch.cat(o, dim=-2).to(q.dtype), final_state
