import torch
from einops import repeat


def page_flip_additive_weight_preprocess_torch(w, initial_state=[None, None]):
    b, n, h, d = w.shape
    if initial_state[0] is not None:
        if len(initial_state[0].shape) == 3: # b h d
            state0 = initial_state[0].unsqueeze(1)
        else:
            state0 = repeat(initial_state[0], "h d -> b n h d", b=b, n=1)
        w = torch.cat([state0, w], dim=1)
    else:
        w = torch.cat([torch.zeros((b, 1, h, d), device=w.device), w], dim=1)

    u = w.cumsum(dim=1)
    if initial_state[1] is not None:
        if len(initial_state[1].shape) == 3: # b h d
            state1 = initial_state[1].unsqueeze(1)
        else:
            state1 = repeat(initial_state[1], "h d -> b n h d", b=b, n=1)
        u_ = torch.cat([state1, u[:, 1:]], dim=1)
    else:
        u_ = torch.cat([torch.zeros((b, 1, h, d), device=w.device), u[:, 1:]], dim=1)

    s = u_.cumsum(dim=1)

    return u, s


def page_flip_additive_recurrence_torch(
    q,
    v,
    w,
    k=None,
    o_gate=None,
    initial_state=None,
    output_final_state=False,
    use_normalize=False,
):
    if initial_state is None:
        initial_state = [None, None, None, None]
    b, n, h, d = q.shape
    e = v.shape[-1]

    u, s = page_flip_additive_weight_preprocess_torch(
        w, initial_state=(initial_state[0], initial_state[1])
    )
    state0 = u[:, -1]
    state1 = s[:, -1]

    if initial_state[2] is not None:
        state2 = initial_state[2]
    else:
        state2 = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)

    if initial_state[3] is not None:
        state3 = initial_state[3]
    else:
        state3 = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)

    o = []
    for i in range(n):
        decay_state2 = u[:, i].to(torch.float32) / u[:, i + 1].to(torch.float32)
        decay_state3 = s[:, i].to(torch.float32) / s[:, i + 1].to(torch.float32)
        qi = q[:, i].to(torch.float32)
        if k is not None:
            ki = k[:, i].to(torch.float32)
        else:
            ki = None
        vi = v[:, i].to(torch.float32)

        if use_normalize:
            if ki is not None:
                state = ((1 - decay_state2) * ki).unsqueeze(-1) * vi.unsqueeze(-2)
            else:
                state = (1 - decay_state2).unsqueeze(-1) * vi.unsqueeze(-2)
        else:
            state = ki.unsqueeze(-1) * vi.unsqueeze(-2)

        state2 = decay_state2.unsqueeze(-1) * state2 + state
        state3 = (
            decay_state3.unsqueeze(-1) * state3
            + (1 - decay_state3).unsqueeze(-1) * state2
        )
        oi = torch.einsum("... d, ... d e -> ... e", qi, state3)
        o.append(oi.unsqueeze(1))

    o = torch.cat(o, dim=1)

    if o_gate is not None:
        o = o * o_gate

    if output_final_state:
        final_state = [u, s, state2, state3]
    else:
        final_state = None

    return o, final_state


if __name__ == "__main__":
    import torch

    b, n, h, d, e = 2, 512, 8, 128, 64
    dtype = torch.float32
    q = torch.randn((b, n, h, d), dtype=dtype).cuda()
    k = torch.randn((b, n, h, d), dtype=dtype).cuda()
    v = torch.randn((b, n, h, e), dtype=dtype).cuda()
    w = torch.randn((b, n, h, d), dtype=dtype).cuda()
    o_gate = torch.randn((b, n, h, e), dtype=dtype).cuda()
    k = None
    use_normalize = True
    o, final_state = page_flip_additive_recurrence_torch(
        q, v, w, k=k, o_gate=o_gate, use_normalize=use_normalize
    )
    print(o.shape)
