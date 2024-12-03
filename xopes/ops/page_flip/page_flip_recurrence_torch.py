import torch


def page_flip_weight_preprocess_torch(w, initial_state=[None, None]):
    b, n, h, d = w.shape
    if initial_state[0] is not None:
        w = torch.cat([initial_state[0].unsqueeze(1), w], dim=1)
    else:
        w = torch.cat([torch.zeros((b, 1, h, d), device=w.device), w], dim=1)

    w_cum = w.cumsum(dim=1)
    if initial_state[1] is not None:
        w_cum = torch.cat([initial_state[1].unsqueeze(1), w_cum[:, 1:]], dim=1)
    else:
        w_cum = torch.cat(
            [torch.zeros((b, 1, h, d), device=w.device), w_cum[:, 1:]], dim=1
        )

    w_cum_cum = w_cum.cumsum(dim=1)

    return w_cum, w_cum_cum


def page_flip_recurrence_torch(
    q,
    v,
    w,
    k=None,
    o_gate=None,
    initial_state=None,
    output_final_state=False,
    decay_type="additive",
    use_normalize=False,
):
    assert decay_type in ["additive", "multiplicative"]
    if initial_state is None:
        initial_state = [None, None, None, None]
    b, n, h, d = q.shape
    e = v.shape[-1]

    w_cum, w_cum_cum = page_flip_weight_preprocess_torch(
        w, initial_state=(initial_state[0], initial_state[1])
    )
    w_cum_final_state = w_cum[:, -1]
    w_cum_cum_final_state = w_cum_cum[:, -1]
    if decay_type == "multiplicative":
        w_cum_cum = torch.exp(w_cum_cum)

    if initial_state[2] is not None:
        s1 = initial_state[2]
    else:
        s1 = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)

    if initial_state[3] is not None:
        s2 = initial_state[3]
    else:
        s2 = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)

    o = []
    for i in range(n):
        decay = w_cum_cum[:, i].to(torch.float32) / w_cum_cum[:, i + 1].to(
            torch.float32
        )
        # !!! important
        norm_factor = w[:, i].to(torch.float32) / w_cum_cum[:, i + 1].to(torch.float32)
        qi = q[:, i].to(torch.float32)
        if k is not None:
            ki = k[:, i].to(torch.float32)
        else:
            ki = None
        vi = v[:, i].to(torch.float32)
        if ki is not None:
            if use_normalize:
                s1 = decay.unsqueeze(-1) * s1 + (norm_factor * ki).unsqueeze(
                    -1
                ) * vi.unsqueeze(-2)
            else:
                s1 = decay.unsqueeze(-1) * s1 + ki.unsqueeze(-1) * vi.unsqueeze(-2)
        else:
            s1 = decay.unsqueeze(-1) * s1 + norm_factor.unsqueeze(-1) * vi.unsqueeze(-2)

        s2 = decay.unsqueeze(-1) * s2 + s1
        oi = torch.einsum("... d, ... d e -> ... e", qi, s2)
        o.append(oi.unsqueeze(1))

    o = torch.cat(o, dim=1)

    if o_gate is not None:
        o = o * o_gate

    if output_final_state:
        final_state = [w_cum_final_state, w_cum_cum_final_state, s1, s2]
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
    o, final_state = page_flip_recurrence_torch(
        q, v, w, k=k, o_gate=o_gate, use_normalize=use_normalize
    )
    print(o.shape)
