import torch


def page_flip_naive_torch(
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
    assert initial_state is None
    b, n, h, d = q.shape
    e = v.shape[-1]

    state1 = torch.zeros(
        (b, h, d), dtype=torch.float32, device=torch.cuda.current_device()
    )
    state2 = torch.zeros(
        (b, h, d, e), dtype=torch.float32, device=torch.cuda.current_device()
    )

    o = []
    for i in range(n):
        qi = q[:, i].to(torch.float32)
        for j in range(i + 1):
            if k is not None:
                kj = k[:, j].to(torch.float32)
            else:
                kj = None
            vj = v[:, j].to(torch.float32)
            wj = w[:, j].to(torch.float32)

            state1_ = state1 + wj
            if decay_type == "additive":
                decay = state1 / state1_
            else:
                decay = torch.exp(state1 - state1_)

            if kj is not None:
                if use_normalize:
                    s = ((1 - decay) * kj).unsqueeze(-1) * vj.unsqueeze(-2)
                else:
                    s = kj.unsqueeze(-1) * vj.unsqueeze(-2)
            else:
                s = (1 - decay).unsqueeze(-1) * vj.unsqueeze(-2)

            state2 = decay.unsqueeze(-1) * state2 + s
            state1 = state1_

        oi = torch.einsum("... d, ... d e -> ... e", qi, state2)
        o.append(oi.unsqueeze(1))

    o = torch.cat(o, dim=1)

    if o_gate is not None:
        o = o * o_gate

    if output_final_state:
        final_state = [state1, state2]
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
    o, final_state = page_flip_naive_torch(
        q, v, w, k=k, o_gate=o_gate, use_normalize=use_normalize
    )
    print(o.shape)
