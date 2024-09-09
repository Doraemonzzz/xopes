import torch

from xopes.ops.act import act_torch
from xopes.ops.flao.non_causal import flao_non_causal_fn


def flao_al_non_causal(
    q,
    k,
    v,
    g,
    q_act="none",
    q_act_dim=None,
    k_act="none",
    k_act_dim=None,
    v_act="none",
    v_act_dim=None,
    g_act="none",
    g_act_dim=None,
    theta=None,
    shape=None,
):
    # use act fn here
    q = act_torch(q, q_act, q_act_dim)
    k = act_torch(k, k_act, k_act_dim)
    v = act_torch(v, v_act, v_act_dim)
    g = act_torch(g, g_act, g_act_dim)

    # use lrpe
    if theta is not None:
        if len(q.shape) == 4:  # 1d case
            q = lrpe_consine_fn(q, theta)
            k = lrpe_consine_fn(k, theta)
        else:
            q = md_lrpe_consine_fn(q, theta, shape)
            k = md_lrpe_consine_fn(k, theta, shape)

    return flao_non_causal_fn(q, k, v, g)


if __name__ == "__main__":
    # unit test
    dtype = torch.bfloat16
    device = torch.cuda.current_device()

    b, h, n, m, d, e = (6, 8, 512, 256, 128, 64)
    q = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((b, h, m, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, h, m, e), dtype=dtype, device=device).requires_grad_()
    g = torch.randn((b, h, n, e), dtype=dtype, device=device).requires_grad_()

    o = flao_al_non_causal(q, k, v, g)
