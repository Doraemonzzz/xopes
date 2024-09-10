import torch

from xopes.ops.act import act_torch
from xopes.ops.flao.non_causal import flao_non_causal_fn
from xopes.ops.lrpe import lrpe_fn  # noqa
from xopes.ops.md_lrpe import md_lrpe_fn  # noqa


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
    # lrpe
    theta=None,
    shape=None,
    lrpe_type="cosine",
    offset=0,
    l=0,
):
    if shape is None:
        shape = q.shape[2:-1]
    shape = torch.tensor(shape, dtype=torch.int32, device=q.device)
    # use act fn here
    q = act_torch(q, q_act, q_act_dim)
    k = act_torch(k, k_act, k_act_dim)
    v = act_torch(v, v_act, v_act_dim)
    g = act_torch(g, g_act, g_act_dim)

    # lrpe
    if theta is not None:
        if len(shape.shape) == 1:  # 1d case
            q = lrpe_fn(q, theta, offset, lrpe_type)
            k = lrpe_fn(k, theta, offset, lrpe_type)
        else:
            q = md_lrpe_fn(q, theta, shape, l, lrpe_type)
            k = md_lrpe_fn(k, theta, shape, l, lrpe_type)

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
    # act
    q_act = "silu"
    q_act_dim = None
    k_act = "silu"
    k_act_dim = None
    v_act = "none"
    v_act_dim = None
    g_act = "silu"
    g_act_dim = None
    # lrpe
    theta = torch.randn((h, d), dtype=dtype, device=device)
    shape = None
    lrpe_type = "cosine"
    offset = 0
    l = 0

    o = flao_al_non_causal(
        q,
        k,
        v,
        g,
        q_act,
        q_act_dim,
        k_act,
        k_act_dim,
        v_act,
        v_act_dim,
        g_act,
        g_act_dim,
        theta,
        shape,
        lrpe_type,
        offset,
        l,
    )
