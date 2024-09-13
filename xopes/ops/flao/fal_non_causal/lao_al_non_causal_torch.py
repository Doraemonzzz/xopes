import torch

from xopes.ops.act import act_torch
from xopes.ops.flao.non_causal import flao_non_causal_fn
from xopes.ops.lrpe import lrpe_fn  # noqa
from xopes.ops.md_lrpe import md_lrpe_fn  # noqa


def lao_al_non_causal_torch(
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
    if theta is not None:
        if shape is None:
            shape = q.shape[2:-1]
        shape = torch.tensor(shape, dtype=torch.int32, device=q.device)

    # use act fn here
    q = act_torch(q, act=q_act, dim=q_act_dim)
    k = act_torch(k, act=k_act, dim=k_act_dim)
    v = act_torch(v, act=v_act, dim=v_act_dim)
    g = act_torch(g, act=g_act, dim=g_act_dim)

    # lrpe
    if theta is not None:
        q = lrpe_fn(
            q, theta, offset=offset, act=q_act, dim=q_act_dim, lrpe_type=lrpe_type
        )
        k = lrpe_fn(k, theta, act=k_act, dim=k_act_dim, lrpe_type=lrpe_type)

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

    o = lao_al_non_causal_torch(
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
