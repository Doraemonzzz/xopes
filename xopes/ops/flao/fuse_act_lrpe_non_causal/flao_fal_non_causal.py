import torch

from xopes.ops.act import act_bwd, act_fwd
from xopes.ops.lrpe import lrpe_bwd, lrpe_fwd  # no qa
from xopes.ops.md_lrpe import md_lrpe_bwd, md_lrpe_fwd  # no qa
from xopes.utils import contiguous


class FusedLinearAttentionOutputGateFusedActLrpeTorch(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        # act fn
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
        q_ = act_fwd(q, q_act, q_act_dim)
        k_ = act_fwd(k, k_act, k_act_dim)
        v_ = act_fwd(v, v_act, v_act_dim)
        g_ = act_fwd(g, g_act, g_act_dim)

        # lrpe fwd
        if theta is not None:
            if len(shape.shape) == 1:
                # 1d case
                q_ = lrpe_fwd(q_, theta, offset, lrpe_type)
                k_ = lrpe_fwd(k_, theta, offset, lrpe_type)
            else:
                q_ = md_lrpe_fwd(q_, theta, shape, l, lrpe_type)
                k_ = md_lrpe_fwd(k_, theta, shape, l, lrpe_type)

        kv = torch.einsum("... n d, ... n e -> ... d e", k_, v_)
        qkv = torch.matmul(q_, kv)
        o = g_ * qkv

        if theta is not None:
            ctx.save_for_backward(q, k, v, g, kv, qkv, q_, k_, theta, shape)
            use_theta = True
        else:
            ctx.save_for_backward(q, k, v, g, kv, qkv, q_, k_)
            ctx.theta = theta
            ctx.shape = shape
            use_theta = False
        # act params
        ctx.q_act = q_act
        ctx.q_act_dim = q_act_dim
        ctx.k_act = k_act
        ctx.k_act_dim = k_act_dim
        ctx.v_act = v_act
        ctx.v_act_dim = v_act_dim
        ctx.g_act = g_act
        ctx.g_act_dim = g_act_dim
        # lrpe params
        ctx.lrpe_type = lrpe_type
        ctx.offset = offset
        ctx.l = l
        ctx.use_theta = use_theta

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        # lrpe params
        lrpe_type = ctx.lrpe_type
        offset = ctx.offset
        l = ctx.l
        use_theta = ctx.use_theta
        # act params
        q_act = ctx.q_act
        q_act_dim = ctx.q_act_dim
        k_act = ctx.k_act
        k_act_dim = ctx.k_act_dim
        v_act = ctx.v_act
        v_act_dim = ctx.v_act_dim
        g_act = ctx.g_act
        g_act_dim = ctx.g_act_dim

        if use_theta:
            q, k, v, g, kv, qkv, q_, k_, theta, shape = ctx.saved_tensors
        else:
            q, k, v, g, kv, qkv, q_, k_ = ctx.saved_tensors
            theta = ctx.theta
            shape = ctx.shape

        v_ = act_fwd(v, v_act, v_act_dim)
        g_ = act_fwd(g, g_act, g_act_dim)

        dg_ = do * qkv
        dqkv = do * g_
        dq_ = torch.einsum("... n e, ... d e -> ... n d", dqkv, kv)
        dkv = torch.einsum("... n d, ... n e -> ... d e", q_, dqkv)
        dk_ = torch.einsum("... n e, ... d e -> ... n d", v_, dkv)
        dv_ = torch.einsum("... n d, ... d e -> ... n e", k_, dkv)

        # lrpe bwd
        if theta is not None:
            q_ = act_fwd(q, q_act, q_act_dim)
            k_ = act_fwd(k, k_act, k_act_dim)
            if len(shape.shape) == 1:  # 1d case
                # print("aaa", q_.shape, dq_.shape)
                dq_ = lrpe_bwd(q_, theta, dq_, offset, lrpe_type)
                # print(dq_.shape)
                dk_ = lrpe_bwd(k_, theta, dk_, offset, lrpe_type)
            else:
                dq_ = md_lrpe_bwd(q_, theta, dq_, shape, l, lrpe_type)
                dk_ = md_lrpe_bwd(k_, theta, dk_, shape, l, lrpe_type)

        dq = act_bwd(q, dq_, q_act, q_act_dim)
        dk = act_bwd(k, dk_, k_act, k_act_dim)
        dv = act_bwd(v, dv_, v_act, v_act_dim)
        dg = act_bwd(g, dg_, g_act, g_act_dim)

        return dq, dk, dv, dg, *(None,) * 13


def flao_fal_non_causal_torch(
    q,
    k,
    v,
    g,
    # act fn
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

    return FusedLinearAttentionOutputGateFusedActLrpeTorch.apply(
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

    o = flao_fal_non_causal_torch(
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
