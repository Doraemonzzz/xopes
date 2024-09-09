import torch

from xopes.ops.act import act_bwd_triton, act_fwd_triton
from xopes.ops.lrpe import lrpe_cosine_bwd, lrpe_cosine_fwd  # no qa
from xopes.ops.md_lrpe import md_lrpe_cosine_bwd, md_lrpe_cosine_fwd  # no qa


class FusedLinearAttentionOutputGateFusedActLrpe(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
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
        q_ = act_fwd_triton(q, q_act, q_act_dim)
        k_ = act_fwd_triton(k, k_act, k_act_dim)
        v_ = act_fwd_triton(v, v_act, v_act_dim)
        g_ = act_fwd_triton(g, g_act, g_act_dim)

        # lrpe fwd
        if theta is not None:
            if len(q.shape) == 4 and (
                shape is None or len(shape.shape) == 1
            ):  # 1d case
                q_ = lrpe_cosine_fwd(q_, theta)
                k_ = lrpe_cosine_fwd(k_, theta)
            else:
                q_ = md_lrpe_consine_fn(q_, theta, shape)
                k_ = md_lrpe_consine_fn(k_, theta, shape)

        kv = torch.einsum("... n d, ... n e -> ... d e", k_, v_)
        qkv = torch.matmul(q_, kv)
        o = g_ * qkv

        ctx.save_for_backward(q, k, v, g, kv, qkv, q_, k_)

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, v, g, kv, qkv, q_, k_ = ctx.saved_tensors

        v_ = act_fwd_triton(v, v_act, v_act_dim)
        act_fwd_triton(g, g_act, g_act_dim)

        dg_ = do * qkv
        dqkv = do * g
        dq_ = torch.einsum("... n e, ... d e -> ... n d", dqkv, kv)
        dkv = torch.einsum("... n d, ... n e -> ... d e", q_, dqkv)
        dk_ = torch.einsum("... n e, ... d e -> ... n d", v_, dkv)
        dv_ = torch.einsum("... n d, ... d e -> ... n e", k_, dkv)

        # lrpe bwd
        if theta is not None:
            if len(q.shape) == 4 and (
                shape is None or len(shape.shape) == 1
            ):  # 1d case
                dq_ = lrpe_cosine_bwd(q_, theta, dq_)
                dk_ = lrpe_cosine_bwd(k_, theta, dk_)
            else:
                dq_ = md_lrpe_consine_fn(q_, theta, dq_, shape)
                dk_ = md_lrpe_consine_fn(k_, theta, dk_, shape)

        dq = act_bwd_triton(q, dq_, q_act, q_act_dim)
        dk = act_bwd_triton(k, dk_, k_act, k_act_dim)
        dv = act_bwd_triton(v, dv_, v_act, v_act_dim)
        dg = act_bwd_triton(g, dg_, g_act, g_act_dim)

        return dq, dk, dv, dg, *(None,) * 10
