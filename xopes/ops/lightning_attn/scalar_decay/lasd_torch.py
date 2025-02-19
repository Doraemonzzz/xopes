# lasd: lightning attention scalar decay
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import repeat

from xopes.ops.act.act_torch import act_torch


def lasd_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    q_act: str = "none",
    k_act: str = "none",
    v_act: str = "none",
    q_norm: bool = False,
    k_norm: bool = False,
    v_norm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Lightning Attention with Scalar Decay in Pytorch.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (H,)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training
        q_act: Activation function for query
        k_act: Activation function for key
        v_act: Activation function for value
        q_norm: Normalize query
        k_norm: Normalize key
        v_norm: Normalize value

    Returns:
        output: Tensor of shape (B, N, H, E)
        state: Tensor of shape (B, H, D, E)
    """
    b, n, h, d = q.shape
    e = v.shape[-1]
    dtype = q.dtype
    q = q.float()
    k = k.float()
    v = v.float()

    if len(ld.shape) == 1:
        ld = ld.unsqueeze(-1).unsqueeze(-1)

    if q_norm:
        q = F.normalize(q, p=2, dim=-1)
    if k_norm:
        k = F.normalize(k, p=2, dim=-1)
    if v_norm:
        v = F.normalize(v, p=2, dim=-1)

    q = act_torch(q, q_act)
    k = act_torch(k, k_act)
    v = act_torch(v, v_act)
    ratio = torch.exp(ld)

    if cu_seqlens is None:
        if initial_state is None:
            state = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
        else:
            state = initial_state
            if len(state.shape) == 3:
                state = repeat(state, "h d e -> b h d e", b=b)

        o = []
        for i in range(n):
            qi = q[:, i]
            ki = k[:, i]
            vi = v[:, i]
            state_ = torch.einsum("b h d, b h e -> b h d e", ki, vi)
            state = ratio * state + state_
            oi = torch.einsum("b h d, b h d e -> b h e", qi, state)
            o.append(oi.unsqueeze(1))
        o = torch.cat(o, dim=1)
    else:
        assert b == 1, "cu_seqlens is only supported for batch size 1"
        q = q.squeeze(0)
        k = k.squeeze(0)
        v = v.squeeze(0)
        b = cu_seqlens.shape[0] - 1

        if initial_state is None:
            state = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
        else:
            state = initial_state
            if state.shape[0] == 1:
                state = state.squeeze(0)
            if len(state.shape) == 3:
                state = repeat(state, "h d e -> b h d e", b=b)

        o = []
        state_array = []
        for i in range(b):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            m = end - start
            q_ = q[start:end]
            k_ = k[start:end]
            v_ = v[start:end]
            state_ = state[i]
            o_array = []
            for j in range(m):
                qi = q_[j]
                ki = k_[j]
                vi = v_[j]
                state__ = torch.einsum("h d, h e -> h d e", ki, vi)
                state_ = ratio * state_ + state__
                oi = torch.einsum("h d, h d e -> h e", qi, state_)
                o_array.append(oi.unsqueeze(0))
            o.append(torch.cat(o_array, dim=0))
            state_array.append(state_.unsqueeze(0))
        o = torch.cat(o, dim=0).unsqueeze(0)
        state = torch.cat(state_array, dim=0)

    return o.to(dtype), state


if __name__ == "__main__":
    b, n, h, d = 2, 16, 12, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    q = torch.randn(b, n, h, d, device=device, dtype=dtype)
    k = torch.randn(b, n, h, d, device=device, dtype=dtype)
    v = torch.randn(b, n, h, d, device=device, dtype=dtype)
    ld = F.logsigmoid(torch.randn(h, device=device))
    output, state = lasd_torch(q, k, v, ld)
