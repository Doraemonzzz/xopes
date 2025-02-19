# lasd: lightning attention scalar decay
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from xopes.ops.act.act_torch import act_torch


def lasd_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
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
    if initial_state is None:
        state = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
    else:
        state = initial_state

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
    o = []
    for i in range(n):
        qi = q[:, i]
        ki = k[:, i]
        vi = v[:, i]
        state_ = torch.einsum("b h d, b h e -> b h d e", ki, vi)
        state = ratio * state + state_
        oi = torch.einsum("b h d, b h d e -> b h e", qi, state)
        o.append(oi)
    o = torch.cat(o, dim=1)

    return o, state


if __name__ == "__main__":
    b, n, h, d = 2, 16, 12, 64
    q = torch.randn(b, n, h, d)
    k = torch.randn(b, n, h, d)
    v = torch.randn(b, n, h, d)
    ld = F.logsigmoid(torch.randn(h))
    output, state = lasd_torch(q, k, v, ld)
