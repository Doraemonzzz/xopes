from typing import Optional

import torch
from einops import rearrange

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None

from ..utils import ln_fused_l2_bwd


def chunk_rnn_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_f: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    gradient_type: int = 0,
    chunk_size: int = 128,
):
    """
    Applies parallel chunk RNN in Pytorch.

    Args:
        q: Query tensor, shape (B, N, H, D)
        k: Key tensor, shape (B, N, H, D)
        v: Value tensor, shape (B, N, H, D)
        log_f: Log frequency tensor, shape (B, N, H, D)
        initial_state: Initial state tensor, shape (B, H, D, D) or (H, D, D)
        scale: Scale tensor, shape (H, D)
        gradient_type:
            0: v - k * s0
            1: v - k - k * s0
            2: v - normalize(k * s0)
            3: v - k - normalize(k * s0)
            4: v
        chunk_size: Chunk size

    Returns:
        Output tensor, shape (B, N, H, D)
        State tensor, shape (B, N, H, D)
    """
    b, n, h, d = q.shape
    c = chunk_size
    if initial_state is None:
        state = torch.zeros((b, h, d, d), dtype=q.dtype, device=q.device)
    else:
        state = initial_state
    o = []

    m = (n + c - 1) // c

    if n % c != 0:
        l = c - n % c
        q = F.pad(q, (0, 0, 0, 0, 0, l))
        k = F.pad(k, (0, 0, 0, 0, 0, l))
        v = F.pad(v, (0, 0, 0, 0, 0, l))
        if log_f is not None:
            log_f = F.pad(log_f, (0, 0, 0, 0, 0, l))
        n += l

    q, k, v = map(lambda x: rearrange(x, "b (m c) h d -> b m c h d", c=c), [q, k, v])
    if log_f is not None:
        log_f = rearrange(log_f, "b (m c) h d -> b m c h d", c=c)

    if gradient_type == 4:
        g = v
    else:
        v_ = torch.einsum("b m c h d, b h d e -> b m c h e", k, state)
        if gradient_type == 0:
            g = v - v_
        elif gradient_type == 1:
            g = v - k - v_
        elif gradient_type == 2:
            g = ln_fused_l2_bwd(v_, v, scale)
        elif gradient_type == 3:
            g = ln_fused_l2_bwd(v_, v - k, scale)

    # intra
    q, k, g = map(lambda x: rearrange(x, "b m c h d -> (b m) c h d", m=m), [q, k, g])
    o_intra = flash_attn_func(q, k, g, causal=True)
    o_intra = rearrange(o_intra, "(b m) c h d -> b (m c) h d", m=m)

    # inter
    states = torch.einsum("b n h d, b n h e -> b h d e", k, g)
    q = rearrange(q, "(b m) c h d -> b m c h d", m=m)
    states = rearrange(states, "(b m) h d e -> b m h d e", m=m)
    for i in range(m):
        # b c h d
        state_ = states[
            :,
            i,
        ]
        if log_f is not None:
            # b h d
            log_fi = torch.mean(log_f[:, i, :, :], dim=1).unsqueeze(-1)
            state = torch.exp(log_fi) * state + state_
        else:
            state = state + state_
        states[
            :,
            i,
        ] = state
    o_inter = torch.einsum("b m c h d, b m h d e -> b m c h e", q, states)
    o_inter = rearrange(o_inter, "b m c h d -> b (m c) h d", m=m)

    o = (o_intra + o_inter)[
        :,
        :n,
    ]
    state = states[
        :,
        -1,
    ]

    return o, state
