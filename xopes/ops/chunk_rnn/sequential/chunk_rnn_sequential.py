from typing import Optional

import torch

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None

from ..utils import ln_fused_l2_bwd


def chunk_rnn_sequential(
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
    Applies sequential chunk RNN in Pytorch.

    Args:
        q: Query tensor, shape (B, N, H, D)
        k: Key tensor, shape (B, N, H, D)
        v: Value tensor, shape (B, N, H, D)
        log_f: Log frequency tensor, shape (B, N, H, D)
        initial_state: Initial state tensor, shape (B, N, H, D)
        scale: Scale tensor, shape (H, D)
        gradient_type:
            0: v - k * st
            1: v - k - k * st
            2: v - normalize(k * st)
            3: v - k - normalize(k * st)
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

    for i in range(m):
        start = i * c
        end = min(start + c, n)
        qi = q[:, start:end, :, :]
        ki = k[:, start:end, :, :]
        vi = v[:, start:end, :, :]

        if log_f is not None:
            log_fi = torch.mean(log_f[:, start:end, :, :], dim=1)
        else:
            log_fi = None

        vi_ = torch.einsum("b c h d, b h d e -> b c h e", ki, state)
        if gradient_type == 0:
            gi = vi - vi_
        elif gradient_type == 1:
            gi = vi - ki - vi_
        elif gradient_type == 2:
            gi = ln_fused_l2_bwd(vi_, vi, scale)
        elif gradient_type == 3:
            gi = ln_fused_l2_bwd(vi_, vi - ki, scale)

        oi_intra = flash_attn_func(qi, ki, gi, causal=True)
        oi_inter = torch.einsum("b c h d, b h d e -> b c h e", ki, state)
        oi = oi_intra + oi_inter
        if log_f is not None:
            ki = ki * torch.exp(log_f[:, start:end, :, :])
        state_ = torch.einsum("b n h d, b n h e -> b h d e", ki, gi)
        o.append(oi)
        torch.exp(log_fi)

        if log_f is not None:
            state = state + state_
        else:
            state = torch.exp(log_fi).unsqueeze(-1) * state + state_

    o = torch.cat(o, dim=1)

    return o, state
