from typing import Optional

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None



def chunk_rnn_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_f: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
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
        chunk_size: Chunk size

    Returns:
        Output tensor, shape (B, N, H, D)
        State tensor, shape (B, N, H, D)
    """
    b, n, h, d = q.shape
    e = v.shape[-1]
    c = chunk_size
    if initial_state is None:
        state = torch.zeros((b, h, d, e), dtype=q.dtype, device=q.device)
    else:
        state = initial_state
    o_inter = []

    m = (n + c - 1) // c
    for i in range(m):
        start = i * chunk_size
        end = min(start + chunk_size, n)
        scale = (end - start) ** -0.5
        qi = q[:, start:end]
        ki = k[:, start:end]
        vi = v[:, start:end]
        log_fi = log_f[:, start:end]

        oi_inter = torch.einsum("b c h d, b h d e -> b c h e", qi, state)
        o_inter.append(oi_inter)
        # b c h -> b h -> b h 1 1
        decay = torch.exp(torch.sum(log_fi, dim=1)).unsqueeze(-1).unsqueeze(-1)

        score = torch.einsum("b c h d, b c h e -> b h d e", ki, vi) * scale
        trans_score = F.softmax(score, dim=-1)
        state = state - torch.einsum("b h d e, b h e f -> b h d f", trans_score, state)
        state = decay * state + trans_score

    o_intra = flash_attn_func(q, k, v, causal=True, window_size=(chunk_size, 0))
    o_inter = torch.cat(o_inter, dim=1)

    o = o_intra + o_inter

    return o, state
