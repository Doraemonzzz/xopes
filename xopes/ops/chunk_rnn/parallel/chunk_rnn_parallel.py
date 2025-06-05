from typing import Optional

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None

from xopes.ops.cumsum import cumsum_fn


def chunk_rnn_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    f: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    decay_weight: Optional[torch.Tensor] = None,
    decay_type: str = "pos",
    decay_fn: str = "mean",
    initial_state: Optional[torch.Tensor] = None,
    chunk_size: int = 128,
    attention_mask: Optional[torch.Tensor] = None,
    start: int = 0,
):
    """
    Applies parallel chunk RNN in Pytorch.

    Args:
        q: Query tensor, shape (B, N, H, D)
        k: Key tensor, shape (B, N, H, D)
        v: Value tensor, shape (B, N, H, E)
        f: Decay tensor, shape (B, N, H, 1) or (B, N, H, D)
        initial_state: Initial state tensor, shape (B, H, D, E) or (H, D, E)
        chunk_size: Chunk size
        attention_mask: Attention mask tensor, shape (B, N)
        start: Start index

    Returns:
        Output tensor, shape (B, N, H, E)
        State tensor, shape (B, N, H, E)
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
        l = end - start
        scale = l**-0.5
        qi = q[:, start:end]
        ki = k[:, start:end]
        vi = v[:, start:end]
        fi = f[:, start:end]

        if decay_fn == "mean":
            log_fi = F.logsigmoid(fi)
            log_fi_cumsum = cumsum_fn(log_fi, dim=1)
            # b c h f -> b h f -> b h f 1
            decay = torch.exp(log_fi_cumsum[:, -1]).unsqueeze(-1)
            qi = qi * torch.exp(log_fi_cumsum)
            ki = ki * torch.exp(log_fi_cumsum[:, -1:] - log_fi_cumsum)
        else:
            fi = torch.einsum("b c h d, c -> b h d", fi, decay_weight[:l]).unsqueeze(-1)
            decay = F.sigmoid(fi)

        if decay_type == "neg":
            decay = 2 * decay - 1

        oi_inter = torch.einsum("b c h d, b h d e -> b c h e", qi, state)
        o_inter.append(oi_inter)

        score = torch.einsum("b c h d, b c h e -> b h d e", ki, vi) * scale
        trans_score = F.softmax(score, dim=-1)
        state = state - torch.einsum("b h d e, b h e f -> b h d f", trans_score, state)
        state = decay * state + trans_score

    o_intra = flash_attn_func(q, k, v, causal=True, window_size=(chunk_size, 0))
    o_inter = torch.cat(o_inter, dim=1)

    o = o_intra + o_inter

    return o, state
