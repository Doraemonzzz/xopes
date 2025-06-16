# ladd: linear attention with delta decay
from typing import Optional, Tuple

import torch
from einops import repeat


def ladd_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    normalize: bool = True,
    rms_norm: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Linear Attention with Delta Decay in Pytorch.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Output tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (B, N, H) or (B, N, H, d)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training
        normalize: Whether to normalize the key
        rms_norm: Whether to use RMS normalization for q and k

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
    ld = ld.float()
    beta = beta.float()
    d**0.5 if rms_norm else 1

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
            beta_i = beta[:, i]
            ldi = ld[:, i]
            if ld.ndim == 3:
                ldi = ldi.unsqueeze(-1)
            # update state
            ratio = torch.exp(ldi)
            state = ratio.unsqueeze(-1) * state

            vi = vi - torch.einsum("b h d, b h d e -> b h e", ki, state)
            state_ = torch.einsum("b h d, b h e -> b h d e", ki, vi) * beta_i.unsqueeze(
                -1
            ).unsqueeze(-1)
            state = state + state_
            oi = torch.einsum("b h d, b h d e -> b h e", qi, state)

            o.append(oi.unsqueeze(1))
        o = torch.cat(o, dim=1)

    return o.to(dtype), state.to(dtype)
