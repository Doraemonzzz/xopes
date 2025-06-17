# krcl: kernel regression with causal linear
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import repeat


def krcl_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Kernel Regression with Causal Linear in Pytorch.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (B, N, H)
        alpha: Alpha tensor of shape (B, N, H)
        beta: Beta tensor of shape (B, N, H)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        o: Tensor of shape (B, N, H, E)
        state: Tensor of shape (B, H, D, E)
    """
    b, n, h, d = k.shape
    e = v.shape[-1]
    dtype = k.dtype
    if q is None:
        q = k
    if alpha is None:
        alpha = torch.ones(b, n, h, device=q.device, dtype=dtype)
    if beta is None:
        beta = torch.ones(b, n, h, device=q.device, dtype=dtype)
    q = q.float()
    k = k.float()
    v = v.float()
    ld = ld.float()
    alpha = alpha.float()
    beta = beta.float()

    q = q * alpha.unsqueeze(-1)
    k = k * beta.unsqueeze(-1)

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
        # update state
        ldi = ld[:, i]
        ratio = torch.exp(ldi)
        state = ratio.unsqueeze(-1).unsqueeze(-1) * state

        oi = vi - torch.einsum("b h d, b h d e -> b h e", qi, state)
        state_ = torch.einsum("b h d, b h e -> b h d e", ki, oi)
        state = state + state_

        o.append(oi.unsqueeze(1))
    o = torch.cat(o, dim=1)

    return o.to(dtype), state.to(dtype)


if __name__ == "__main__":
    b, n, h, d = 2, 16, 12, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    q = torch.randn(b, n, h, d, device=device, dtype=dtype)
    k = torch.randn(b, n, h, d, device=device, dtype=dtype)
    v = torch.randn(b, n, h, d, device=device, dtype=dtype)
    ld = F.logsigmoid(torch.randn(b, n, h, device=device))
    o, state = krcl_torch(q, k, v, ld)
