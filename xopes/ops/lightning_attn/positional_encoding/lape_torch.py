# lape: lightning attention positional encoding
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import repeat

from xopes.ops.lightning_attn.constant_decay import lacd_torch


def lape_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Lightning Attention with Scalar Decay in Pytorch.

    Args:
        q: Query tensor of shape (H, D)
        k: Key tensor of shape (H, D)
        v: Value tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (H,)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        output: Tensor of shape (B, N, H, E)
        state: Tensor of shape (B, H, D, E)
    """
    b, n, h, e = v.shape
    q.shape[-1]
    q = repeat(q, "h d -> b n h d", b=b, n=n)
    k = repeat(k, "h d -> b n h d", b=b, n=n)

    return lacd_torch(
        q=q,
        k=k,
        v=v,
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )


if __name__ == "__main__":
    b, n, h, d = 2, 16, 12, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    q = torch.randn(h, d, device=device, dtype=dtype)
    k = torch.randn(h, d, device=device, dtype=dtype)
    v = torch.randn(b, n, h, d, device=device, dtype=dtype)
    ld = F.logsigmoid(torch.randn(h, device=device))
    output, state = lape_torch(q, k, v, ld)
