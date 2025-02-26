# lasd: lightning attention scalar decay
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def lasd_parallel_torch(
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
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (H,)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        output: Tensor of shape (B, N, H, E)
        state: Tensor of shape (B, H, D, E)
    """
    b, n, h, d = q.shape
    v.shape[-1]
    dtype = q.dtype
    q = q.float()
    k = k.float()
    v = v.float()

    array = torch.arange(n, dtype=torch.int32, device=q.device)
    diff = array.unsqueeze(1) - array.unsqueeze(0)
    if ld is not None:
        decay = ld.unsqueeze(-1).unsqueeze(-1) * diff
        mask = torch.exp(torch.where(diff >= 0, decay, 0))
    else:
        mask = torch.where(diff >= 0, 1, 0)

    score = torch.einsum("b n h d, b m h d -> b h n m", q, k)
    score = score * mask
    o = torch.einsum("b h n m, b m h e -> b n h e", score, v)

    return o.to(dtype), None


if __name__ == "__main__":
    b, n, h, d = 2, 16, 12, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    q = torch.randn(b, n, h, d, device=device, dtype=dtype)
    k = torch.randn(b, n, h, d, device=device, dtype=dtype)
    v = torch.randn(b, n, h, d, device=device, dtype=dtype)
    ld = F.logsigmoid(torch.randn(h, device=device))
    output, state = lasd_torch(q, k, v, ld)
