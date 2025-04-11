from typing import Optional

import torch
import torch.nn.functional as F


def tpa_q_full_decode_torch(
    q: torch.Tensor,
    ak: torch.Tensor,
    av: torch.Tensor,
    bk: torch.Tensor,
    bv: torch.Tensor,
    scale: Optional[float] = None,
    scale_k: Optional[float] = None,
    scale_v: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Apply Flash Attention for Tensor Product Attention.

    Args:
        q: Query A tensor of shape (B, N, H, D)
        ak: Key A tensor of shape (B, M, H)
        av: Value A tensor of shape (B, M, H)
        bk: Key B tensor of shape (B, M, D)
        bv: Value B tensor of shape (B, M, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        Output tensor of shape (B, N, H, E)
    """
    b, n, h, d = q.shape
    assert n == 1, "n must be 1 when using tpa_decode_torch"

    if scale is None:
        scale = d**-0.5
    if scale_k is None:
        scale_k = 1
    if scale_v is None:
        scale_v = 1

    k = torch.einsum("b m h, b m d -> b m h d", ak, bk) * scale_k
    v = torch.einsum("b m h, b m e -> b m h e", av, bv) * scale_v

    score = torch.einsum("b n h d, b m h d -> b h n m", q, k) * scale
    prob = F.softmax(score, dim=-1)
    o = torch.einsum("b h n m, b m h e -> b n h e", prob, v)

    return o


if __name__ == "__main__":
    b, m, h, d, e = 2, 512, 32, 128, 64
    n = 1
    dtype = torch.bfloat16
    q = torch.randn((b, n, h, d), dtype=dtype).cuda()
    ak = torch.randn((b, m, h), dtype=dtype).cuda()
    av = torch.randn((b, m, h), dtype=dtype).cuda()
    bk = torch.randn((b, m, d), dtype=dtype).cuda()
    bv = torch.randn((b, m, e), dtype=dtype).cuda()
    o = tpa_q_full_decode_torch(q, ak, av, bk, bv)
    print(o.shape)
