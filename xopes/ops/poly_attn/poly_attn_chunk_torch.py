from typing import Optional

import torch
from einops import rearrange


def poly_attn_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: int = 4,
    chunk_size: int = 256,
    scale: float = -1,
    causal: bool = False,
    mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    Apply Polynomial Attention in Pytorch.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        p: Order of the polynomial
        chunk_size: Chunk size
        scale: Scale of the polynomial
        causal: Whether to use causal attention
        mask: Mask tensor of shape (N, N)
    """
    dtype = q.dtype
    b, n, h, d = q.shape
    m = v.shape[1]
    e = v.shape[-1]
    if scale == -1:
        scale = d**-0.5
    if causal:
        if mask is None:
            mask = torch.tril(torch.ones(chunk_size, chunk_size).to(q))

    n1 = (n + chunk_size - 1) // chunk_size
    m1 = (m + chunk_size - 1) // chunk_size
    o = []
    for i in range(n1):
        start_i = i * chunk_size
        end_i = min(start_i + chunk_size, n)
        ni = end_i - start_i
        qi = q[
            :,
            start_i:end_i,
        ]
        # max value
        mi = torch.full(
            (b, ni, h, 1), float("-inf"), device=q.device, dtype=torch.float32
        )
        # sum exp: sum exp(si - mi)
        li = torch.zeros((b, ni, h, 1), device=q.device, dtype=torch.float32)

        oi = torch.zeros((b, ni, h, e), device=q.device, dtype=torch.float32)
        for j in range(m1):
            if causal and j > i:
                continue

            start_j = j * chunk_size
            end_j = min(start_j + chunk_size, m)
            mj = end_j - start_j
            kj = k[:, start_j:end_j]
            vj = v[:, start_j:end_j]
            score = torch.einsum("b n h d, b m h d -> b h n m", qi, kj) * scale
            log_score = p * torch.log(torch.abs(1 + score / p))
            if causal:
                if j == i:
                    log_score = log_score.masked_fill(
                        mask[:ni, :mj] == 0, float("-inf")
                    )

            log_score_max = torch.max(log_score, dim=-1, keepdim=True).values
            log_score_safe = log_score - log_score_max
            score_safe = torch.exp(log_score_safe)
            score_sum = torch.sum(score_safe, dim=-1, keepdim=True)
            score = score_safe / score_sum
            oi_ = torch.einsum("b h n m, b m h e -> b n h e", score, vj)

            log_score_max = rearrange(log_score_max, "b h n m -> b n h m")
            score_sum = rearrange(score_sum, "b h n m -> b n h m")
            mi_ = torch.maximum(mi, log_score_max)
            li_ = torch.exp(mi - mi_) * li + torch.exp(log_score_max - mi_) * score_sum

            pi = li / li_ * torch.exp(mi - mi_)

            oi = pi * oi + (1 - pi) * oi_
            mi = mi_
            li = li_

        o.append(oi)

    o = torch.cat(o, dim=1).to(dtype)

    return o


if __name__ == "__main__":
    b, n, h, d = 2, 16, 12, 16
    p = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    q = torch.randn(b, n, h, d, device=device, dtype=dtype)
    k = torch.randn(b, n, h, d, device=device, dtype=dtype)
    v = torch.randn(b, n, h, d, device=device, dtype=dtype)
    output = poly_attn_chunk_torch(q, k, v, p)
    print(output.shape)
