from typing import Optional

import torch
import torch.nn.functional as F


def poly_attn_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: int = 4,
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
        scale: Scale of the polynomial
        causal: Whether to use causal attention
        mask: Mask tensor of shape (N, N)
    """
    dtype = q.dtype
    b, n, h, d = q.shape
    v.shape[-1]
    if scale == -1:
        scale = d**-0.5
    score = torch.einsum("b n h d, b m h d -> b h n m", q, k) * scale
    score = 1 + score / p
    if causal:
        if mask is None:
            mask = torch.tril(torch.ones(n, n).to(q))
        score = score.masked_fill(mask == 0, 0)

    score_max = torch.max(torch.abs(score), dim=-1, keepdim=True).values
    score_safe = (score / score_max) ** p
    score_sum = torch.sum(score_safe, dim=-1, keepdim=True)
    score = score_safe / score_sum

    return torch.einsum("b h n m, b m h e -> b n h e", score, v).to(dtype)


def poly_attn_log_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: int = 4,
    scale: float = -1,
    causal: bool = False,
    mask: Optional[torch.Tensor] = None,
    eps=1e-6,
    **kwargs,
):
    """
    Apply Polynomial Attention in Pytorch.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        p: Order of the polynomial
        scale: Scale of the polynomial
        causal: Whether to use causal attention
        mask: Mask tensor of shape (N, N)
    """
    dtype = q.dtype
    b, n, h, d = q.shape
    v.shape[-1]
    if scale == -1:
        scale = d**-0.5
    score = torch.einsum("b n h d, b m h d -> b h n m", q, k) * scale
    log_score = p * torch.log(torch.abs(1 + score / p))
    if causal:
        if mask is None:
            mask = torch.tril(torch.ones(n, n).to(q))
        log_score = log_score.masked_fill(mask == 0, float("-inf"))

    log_score_max = torch.max(log_score, dim=-1, keepdim=True).values
    log_score_safe = log_score - log_score_max
    score_safe = torch.exp(log_score_safe)
    score_sum = torch.sum(score_safe, dim=-1, keepdim=True) + eps
    score = score_safe / score_sum

    return torch.einsum("b h n m, b m h e -> b n h e", score, v).to(dtype)


def poly_attn_naive_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p: int = 4,
    poly_type: int = 1,
    scale: float = -1,
    causal: bool = False,
    mask: Optional[torch.Tensor] = None,
    eps=1e-6,
    **kwargs,
):
    """
    Apply Polynomial Attention in Pytorch.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        p: Order of the polynomial
        poly_type: Type of the polynomial
        scale: Scale of the polynomial
        causal: Whether to use causal attention
        mask: Mask tensor of shape (N, N)
    """
    dtype = q.dtype
    b, n, h, d = q.shape
    v.shape[-1]
    if scale == -1:
        scale = d**-0.5
    score = torch.einsum("b n h d, b m h d -> b h n m", q, k) * scale

    if poly_type == 1:
        score = torch.pow(1 + score / p, p)
    elif poly_type == 2:
        score = torch.pow(score, p)

    if causal:
        if mask is None:
            mask = torch.tril(torch.ones(n, n).to(q))
        score = score.masked_fill(mask == 0, 0)

    score_sum = torch.sum(score, dim=-1, keepdim=True) + eps
    score = score / score_sum

    return torch.einsum("b h n m, b m h e -> b n h e", score, v).to(dtype)


def softmax_attn_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
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
        scale: Scale of the polynomial
        causal: Whether to use causal attention
        mask: Mask tensor of shape (N, N)
    """
    dtype = q.dtype
    b, n, h, d = q.shape
    v.shape[-1]
    if scale == -1:
        scale = d**-0.5
    score = torch.einsum("b n h d, b m h d -> b h n m", q, k) * scale
    if causal:
        if mask is None:
            mask = torch.tril(torch.ones(n, n).to(q))
        score = score.masked_fill(mask == 0, float("-inf"))

    p = F.softmax(score, dim=-1)

    return torch.einsum("b h n m, b m h e -> b n h e", p, v).to(dtype)


if __name__ == "__main__":
    from xopes.utils.test_utils import get_abs_err, get_err_ratio

    b, n, h, d = 2, 16, 12, 16
    p = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    q = torch.randn(b, n, h, d, device=device, dtype=dtype)
    k = torch.randn(b, n, h, d, device=device, dtype=dtype)
    v = torch.randn(b, n, h, d, device=device, dtype=dtype)
    output = poly_attn_torch(q, k, v, p=p)
    output_log = poly_attn_log_torch(q, k, v, p=p)
    output_softmax = softmax_attn_torch(q, k, v)
    print(torch.norm(output - output_log))
    print(torch.norm(output_log - output_softmax))
    print("abs_err", get_abs_err(output_log, output_softmax))
    print("err_ratio", get_err_ratio(output_log, output_softmax))
