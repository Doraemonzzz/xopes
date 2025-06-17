from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from xopes.ops.cumsum import cumsum_fn
from xopes.utils import contiguous


########## pytorch implementation reference ##########
@contiguous
def krcl_inverse_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    ld: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    BLOCK_N: int = 128,
):
    k.dtype
    b, n, h, d = k.shape
    if q is None:
        q = k.clone()
    q = q.float()
    k = k.float()
    ld = ld.float()
    if alpha is not None:
        alpha = alpha.float()
        q = q * alpha.unsqueeze(-1)
    if beta is not None:
        beta = beta.float()
        k = k * beta.unsqueeze(-1)

    if n % BLOCK_N != 0:
        pad_n = BLOCK_N - n % BLOCK_N
        q = F.pad(q, (0, 0, 0, 0, 0, pad_n))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_n))
        ld = F.pad(ld, (0, 0, 0, pad_n))
        n = n + pad_n

    ld_cumsum = cumsum_fn(ld, dim=1, reverse=reverse)
    if reverse:
        mask = torch.tril(torch.ones(n, n, device=k.device, dtype=torch.bool))
    else:
        mask = torch.triu(torch.ones(n, n, device=k.device, dtype=torch.bool))

    inv_list = []
    l = (n + BLOCK_N - 1) // BLOCK_N
    for i in range(l):
        start = i * BLOCK_N
        end = min(start + BLOCK_N, n)
        m = end - start
        mask_ = mask[:m, :m]
        qi = q[:, start:end]
        ki = k[:, start:end]
        ldi = ld_cumsum[:, start:end]
        ldi = rearrange(ldi, "b n h -> b h n")

        diff = ldi.unsqueeze(-1) - ldi.unsqueeze(-2)
        diff = torch.where(mask_, 0, diff)
        score = torch.einsum("b n h d, b m h d -> b h n m", qi, ki)
        score = torch.exp(diff) * score
        score = torch.where(mask_, 0, score)

        if reverse:
            score = rearrange(score, "b h n m -> b h m n")

        # jacobian method
        b1 = torch.eye(m, device=k.device)
        inv = torch.eye(m, device=k.device)
        L = -score
        # C = (I + A) ^ -1, Ck = -A * Ck + I
        for i in range(m):
            inv = torch.einsum("...ij,...jk->...ik", L, inv) + b1

        if reverse:
            inv = rearrange(inv, "b h m n -> b h n m")

        inv_list.append(inv.unsqueeze(2))

    inv = torch.cat(inv_list, dim=2)

    return inv
