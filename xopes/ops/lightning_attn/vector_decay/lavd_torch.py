# lavd: lightning attention vector decay
from typing import Optional, Tuple

import torch


def lavd_torch(
    q: torch.Tensor,
    ldk: torch.Tensor,
    ldv: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Lightning Attention with Vector Decay in Pytorch.

    Args:
        q: Query tensor, (B, N, H, D)
        ldk: Log Decay vector for key, (B, N, H, D)
        ldv: Log Decay vector for value, (B, N, H, E)
        k: Key tensor, if not provided, 1 - exp(ldk) is used (B, N, H, D)
        v: Value tensor, if not provided, 1 - exp(ldv) is used (B, N, H, E)
        state: State tensor, (B, H, D, E)

    Returns:
        Output tensor, (B, N, H, E)
        State tensor, (B, H, D, E)
    """
    dtype = q.dtype
    q = q.float()
    ldk = ldk.float()
    ldv = ldv.float()
    if k is not None:
        k = k.float()
    if v is not None:
        v = v.float()

    b, n, h, d = q.shape
    e = v.shape[-1]

    o = torch.zeros((b, n, h, e), dtype=q.dtype, device=q.device)

    if state is None:
        state = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)

    for i in range(n):
        qi = q[:, i]
        dk_i = torch.exp(ldk[:, i])
        dv_i = torch.exp(ldv[:, i])
        if k is not None:
            ki = k[:, i]
        else:
            ki = 1 - dk_i

        if v is not None:
            vi = v[:, i]
        else:
            vi = 1 - dv_i

        state = ki.unsqueeze(-1) * state * vi.unsqueeze(-2) + torch.einsum(
            "b h d, b h e -> b h d e", ki, vi
        )
        oi = torch.einsum("b h d, b h d e -> b h e", qi, state)
        o[:, i] = oi

    return o.to(dtype), state.to(dtype)


if __name__ == "__main__":
    b, n, h, d = 2, 8, 12, 128
    e = 64
    dtype = torch.bfloat16
    q = torch.randn((b, n, h, d), dtype=dtype).cuda()
    ldk = torch.randn((b, n, h, d), dtype=dtype).cuda()
    ldv = torch.randn((b, n, h, e), dtype=dtype).cuda()
    k = torch.randn((b, n, h, d), dtype=dtype).cuda()
    v = torch.randn((b, n, h, e), dtype=dtype).cuda()
    o, state = lavd_torch(q, ldk, ldv, k, v)
    print(o.shape)
