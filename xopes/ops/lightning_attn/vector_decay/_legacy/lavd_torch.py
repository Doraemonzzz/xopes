# lavd: lightning attention vector decay
from typing import Optional, Tuple

import torch


def lavd_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Lightning Attention with Vector Decay in Pytorch.

    Args:
        q: Query tensor, shape (B, N, H, D)
        k: Key tensor, shape (B, N, H, D)
        v: Value tensor, shape (B, N, H, E)
        ldk: Log Decay vector for key, shape (B, N, H, D), if not provided uses log(1 - exp(k))
        ldv: Log Decay vector for value, shape (B, N, H, E), if not provided uses log(1 - exp(v))
        use_ldk: Whether to use log decay for key
        use_ldv: Whether to use log decay for value
        initial_state: Initial state tensor, shape (B, H, D, E) or (H, D, E)

    Returns:
        Output tensor, shape (B, N, H, E)
        State tensor, shape (B, H, D, E)
    """
    if ldk is not None:
        use_ldk = True
    if ldv is not None:
        use_ldv = True
    assert use_ldk or use_ldv, "At least one of ldk or ldv must be used"

    dtype = q.dtype
    q = q.float()
    k = k.float()
    v = v.float()

    if use_ldk and ldk is None:
        ldk = torch.log(1 - k)
    ldk = ldk.float()

    if use_ldv and ldv is None:
        ldv = torch.log(1 - v)
    ldv = ldv.float()

    b, n, h, d = q.shape
    e = v.shape[-1]

    o = torch.zeros((b, n, h, e), dtype=q.dtype, device=q.device)

    if initial_state is None:
        state = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)
    else:
        state = initial_state

    for i in range(n):
        qi = q[:, i]
        ki = k[:, i]
        vi = v[:, i]
        if use_ldk:
            dk_i = torch.exp(ldk[:, i])
        if use_ldv:
            dv_i = torch.exp(ldv[:, i])

        state_ = torch.einsum("b h d, b h e -> b h d e", ki, vi)

        if use_ldk:
            state = dk_i.unsqueeze(-1) * state

        if use_ldv:
            state = state * dv_i.unsqueeze(-2)

        state = state + state_
        oi = torch.einsum("b h d, b h d e -> b h e", qi, state)
        o[:, i] = oi

    return o.to(dtype), state.to(dtype)


if __name__ == "__main__":
    b, n, h, d = 2, 8, 12, 128
    e = 64
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
    ldk = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    ldv = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
    initial_state = torch.randn(
        (b, h, d, e), dtype=dtype, device=device
    ).requires_grad_()

    o, state = lavd_torch(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        initial_state=initial_state,
    )
    (o.sum() + state.sum()).backward()
    print(o.shape)
