from typing import Optional, Tuple

import torch


def laer_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Lightning Attention with Element-wise Recurrence.

    Args:
        q: Query tensor of shape (B, N, D)
        k: Key tensor of shape (B, N, D)
        v: Value tensor of shape (B, N, D)
        ld: Logarithmic decay tensor of shape (B, N, D)
        initial_state: Initial state tensor of shape (B, D)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        Output tensor of shape (B, N, D)
        State Tensor of shape (B, D)
    """
    dtype = q.dtype
    q = q.float()
    k = k.float()
    v = v.float()
    ld = ld.float()

    b, n, d = q.shape

    if initial_state is None:
        state = torch.zeros(b, d, dtype=torch.float32, device=q.device)
    else:
        state = initial_state

    o = torch.zeros(b, n, d, dtype=torch.float32, device=q.device)
    for i in range(n):
        qi = q[:, i]
        ki = k[:, i]
        vi = v[:, i]
        ldi = ld[:, i]
        decay = torch.exp(ldi)
        state_i = ki * vi
        state = decay * state + state_i
        o[:, i] = qi * state

    return o.to(dtype), state.to(dtype)


if __name__ == "__main__":
    import torch.nn.functional as F

    b, n, d = 2, 512, 128
    dtype = torch.bfloat16
    q = torch.randn((b, n, d), dtype=dtype).cuda()
    k = torch.randn((b, n, d), dtype=dtype).cuda()
    v = torch.randn((b, n, d), dtype=dtype).cuda()
    ld = F.logsigmoid(torch.randn((b, n, d), dtype=dtype).cuda())
    o = laer_torch(q, k, v, ld)
    print(o.shape)
