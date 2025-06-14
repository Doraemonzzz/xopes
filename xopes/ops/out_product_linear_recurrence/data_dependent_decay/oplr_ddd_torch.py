from typing import Optional

import torch


def oplr_ddd_torch(
    xk: torch.Tensor,
    xv: torch.Tensor,
    log_decay: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Applies Out Product Linear Recurrence with data-dependent decay.

    Args:
        xv: Input tensor of shape (B, N, E)
        xk: Expansion vector of shape (B, N, D)
        log_decay: Data-dependent decay of shape (B, N, D)

    Returns:
        Output tensor of shape (B, N, D, E)
    """
    dtype = xk.dtype
    xk = xk.float()
    xv = xv.float()
    if log_decay is None:
        assert torch.all(xk <= 1), "xk must be all negative when decay is None"
        log_decay = torch.log(1 - xk)
    log_decay = log_decay.float()

    b, n, d = xk.shape
    e = xv.shape[-1]

    state = torch.zeros(b, d, e, dtype=torch.float32, device=xv.device)
    o = torch.zeros(b, n, d, e, dtype=torch.float32, device=xv.device)
    for i in range(n):
        xv_i = xv[:, i]
        xk_i = xk[:, i]
        log_decay_i = log_decay[:, i]
        decay = torch.exp(log_decay_i).unsqueeze(-1)
        state_i = torch.einsum("b d, b e -> b d e", xk_i, xv_i)
        state = decay * state + state_i
        o[:, i] = state

    return o.to(dtype)


if __name__ == "__main__":
    import torch.nn.functional as F

    b, n, d, e = 2, 512, 128, 128
    dtype = torch.bfloat16
    xv = torch.randn((b, n, e), dtype=dtype).cuda()
    xk = torch.randn((b, n, d), dtype=dtype).cuda()
    log_decay = F.logsigmoid(torch.randn((b, n, d), dtype=dtype).cuda())
    o = oplr_ddd_torch(xk, xv, log_decay)
    print(o.shape)
