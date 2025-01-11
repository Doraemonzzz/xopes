from typing import Optional

import torch


def oplr_data_dependent_decay_torch(
    xk: torch.Tensor,  # b n d
    xv: torch.Tensor,  # b n e
    log_decay: Optional[torch.Tensor],  # b n d
) -> torch.Tensor:
    """
    Applies Out Product Linear Recurrence with data-dependent decay.

    Args:
        xv: Input tensor
        xk: Expansion vector

    Returns:
        Output tensor
    """
    if log_decay is None:
        assert torch.all(xk <= 1), "xk must be all negative when decay is None"
        log_decay = torch.log(1 - xk.float())

    b, n, d = xk.shape
    e = xv.shape[-1]

    state = torch.zeros(b, d, e, dtype=torch.float32, device=xv.device)
    o = torch.zeros(b, n, d, e, dtype=torch.float32, device=xv.device)
    for i in range(n):
        xv_i = xv[:, i]
        xk_i = xk[:, i]
        log_decay_i = log_decay[:, i]
        decay = torch.exp(log_decay_i.float()).unsqueeze(-1)
        state_i = torch.einsum("b d, b e -> b d e", xk_i, xv_i)
        state = decay * state + state_i
        o[:, i] = state

    return o.contiguous()


if __name__ == "__main__":
    import torch.nn.functional as F

    b, n, d, e = 2, 512, 128, 128
    dtype = torch.bfloat16
    xv = torch.randn((b, n, e), dtype=dtype).cuda()
    xk = torch.randn((b, n, d), dtype=dtype).cuda()
    log_decay = F.logsigmoid(torch.randn((b, n, d), dtype=dtype).cuda())
    o = oplr_data_dependent_decay_torch(xk, xv, log_decay)
    print(o.shape)
