# iav: implicit attention with v
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import repeat


def ilav_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    o: torch.Tensor,
    ld: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Implicit Linear Attention with V in Pytorch.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        o: Output tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (B, N, H)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        output: Tensor of shape (B, N, H, E)
        state: Tensor of shape (B, H, D, E)
    """
    b, n, h, d = q.shape
    e = o.shape[-1]
    dtype = q.dtype
    q = q.float()
    k = k.float()
    o = o.float()
    ld = ld.float()

    if cu_seqlens is None:
        if initial_state is None:
            state = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
        else:
            state = initial_state
            if len(state.shape) == 3:
                state = repeat(state, "h d e -> b h d e", b=b)

        v = []
        for i in range(n):
            qi = q[:, i]
            ki = k[:, i]
            oi = o[:, i]
            # update state
            ldi = ld[:, i]
            ratio = torch.exp(ldi)
            state = ratio.unsqueeze(-1).unsqueeze(-1) * state

            vi = oi - torch.einsum("b h d, b h d e -> b h e", qi, state)
            state_ = (1 - ratio.unsqueeze(-1).unsqueeze(-1)) * torch.einsum(
                "b h d, b h e -> b h d e", ki, vi
            )
            state = state + state_

            v.append(vi.unsqueeze(1))
        v = torch.cat(v, dim=1)
    else:
        assert b == 1, "cu_seqlens is only supported for batch size 1"
        q = q.squeeze(0)
        k = k.squeeze(0)
        o = o.squeeze(0)
        if ld is not None:
            ld = ld.squeeze(0)
        b = cu_seqlens.shape[0] - 1

        if initial_state is None:
            state = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
        else:
            state = initial_state
            if state.shape[0] == 1:
                state = state.squeeze(0)
            if len(state.shape) == 3:
                state = repeat(state, "h d e -> b h d e", b=b)

        v = []
        state_array = []
        for i in range(b):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            m = end - start
            q_ = q[start:end]
            k_ = k[start:end]
            o_ = o[start:end]
            ld_ = ld[start:end]
            state_ = state[i]
            v_array = []
            for j in range(m):
                qi = q_[j]
                ki = k_[j]
                oi = o_[j]

                # update state
                ldi = ld_[j]
                ratio = torch.exp(ldi)
                state_ = ratio.unsqueeze(-1).unsqueeze(-1) * state_

                vi = oi - torch.einsum("b h d, b h d e -> b h e", qi, state_)
                v_array.append(vi.unsqueeze(0))

                state__ = (1 - ratio.unsqueeze(-1).unsqueeze(-1)) * torch.einsum(
                    "h d, h e -> h d e", ki, vi
                )
                state_ = state_ + state__

            v.append(torch.cat(v_array, dim=0))
            state_array.append(state_.unsqueeze(0))
        v = torch.cat(v, dim=0).unsqueeze(0)
        state = torch.cat(state_array, dim=0)

    return v.to(dtype), state.to(dtype)


if __name__ == "__main__":
    b, n, h, d = 2, 16, 12, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    q = torch.randn(b, n, h, d, device=device, dtype=dtype)
    k = torch.randn(b, n, h, d, device=device, dtype=dtype)
    o = torch.randn(b, n, h, d, device=device, dtype=dtype)
    ld = F.logsigmoid(torch.randn(b, n, h, device=device))
    output, state = ilav_torch(q, k, o, ld)
