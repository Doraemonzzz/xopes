from typing import Optional, Tuple

import torch
from einops import repeat

from xopes.ops.lightning_attn.constant_decay import lacd_parallel_triton


def lape_parallel_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    save_states: bool = True,
    use_chunk_loop: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Lightning Attention Parallel with Scalar Decay in Triton.

    Args:
        q: Query tensor of shape (H, D)
        k: Key tensor of shape (H, D)
        v: Value tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (H,)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training
        save_states: Whether to save the states
        use_chunk_loop: Whether to use chunk loop

    Returns:
        output: Tensor of shape (B, N, H, E)
        state: Tensor of shape (B, H, D, E)
    """
    b, n, h, e = v.shape
    q.shape[-1]
    q = repeat(q, "h d -> b n h d", b=b, n=n)
    k = repeat(k, "h d -> b n h d", b=b, n=n)

    return lacd_parallel_triton(
        q=q,
        k=k,
        v=v,
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        save_states=save_states,
        use_chunk_loop=use_chunk_loop,
    )


if __name__ == "__main__":
    import torch.nn.functional as F

    b, n, h, d = 2, 16, 12, 64
    e = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    q = torch.randn(h, d, device=device, dtype=dtype).requires_grad_(True)
    k = torch.randn(h, d, device=device, dtype=dtype).requires_grad_(True)
    v = torch.randn(b, n, h, e, device=device, dtype=dtype).requires_grad_(True)
    ld = F.logsigmoid(torch.randn(h, device=device))
    initial_state = torch.randn(b, h, d, e, device=device, dtype=dtype).requires_grad_(
        True
    )
    output, final_state = lape_parallel_triton(q, k, v, ld, initial_state)
    loss = output.sum() + final_state.sum()
    loss.backward()
