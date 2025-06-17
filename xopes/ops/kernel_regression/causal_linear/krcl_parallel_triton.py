# krcl: kernel regression with causal linear
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton

from xopes.ops.cumsum import chunk_cumsum_decay_fn
from xopes.ops.kernel_regression.causal_linear.utils import _krcl_parallel_inverse
from xopes.utils import contiguous


@contiguous
def krcl_parallel_inverse(
    q: torch.Tensor,
    k: torch.Tensor,
    ld: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    BLOCK_N: int = 128,
):
    b, n, h, d = k.shape
    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_q = q is not None
    use_alpha = alpha is not None
    use_beta = beta is not None

    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)
    inv = torch.empty(
        (b, h, NUM_BLOCK_N, BLOCK_N, BLOCK_N), device=k.device, dtype=k.dtype
    )

    def grid(meta):
        return (b, h, NUM_BLOCK_N)

    ld_cumsum = chunk_cumsum_decay_fn(ld, reverse=reverse, chunk_size=BLOCK_N)

    _krcl_parallel_inverse[grid](
        Q=q,
        K=k,
        INV=inv,
        LOG_DECAY=ld_cumsum,
        ALPHA=alpha,
        BETA=beta,
        USE_Q=use_q,
        USE_ALPHA=use_alpha,
        USE_BETA=use_beta,
        B=b,
        N=n,
        H=h,
        D=d,
        CU_SEQLENS=cu_seqlens,
        USE_CU_SEQLENS=use_cu_seqlens,
        REVERSE=reverse,
        BLOCK_N=BLOCK_N,
        NUM_BLOCK_N=NUM_BLOCK_N,
    )

    return inv


########## Fwd start ##########
@contiguous
def krcl_parallel_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    # todo: tril and triu
    pass


########## Bwd start ##########
def krcl_recurrence_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    ld: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    dfinal_state: torch.Tensor,
    cu_seqlens: torch.LongTensor,
):
    pass


class KrclParallelFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        ld,
        alpha,
        beta,
        initial_state=None,
        cu_seqlens=None,
    ):
        # Forward computation
        o, final_state = krcl_parallel_fwd(
            q=q,
            k=k,
            v=v,
            ld=ld,
            alpha=alpha,
            beta=beta,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

        # Save tensors needed for backward
        ctx.save_for_backward(
            q, k, v, o, ld, alpha, beta, initial_state, final_state, cu_seqlens
        )

        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dfinal_state):
        (
            q,
            k,
            v,
            o,
            ld,
            alpha,
            beta,
            initial_state,
            final_state,
            cu_seqlens,
        ) = ctx.saved_tensors

        dq, dk, dv, dld, dalpha, dbeta, dinitial_state = krcl_parallel_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            do=do,
            ld=ld,
            alpha=alpha,
            beta=beta,
            initial_state=initial_state,
            final_state=final_state,
            dfinal_state=dfinal_state,
            cu_seqlens=cu_seqlens,
        )

        return (dq, dk, dv, dld, dalpha, dbeta, dinitial_state, None)


def krcl_parallel_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Kernel Regression with Causal Linear in Pytorch.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (B, N, H)
        alpha: Alpha tensor of shape (B, N, H)
        beta: Beta tensor of shape (B, N, H)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        o: Tensor of shape (B, N, H, E)
        state: Tensor of shape (B, H, D, E)
    """
    return KrclParallelFunction.apply(
        q, k, v, ld, alpha, beta, initial_state, cu_seqlens
    )


if __name__ == "__main__":
    b, n, h, d = 2, 16, 12, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    q = torch.randn(b, n, h, d, device=device, dtype=dtype)
    k = torch.randn(b, n, h, d, device=device, dtype=dtype)
    v = torch.randn(b, n, h, d, device=device, dtype=dtype)
    ld = F.logsigmoid(torch.randn(b, n, h, device=device))
    o, state = krcl_parallel_triton(q, k, v, ld)
