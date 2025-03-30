from typing import Optional

import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs, prod


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8],
        }
    ),
    key=[
        "B",
        "N",
        "REVERSE",
    ],
)
@triton.jit
def _cumsum(
    X,  # B N H
    O,  # B N H
    CU_SEQLENS,  # M
    BLOCK_N: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    REVERSE: tl.constexpr,  # if True, o[i] = x[n] + x[n-1] + ... + x[n - i + 1]
):
    off_b = tl.program_id(0)
    tl.program_id(1)
    off_h = off_b % H
    off_b = off_b // H

    offset_b = off_b * N * H
    offset_h = off_h

    # mask
    if REVERSE:
        array = BLOCK_N - 1 - tl.arange(0, BLOCK_N) - (BLOCK_N - N) % BLOCK_N
        mask = array >= 0
    else:
        array = tl.arange(0, BLOCK_N)
        mask = array < N

    # compute block ptr
    x_block_ptr = X + offset_b + array * H + offset_h
    o_block_ptr = O + offset_b + array * H + offset_h

    # load
    x = tl.load(x_block_ptr, mask=mask, other=0.0).to(tl.float32)

    # compute cumsum
    o = tl.cumsum(x)

    # store
    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask)


class CumSumNoReshapeTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, dim=-1, reverse=False, cu_seqlens=None):
        if dim < 0:
            dim = len(x.shape) + dim
        b = prod(x.shape, start_dim=0, end_dim=dim - 1)
        n = x.shape[dim]
        h = prod(x.shape, start_dim=dim + 1)
        m = 1

        # allocate output
        o = torch.empty_like(x)

        # Less than 64KB per feature
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(n))

        grid = (b * h, m)
        _cumsum[grid](
            X=x,
            O=o,
            CU_SEQLENS=cu_seqlens,
            BLOCK_N=BLOCK_N,
            B=b,
            N=n,
            H=h,
            REVERSE=reverse,
        )

        ctx.b = b
        ctx.n = n
        ctx.h = h
        ctx.reverse = reverse
        ctx.save_for_backward(cu_seqlens)

        return o.contiguous()

    @staticmethod
    @contiguous
    def backward(ctx, do):
        b = ctx.b
        n = ctx.n
        h = ctx.h
        reverse = ctx.reverse
        cu_seqlens = ctx.saved_tensors[0]
        m = 1

        # allocate gradient tensor
        dx = torch.empty_like(do)

        # Less than 64KB per feature
        MAX_FUSED_SIZE = 65536 // do.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(n))

        grid = (b * h, m)
        _cumsum[grid](
            X=do,
            O=dx,
            CU_SEQLENS=cu_seqlens,
            BLOCK_N=BLOCK_N,
            B=b,
            N=n,
            H=h,
            REVERSE=not reverse,
        )

        return dx, None, None, None


def cumsum_no_reshape_triton(
    x: torch.Tensor,
    dim: int = -1,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> torch.Tensor:
    """
    Applies cumulative sum using Triton.

    Args:
        x: Input tensor
        dim: Dimension to apply cumsum over
        reverse: If True, compute reverse cumsum
        cu_seqlens: The cumulative sequence lengths of the input tensor.

    Returns:
        Cumulative sum of input tensor along specified dimension
    """
    o = CumSumNoReshapeTriton.apply(x, dim, reverse, cu_seqlens)

    return o


if __name__ == "__main__":
    # Test code
    b, n = 2, 512
    dtype = torch.float32
    x = torch.randn((b, n), dtype=dtype).cuda()
    o = cumsum_no_reshape_triton(x, dim=-1)
    print(o.shape)
