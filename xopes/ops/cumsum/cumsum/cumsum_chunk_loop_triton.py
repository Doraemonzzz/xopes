from typing import Optional

import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs, prod


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8],
            "BLOCK_H": [
                16,
                32,
                64,
            ],
            "BLOCK_N": [128],
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
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    REVERSE: tl.constexpr,  # if True, o[i] = x[n] + x[n-1] + ... + x[n - i + 1]
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)

    # compute offset
    offset_b = off_b * N * H
    offset_h = off_h * BLOCK_H

    # array and mask
    array_h = offset_h + tl.arange(0, BLOCK_H)
    mask_h = array_h < H

    if REVERSE:
        array_n = N - 1 - tl.arange(0, BLOCK_N)
        stride = -1
    else:
        array_n = tl.arange(0, BLOCK_N)
        stride = 1

    NUM_BLOCK_N = triton.cdiv(N, BLOCK_N)
    x_cumsum = tl.zeros([BLOCK_H], dtype=tl.float32)

    for i in range(NUM_BLOCK_N):
        mask_n = (array_n >= 0) & (array_n < N)
        mask = mask_n[:, None] & mask_h[None, :]

        # compute block ptr
        x_block_ptr = X + offset_b + array_n[:, None] * H + array_h[None, :]
        o_block_ptr = O + offset_b + array_n[:, None] * H + array_h[None, :]

        # load
        x = tl.load(x_block_ptr, mask=mask, other=0.0).to(tl.float32)

        # compute cumsum
        o = tl.cumsum(x, axis=0) + x_cumsum[None, :]
        x_cumsum += tl.sum(x, axis=0)

        # store
        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask)

        # update
        array_n += stride * BLOCK_N


class CumSumChunkLoopTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, dim=-1, reverse=False, cu_seqlens=None):
        if dim < 0:
            dim = len(x.shape) + dim
        b = prod(x.shape, start_dim=0, end_dim=dim - 1)
        n = x.shape[dim]
        h = prod(x.shape, start_dim=dim + 1)
        MAX_BLOCK_H = triton.next_power_of_2(h)

        # allocate output
        o = torch.empty_like(x)

        def grid(meta):
            NUM_BLOCK_H = triton.cdiv(h, meta["BLOCK_H"])
            return (b, NUM_BLOCK_H)

        _cumsum[grid](
            X=x,
            O=o,
            B=b,
            N=n,
            H=h,
            REVERSE=reverse,
        )

        ctx.b = b
        ctx.n = n
        ctx.h = h
        ctx.MAX_BLOCK_H = MAX_BLOCK_H
        ctx.reverse = reverse

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        b = ctx.b
        n = ctx.n
        h = ctx.h
        ctx.MAX_BLOCK_H
        reverse = ctx.reverse

        # allocate gradient tensor
        dx = torch.empty_like(do)

        def grid(meta):
            NUM_BLOCK_H = triton.cdiv(h, meta["BLOCK_H"])
            return (b, NUM_BLOCK_H)

        _cumsum[grid](
            X=do,
            O=dx,
            B=b,
            N=n,
            H=h,
            REVERSE=not reverse,
        )

        return dx.contiguous(), None, None, None


def cumsum_chunk_loop_triton(
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
    o = CumSumChunkLoopTriton.apply(x, dim, reverse, cu_seqlens)

    return o


if __name__ == "__main__":
    # Test code
    b, n = 2, 512
    dtype = torch.float32
    x = torch.randn((b, n), dtype=dtype).cuda()
    o = cumsum_chunk_loop_triton(x, dim=-1)
    print(o.shape)
