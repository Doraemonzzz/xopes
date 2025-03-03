from typing import Optional

import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


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
    X,  # B N
    O,  # B N
    CU_SEQLENS,  # M
    BLOCK_N: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    REVERSE: tl.constexpr,  # if True, o[i] = x[n] + x[n-1] + ... + x[n - i + 1]
    USE_CU_SEQLENS: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_m = tl.program_id(1)

    # compute offset
    if not USE_CU_SEQLENS:
        offset = off_b * N
    else:
        start = tl.load(CU_SEQLENS + off_m)
        end = tl.load(CU_SEQLENS + off_m + 1)
        offset = off_b * N + start
        N = end - start

    # mask
    if REVERSE:
        array = BLOCK_N - 1 - tl.arange(0, BLOCK_N) - (BLOCK_N - N) % BLOCK_N
        mask = array >= 0
    else:
        array = tl.arange(0, BLOCK_N)
        mask = array < N

    # compute block ptr
    x_block_ptr = X + offset + array
    o_block_ptr = O + offset + array

    # load
    x = tl.load(x_block_ptr, mask=mask, other=0.0).to(tl.float32)

    # compute cumsum
    o = tl.cumsum(x)

    # store
    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask)


class CumSumTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, dim=-1, reverse=False, cu_seqlens=None):
        b, n = x.shape

        use_cu_seqlens = cu_seqlens is not None
        if use_cu_seqlens:
            m = cu_seqlens.shape[0] - 1
        else:
            m = 1

        # allocate output
        o = torch.empty_like(x)

        # Less than 64KB per feature
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(n))
        if use_cu_seqlens:
            max_seq_len = triton.next_power_of_2(
                torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).item()
            )
            BLOCK_N = min(BLOCK_N, max_seq_len)

        grid = (b, m)
        _cumsum[grid](
            X=x,
            O=o,
            CU_SEQLENS=cu_seqlens,
            BLOCK_N=BLOCK_N,
            B=b,
            N=n,
            REVERSE=reverse,
            USE_CU_SEQLENS=use_cu_seqlens,
        )

        ctx.dim = dim
        ctx.reverse = reverse
        ctx.save_for_backward(cu_seqlens)

        return o.contiguous()

    @staticmethod
    @contiguous
    def backward(ctx, do):
        ctx.dim
        reverse = ctx.reverse
        cu_seqlens = ctx.saved_tensors[0]

        # reshape tensors
        b, n = do.shape

        use_cu_seqlens = cu_seqlens is not None
        if use_cu_seqlens:
            m = cu_seqlens.shape[0] - 1
        else:
            m = 1

        # allocate gradient tensor
        dx = torch.empty_like(do)

        # Less than 64KB per feature
        MAX_FUSED_SIZE = 65536 // do.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(n))
        if use_cu_seqlens:
            max_seq_len = triton.next_power_of_2(
                torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).item()
            )
            BLOCK_N = min(BLOCK_N, max_seq_len)

        grid = (b, m)
        _cumsum[grid](
            X=do,
            O=dx,
            CU_SEQLENS=cu_seqlens,
            BLOCK_N=BLOCK_N,
            B=b,
            N=n,
            REVERSE=not reverse,
            USE_CU_SEQLENS=use_cu_seqlens,
        )

        return dx.contiguous(), None, None, None


def cumsum_triton(
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
    if dim != -1:
        x = x.transpose(dim, -1)

    shape = x.shape

    # reshape input data into 2D tensor
    x = x.reshape(-1, x.shape[-1]).contiguous()
    o = CumSumTriton.apply(x, -1, reverse, cu_seqlens)
    o = o.reshape(shape)

    if dim != -1:
        o = o.transpose(dim, -1)

    return o


if __name__ == "__main__":
    # Test code
    b, n = 2, 512
    dtype = torch.float32
    x = torch.randn((b, n), dtype=dtype).cuda()
    o = cumsum_triton(x, dim=-1)
    print(o.shape)
