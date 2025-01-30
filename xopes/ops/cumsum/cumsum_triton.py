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
    BLOCK_N: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    REVERSE: tl.constexpr,  # if True, o[i] = x[n] + x[n-1] + ... + x[n - i + 1]
):
    off_b = tl.program_id(0)

    # compute offset
    offset = off_b * N

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
    def forward(ctx, x, dim=-1, reverse=False):
        if dim != -1:
            x = x.transpose(dim, -1)

        # reshape input data into 2D tensor
        x_ = x.reshape(-1, x.shape[-1]).contiguous()
        b, n = x_.shape

        # allocate output
        o = torch.empty_like(x_)

        # Less than 64KB per feature
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(n))
        if n > BLOCK_N:
            raise RuntimeError("CumSum doesn't support sequence length >= 64KB.")

        grid = (b,)
        _cumsum[grid](
            X=x_,
            O=o,
            BLOCK_N=BLOCK_N,
            B=b,
            N=n,
            REVERSE=reverse,
        )

        ctx.dim = dim
        ctx.reverse = reverse

        o = o.reshape_as(x)

        if dim != -1:
            o = o.transpose(dim, -1)

        return o.contiguous()

    @staticmethod
    @contiguous
    def backward(ctx, do):
        dim = ctx.dim
        reverse = ctx.reverse
        if dim != -1:
            do = do.transpose(dim, -1)

        # reshape tensors
        do_ = do.reshape(-1, do.shape[-1]).contiguous()
        b, n = do_.shape

        # allocate gradient tensor
        dx = torch.empty_like(do_)

        # Less than 64KB per feature
        MAX_FUSED_SIZE = 65536 // do.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(n))
        if n > BLOCK_N:
            raise RuntimeError("CumSum doesn't support sequence length >= 64KB.")

        grid = (b,)
        _cumsum[grid](
            X=do_,
            O=dx,
            BLOCK_N=BLOCK_N,
            B=b,
            N=n,
            REVERSE=not reverse,
        )

        dx = dx.reshape_as(do)

        if dim != -1:
            dx = dx.transpose(dim, -1)

        return dx.contiguous(), None, None


def cumsum_triton(
    x: torch.Tensor, dim: int = -1, reverse: bool = False
) -> torch.Tensor:
    """
    Applies cumulative sum using Triton.

    Args:
        x: Input tensor
        dim: Dimension to apply cumsum over
        reverse: If True, compute reverse cumsum

    Returns:
        Cumulative sum of input tensor along specified dimension
    """
    return CumSumTriton.apply(x, dim, reverse)


if __name__ == "__main__":
    # Test code
    b, n = 2, 512
    dtype = torch.float32
    x = torch.randn((b, n), dtype=dtype).cuda()
    o = cumsum_triton(x, dim=-1)
    print(o.shape)
