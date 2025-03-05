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
def _chunk_cumsum(
    X,  # B N
    O,  # B N
    B: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    REVERSE: tl.constexpr,  # if True, o[i] = x[n] + x[n-1] + ... + x[n - i + 1]
):
    off_b = tl.program_id(0)
    off_c = tl.program_id(1)

    # compute offset
    offset_n = off_b * N
    offset_c = off_c * C

    # mask
    if REVERSE:
        array = offset_c + BLOCK_C - 1 - tl.arange(0, BLOCK_C) - (BLOCK_C - C) % BLOCK_C
        mask = (array >= offset_c) & (array < N)
    else:
        array = offset_c + tl.arange(0, BLOCK_C)
        mask = array < N

    # compute block ptr
    x_block_ptr = X + offset_n + array
    o_block_ptr = O + offset_n + array

    # load
    x = tl.load(x_block_ptr, mask=mask, other=0.0).to(tl.float32)

    # compute cumsum
    o = tl.cumsum(x)

    # store
    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask)


class ChunkCumSumTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, dim=-1, reverse=False, chunk_size=128):
        b, n = x.shape
        m = (n + chunk_size - 1) // chunk_size
        BLOCK_C = triton.next_power_of_2(chunk_size)

        # allocate output
        o = torch.empty_like(x)

        grid = (b, m)
        _chunk_cumsum[grid](
            X=x,
            O=o,
            B=b,
            N=n,
            C=chunk_size,
            BLOCK_C=BLOCK_C,
            REVERSE=reverse,
        )

        ctx.dim = dim
        ctx.reverse = reverse
        ctx.chunk_size = chunk_size

        return o.contiguous()

    @staticmethod
    @contiguous
    def backward(ctx, do):
        ctx.dim
        reverse = ctx.reverse
        chunk_size = ctx.chunk_size

        b, n = do.shape
        m = (n + chunk_size - 1) // chunk_size
        BLOCK_C = triton.next_power_of_2(chunk_size)

        # allocate gradient tensor
        dx = torch.empty_like(do)

        grid = (b, m)
        _chunk_cumsum[grid](
            X=do,
            O=dx,
            B=b,
            N=n,
            C=chunk_size,
            BLOCK_C=BLOCK_C,
            REVERSE=not reverse,
        )

        return dx.contiguous(), None, None, None


def chunk_cumsum_triton(
    x: torch.Tensor,
    dim: int = -1,
    reverse: bool = False,
    chunk_size: int = 128,
) -> torch.Tensor:
    """
    Applies cumulative sum using Triton.

    Args:
        x: Input tensor
        dim: Dimension to apply cumsum over
        reverse: If True, compute reverse cumsum
        chunk_size: The size of the chunks to use for the cumulative sum.

    Returns:
        Cumulative sum of input tensor along specified dimension
    """
    if dim != -1:
        x = x.transpose(dim, -1)

    shape = x.shape

    # reshape input data into 2D tensor
    x = x.reshape(-1, x.shape[-1]).contiguous()
    o = ChunkCumSumTriton.apply(x, -1, reverse, chunk_size)
    o = o.reshape(shape)

    if dim != -1:
        o = o.transpose(dim, -1)

    return o.contiguous()


if __name__ == "__main__":
    # Test code
    b, n = 2, 512
    dtype = torch.float32
    x = torch.randn((b, n), dtype=dtype).cuda()
    o = cumsum_triton(x, dim=-1)
    print(o.shape)
