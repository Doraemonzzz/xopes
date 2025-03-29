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
def _chunk_reverse_cumsum(
    X,  # B N
    O,  # B N
    B: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_c = tl.program_id(1)

    # compute offset
    offset_n = off_b * N
    offset_c = off_c * C

    # mask
    array_x = (
        offset_c + BLOCK_C - 1 - tl.arange(0, BLOCK_C) - (BLOCK_C - C) % BLOCK_C + 1
    )
    array_o = offset_c + BLOCK_C - 1 - tl.arange(0, BLOCK_C) - (BLOCK_C - C) % BLOCK_C
    mask_x = (array_x >= offset_c) & (array_x < N)
    mask_o = (array_o >= offset_c) & (array_o < N)

    # compute block ptr
    x_block_ptr = X + offset_n + array_x
    o_block_ptr = O + offset_n + array_o

    # load
    x = tl.load(x_block_ptr, mask=mask_x, other=0.0).to(tl.float32)

    # compute cumsum
    o = tl.cumsum(x)

    # store
    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask_o)


@contiguous
def chunk_reverse_cumsum_triton(
    x: torch.Tensor,
    dim: int = -1,
    chunk_size: int = 128,
    **kwargs,
) -> torch.Tensor:
    """
    Compute the chunk reverse cumulative sum of a tensor along a specified dimension.
    if the input is x1, ... , xn, we first pad zero to the last position and drop the first position-> x2, ... , xn, 0,
    then do reverse chunk cumsum, this function is used for linear attention grad computation.

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

    b, n = x.shape
    m = (n + chunk_size - 1) // chunk_size
    BLOCK_C = triton.next_power_of_2(chunk_size)

    # allocate output
    o = torch.empty_like(x)

    grid = (b, m)
    _chunk_reverse_cumsum[grid](
        X=x,
        O=o,
        B=b,
        N=n,
        C=chunk_size,
        BLOCK_C=BLOCK_C,
    )

    o = o.reshape(shape)

    if dim != -1:
        o = o.transpose(dim, -1)

    return o.contiguous()


if __name__ == "__main__":
    # Test code
    b, n = 2, 512
    dtype = torch.float32
    x = torch.randn((b, n), dtype=dtype).cuda()
    o = chunk_reverse_cumsum_triton(x, dim=-1)
    print(o.shape)
