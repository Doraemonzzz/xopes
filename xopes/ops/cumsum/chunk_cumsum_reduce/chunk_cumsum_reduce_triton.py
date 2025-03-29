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
def _chunk_cumsum_reduce(
    X,  # B N
    O,  # B N
    B: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    REVERSE: tl.constexpr,  # if True, o[i] = x[n] + x[n-1] + ... + x[n - i + 1]
):
    off_b = tl.program_id(0)

    offset_b = off_b * N

    # compute offset
    if REVERSE:
        offset_n = (NUM_BLOCKS - 1) * BLOCK_N
        offset_n_sum = (NUM_BLOCKS - 1) * BLOCK_N
        stride = -1
    else:
        offset_n = 0
        offset_n_sum = BLOCK_N - 1
        stride = 1

    array = tl.arange(0, BLOCK_N)
    s = tl.zeros([1], dtype=tl.float32)

    for i in range(NUM_BLOCKS):
        array_n = offset_n + array
        mask = array_n < N
        mask_ = offset_n_sum < N
        x = tl.load(X + offset_b + array_n, mask=mask, other=0.0).to(tl.float32)
        x_ = tl.load(X + offset_b + offset_n_sum, mask=mask_, other=0.0).to(tl.float32)
        x += s
        s += x_

        tl.store(O + offset_b + array_n, x.to(O.dtype.element_ty), mask=mask)

        offset_n += stride * BLOCK_N
        offset_n_sum += stride * BLOCK_N


@contiguous
def chunk_cumsum_reduce_triton(
    x: torch.Tensor,
    dim: int = -1,
    reverse: bool = False,
    chunk_size: int = 128,
    **kwargs,
) -> torch.Tensor:
    """
    Convert chunked cumulative sums into a complete cumulative sum result.

    This function takes a tensor that has already been processed by chunk_cumsum
    (where local cumulative sums within each chunk have been calculated), and combines
    these local cumulative sums into a complete cumulative sum result.

    Args:
        x: Input tensor that has already been processed by chunk_cumsum (with local cumulative sums within chunks)
        dim: The dimension along which to compute the cumulative sum
        reverse: If True, compute the cumulative sum in reverse order
        chunk_size: The size of the chunks used for the cumulative sum

    Returns:
        The complete cumulative sum of the input tensor
    """
    if dim != -1:
        x = x.transpose(dim, -1)

    shape = x.shape

    # reshape input data into 2D tensor
    x = x.reshape(-1, x.shape[-1]).contiguous()
    b, n = x.shape
    # allocate output
    o = torch.empty_like(x)

    grid = (b,)
    _chunk_cumsum_reduce[grid](
        X=x,
        O=o,
        B=b,
        N=n,
        BLOCK_N=chunk_size,
        NUM_BLOCKS=triton.cdiv(n, chunk_size),
        REVERSE=reverse,
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
    o = chunk_cumsum_reduce_triton(x, dim=-1)
    print(o.shape)
