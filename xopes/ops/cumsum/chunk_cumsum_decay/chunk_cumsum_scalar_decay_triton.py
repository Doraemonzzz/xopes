import torch
import triton
import triton.language as tl

from xopes.utils import generate_configs


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8],
        }
    ),
    key=[
        "B",
        "N",
        "H" "REVERSE",
    ],
)
@triton.jit
def _chunk_cumsum_scalar_decay(
    X,  # B N H
    O,  # B N H
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    REVERSE: tl.constexpr,  # if True, o[i] = x[n] + x[n-1] + ... + x[n - i + 1]
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_c = tl.program_id(2)

    # compute offset
    offset_b = off_b * N * H
    offset_h = off_h
    offset_c = off_c * C

    # mask
    if REVERSE:
        array_o = (
            offset_c + BLOCK_C - 1 - tl.arange(0, BLOCK_C) - (BLOCK_C - C) % BLOCK_C
        )
        array_x = array_o + 1
        mask_o = (array_o >= offset_c) & (array_o < N)
        mask_x = (array_x >= offset_c) & (array_x < N)
    else:
        array_o = offset_c + tl.arange(0, BLOCK_C)
        array_x = array_o
        mask_o = (array_o >= offset_c) & (array_o < N)
        mask_x = mask_o

    # compute block ptr
    x_block_ptr = X + offset_b + offset_h + array_x * H
    o_block_ptr = O + offset_b + offset_h + array_o * H

    # load
    x = tl.load(x_block_ptr, mask=mask_x, other=0.0).to(tl.float32)

    # compute cumsum
    o = tl.cumsum(x)

    # store
    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask_o)


def chunk_cumsum_scalar_decay_triton(
    x: torch.Tensor,
    reverse: bool = False,
    chunk_size: int = 128,
    **kwargs,
) -> torch.Tensor:
    """
    Applies chunk cumulative sum for log decay in linear attentionusing Triton.

    Args:
        x: Input tensor of shape (B, N, H), operate on N dimension
        reverse: If True, compute reverse cumsum
        chunk_size: The size of the chunks to use for the cumulative sum.

    Returns:
        Cumulative sum of input tensor along specified dimension
    """
    b, n, h = x.shape
    m = (n + chunk_size - 1) // chunk_size
    BLOCK_C = triton.next_power_of_2(chunk_size)

    # allocate output
    o = torch.empty_like(x)

    grid = (b, h, m)
    _chunk_cumsum_scalar_decay[grid](
        X=x,
        O=o,
        B=b,
        N=n,
        H=h,
        C=chunk_size,
        BLOCK_C=BLOCK_C,
        REVERSE=reverse,
    )

    return o


if __name__ == "__main__":
    # Test code
    b, n, h = 2, 512, 128
    dtype = torch.float32
    x = torch.randn((b, n, h), dtype=dtype).cuda()
    o = chunk_cumsum_scalar_decay_triton(x, reverse=True, chunk_size=128)
    print(o.shape)
