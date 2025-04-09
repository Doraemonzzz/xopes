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
        }
    ),
    key=[
        "B",
        "N",
        "H",
        "REVERSE",
    ],
)
@triton.jit
def _chunk_cumsum_decay(
    X,  # B N H
    O,  # B N H
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    REVERSE: tl.constexpr,  # if True, o[i] = x[n] + x[n-1] + ... + x[n - i + 1]
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_c = tl.program_id(2)

    # compute offset
    offset_b = off_b * N * H
    offset_h = off_h * BLOCK_H + tl.arange(0, BLOCK_H)
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

    mask_h = offset_h < H
    mask_x_block = mask_x[:, None] & mask_h[None, :]
    mask_o_block = mask_o[:, None] & mask_h[None, :]

    # compute block ptr
    x_block_ptr = X + offset_b + array_x[:, None] * H + offset_h[None, :]
    o_block_ptr = O + offset_b + array_o[:, None] * H + offset_h[None, :]

    # load
    x = tl.load(x_block_ptr, mask=mask_x_block, other=0.0).to(tl.float32)

    # compute cumsum
    o = tl.cumsum(x)

    # store
    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask_o_block)


@contiguous
def chunk_cumsum_decay_triton(
    x: torch.Tensor,
    reverse: bool = False,
    chunk_size: int = 128,
    **kwargs,
) -> torch.Tensor:
    """
    Applies chunk cumulative sum for log decay in linear attentionusing Triton.

    Args:
        x: Input tensor of shape (B, N, ...), operate on N dimension
        reverse: If True, compute reverse cumsum
        chunk_size: The size of the chunks to use for the cumulative sum.

    Returns:
        Cumulative sum of input tensor along specified dimension
    """
    b, n = x.shape[:2]
    h = prod(x.shape, start_dim=2)
    m = (n + chunk_size - 1) // chunk_size
    BLOCK_C = triton.next_power_of_2(chunk_size)

    # allocate output
    o = torch.empty_like(x)

    def grid(meta):
        NUM_BLOCK_H = triton.cdiv(h, meta["BLOCK_H"])
        return (b, NUM_BLOCK_H, m)

    _chunk_cumsum_decay[grid](
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
    o = chunk_cumsum_decay_triton(x, reverse=True, chunk_size=128)
    print(o.shape)
