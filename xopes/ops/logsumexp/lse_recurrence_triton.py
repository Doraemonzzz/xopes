import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
            "BLOCK_N": [128, 256, 512, 1024],
        }
    ),
    key=["N"],  # N represents the dimension we're reducing over
)
@triton.jit
def _lse_fwd(
    X,  # ... N
    O,  # ... 1
    B: tl.constexpr,  # batch size
    N: tl.constexpr,  # size of reduction dimension
    BLOCK_N: tl.constexpr,  # block size
):
    off_b = tl.program_id(0)
    # compute offset
    offset_x = off_b * N
    offset_o = off_b
    # mask
    array_n = tl.arange(0, BLOCK_N)
    # compute block ptr
    x_block_ptr = X + offset_x + array_n
    o_block_ptr = O + offset_o + tl.arange(0, 1)
    m = tl.full([1], -float("inf"), dtype=tl.float32)
    sse = tl.full([1], 0, dtype=tl.float32)

    for i in range(tl.cdiv(N, BLOCK_N)):
        mask_n = array_n < N
        x = tl.load(x_block_ptr, mask=mask_n, other=-float("inf"))
        mi = tl.max(x)
        m_ = tl.maximum(m, mi)
        sse = tl.exp(m - m_) * sse + tl.sum(tl.exp(x - m_))
        m = m_
        x_block_ptr += BLOCK_N
        array_n += BLOCK_N

    o = tl.log(sse) + m
    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty))


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
            "BLOCK_N": [128, 256, 512, 1024],
        }
    ),
    key=["N", "BLOCK_N"],
)
@triton.jit
def _lse_bwd(
    X,  # ... N
    O,  # ... 1
    DX,  # ... N
    DO,  # ... N
    B: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    # compute offset
    offset_b = off_b * n
    offset_n = off_n * BLOCK_N
    offset = offset_b + offset_n
    # mask
    array_n = offset_n + tl.arange(0, BLOCK_N)
    mask_n = array_n < N
    # compute block ptr
    x_block_ptr = X + offset + array_n
    o_block_ptr = O + offset + tl.arange(0, 1)
    dx_block_ptr = DX + offset + array_n
    do_block_ptr = DO + offset + array_n

    # load
    x = tl.load(x_block_ptr, mask=mask_n, other=0).to(tl.float32)
    o = tl.load(o_block_ptr, mask=mask_n, other=1).to(tl.float32)
    do = tl.load(do_block_ptr, mask=mask_n, other=0).to(tl.float32)

    p = tl.exp(x - o)
    dx = do * p

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=mask_n)


class LseRecurrenceTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x: torch.Tensor, dim: int, keepdim: bool = False):
        # Prepare output tensor and compute LSE

        if dim != -1:
            x = x.transpose(dim, -1).contiguous()

        o = torch.empty(list(x.shape[:-1]) + [1], dtype=x.dtype, device=x.device)

        shape = x.shape
        b = torch.prod(torch.tensor(shape[:-1])).item()
        n = shape[-1]

        grid = (b,)
        _lse_fwd[grid](X=x, O=o, B=n, N=n)

        ctx.save_for_backward(x, o)
        ctx.b = b
        ctx.n = n
        ctx.dim = dim
        ctx.keepdim = keepdim

        if dim != -1:
            o = o.transpose(dim, -1)

        if not keepdim:
            o = o.squeeze(dim)

        return o.contiguous()

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, o = ctx.saved_tensors
        dim = ctx.dim
        ctx.keepdim
        b = ctx.b
        n = ctx.n

        if dim != -1:
            do_ = do.transpose(dim, -1).contiguous()

        # ..., n
        dx = torch.empty_like(x)

        def grid(meta):
            return (b, triton.cdiv(n, meta["BLOCK_N"]))

        _lse_bwd[grid](
            DO=do_,
            X=x,
            O=o,
            DX=dx,
            B=b,
            N=n,
        )

        if dim != -1:
            dx = dx.transpose(dim, -1)

        dx = dx.reshape_as(do)

        return dx, None, None


def lse_recurrence_triton(
    x: torch.Tensor, dim: int, keepdim: bool = False
) -> torch.Tensor:
    """
    Compute log(sum(exp(x))) along a dimension using Triton.

    Args:
        x: Input tensor
        dim: Dimension to reduce over
        keepdim: Whether to keep the reduced dimension

    Returns:
        Result of log(sum(exp(x)))
    """
    return LseRecurrenceTriton.apply(x, dim, keepdim)


if __name__ == "__main__":
    # Test code
    x = torch.randn(2, 512, device="cuda")
    result = lse_recurrence_triton(x, dim=-1)
    print(result.shape)
