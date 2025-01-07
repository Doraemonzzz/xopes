import torch
import triton
import triton.language as tl

from xopes.ops.logsumexp.lse_recurrence_triton import _lse_bwd
from xopes.utils import contiguous, generate_configs

MIN_BLOCK_N = 128


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
            "BLOCK_N": [MIN_BLOCK_N, MIN_BLOCK_N * 2, MIN_BLOCK_N * 4, MIN_BLOCK_N * 8],
        }
    ),
    key=["N"],  # N represents the dimension we're reducing over
)
@triton.jit
def _lse_parallel(
    X,  # ... N
    SSE,  # ... G
    MAX,  # ... G
    B: tl.constexpr,  # batch size
    N: tl.constexpr,  # size of reduction dimension
    G: tl.constexpr,  # number of groups
    BLOCK_N: tl.constexpr,  # block size
):
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    # compute offset
    offset_x = off_b * N + off_n * BLOCK_N
    offset_sm = off_b * G + off_n
    # mask
    array_n = tl.arange(0, BLOCK_N)
    # compute block ptr
    x_block_ptr = X + offset_x + array_n
    sse_block_ptr = SSE + offset_sm + tl.arange(0, 1)
    max_block_ptr = MAX + offset_sm + tl.arange(0, 1)

    # m = tl.full([1], -float("inf"), dtype=tl.float32)
    # sse = tl.full([1], 0, dtype=tl.float32)

    mask_n = (off_n * BLOCK_N + array_n) < N
    x = tl.load(x_block_ptr, mask=mask_n, other=-float("inf"))
    m = tl.max(x)
    sse = tl.sum(tl.exp(x - m))

    tl.store(sse_block_ptr, sse.to(sse_block_ptr.dtype.element_ty))
    tl.store(max_block_ptr, m.to(max_block_ptr.dtype.element_ty))


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
        }
    ),
    key=["N"],  # N represents the dimension we're reducing over
)
@triton.jit
def _lse_reduce(
    SSE,  # ... G
    MAX,  # ... G
    O,  # ... 1
    B: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
):
    off_b = tl.program_id(0)
    # compute offset
    offset_sm = off_b * G
    offset_o = off_b
    # compute block ptr
    sse_block_ptr = SSE + offset_sm + tl.arange(0, 1)
    max_block_ptr = MAX + offset_sm + tl.arange(0, 1)
    o_block_ptr = O + offset_o + tl.arange(0, 1)

    m = tl.full([1], -float("inf"), dtype=tl.float32)
    sse = tl.full([1], 0, dtype=tl.float32)
    for i in range(G):
        mi = tl.load(max_block_ptr)
        sse_i = tl.load(sse_block_ptr)
        m_ = tl.maximum(m, mi)
        sse = tl.exp(m - m_) * sse + tl.exp(mi - m_) * sse_i
        m = m_
        # update ptr
        max_block_ptr += 1
        sse_block_ptr += 1

    o = tl.log(sse) + m
    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty))


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
            "BLOCK_N": [128, 256, 512, 1024],
        }
    ),
    key=[
        "N",
    ],
)
@triton.jit
def _lse_bwd(
    X,  # ... N
    O,  # ... 1
    DX,  # ... N
    DO,  # ... 1
    B: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    # compute offset
    offset_b = off_b * N
    offset_n = off_n * BLOCK_N
    offset_x = offset_b + offset_n
    offset_o = off_b
    # mask
    array_n = offset_n + tl.arange(0, BLOCK_N)
    mask_n = array_n < N
    # compute block ptr
    x_block_ptr = X + offset_x + tl.arange(0, BLOCK_N)
    o_block_ptr = O + offset_o + tl.arange(0, 1)
    dx_block_ptr = DX + offset_x + tl.arange(0, BLOCK_N)
    do_block_ptr = DO + offset_o + tl.arange(0, 1)

    # load
    x = tl.load(x_block_ptr, mask=mask_n, other=-float("inf")).to(tl.float32)
    o = tl.load(o_block_ptr).to(tl.float32)
    do = tl.load(do_block_ptr)

    p = tl.exp(x - o)
    dx = do * p

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=mask_n)


class LseParallelTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x: torch.Tensor, dim: int, keepdim: bool = False):
        # update dim to be in the range of -x.ndim + 1 to 0
        if dim >= 0:
            dim = dim - x.ndim

        shape_origin = x.shape
        if dim != -1:
            x = x.transpose(dim, -1).contiguous()

        shape = x.shape
        b = int(torch.prod(torch.tensor(shape[:-1])).item())
        n = shape[-1]

        g = triton.cdiv(n, MIN_BLOCK_N)
        sse = torch.empty(list(x.shape[:-1]) + [g], dtype=x.dtype, device=x.device)
        m = torch.empty(list(x.shape[:-1]) + [g], dtype=x.dtype, device=x.device)
        o = torch.empty(list(x.shape[:-1]) + [1], dtype=x.dtype, device=x.device)

        def grid(meta):
            return (b, triton.cdiv(n, meta["BLOCK_N"]))

        _lse_parallel[grid](X=x, SSE=sse, MAX=m, B=b, N=n, G=g)

        grid = (b,)
        _lse_reduce[grid](SSE=sse, MAX=m, O=o, B=b, N=g, G=g)

        ctx.save_for_backward(x, o)
        ctx.b = b
        ctx.n = n
        ctx.shape_origin = shape_origin
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
        keepdim = ctx.keepdim
        b = ctx.b
        n = ctx.n
        shape_origin = ctx.shape_origin

        if not keepdim:
            do = do.unsqueeze(dim)

        if dim != -1:
            do = do.transpose(dim, -1).contiguous()

        # ..., n
        dx = torch.empty_like(x)

        def grid(meta):
            return (b, triton.cdiv(n, meta["BLOCK_N"]))

        _lse_bwd[grid](
            DO=do,
            X=x,
            O=o,
            DX=dx,
            B=b,
            N=n,
        )

        if dim != -1:
            dx = dx.transpose(dim, -1)

        dx = dx.reshape(shape_origin).contiguous()

        return dx, None, None


def lse_parallel_triton(
    x: torch.Tensor, dim: int, keepdim: bool = False
) -> torch.Tensor:
    """
    Compute log(sum(exp(x))) along a dimension using Triton, using parallel reduction.

    Args:
        x: Input tensor
        dim: Dimension to reduce over
        keepdim: Whether to keep the reduced dimension

    Returns:
        Result of log(sum(exp(x)))
    """
    return LseParallelTriton.apply(x, dim, keepdim)


if __name__ == "__main__":
    # Test code
    x = torch.randn(2, 512, device="cuda")
    result = lse_parallel_triton(x, dim=-1)
    print(result.shape)
