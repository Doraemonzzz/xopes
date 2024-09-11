import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs({"num_warps": [1, 2, 4, 8, 16, 32], "num_stages": [2, 4]}),
    key=["n", "d"],
)
@triton.jit
def _softmax_no_cache_fwd_triton(
    X,
    O,
    n: tl.constexpr,
    d: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_n = tl.program_id(0)
    # compute offset
    offset_n = off_n * d
    # mask
    d_mask = tl.arange(0, BLOCK) < d

    # compute
    x_block_ptr = X + offset_n + tl.arange(0, BLOCK)
    o_block_ptr = O + offset_n + tl.arange(0, BLOCK)
    x = tl.load(x_block_ptr, mask=d_mask, other=-float("inf"))
    # for stable
    x_minus_max = x - tl.max(x, axis=0)
    # softmax
    numerator = tl.exp(x_minus_max)
    denominator = tl.sum(numerator)
    o = numerator / denominator

    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=d_mask)


@triton.autotune(
    generate_configs({"num_warps": [1, 2, 4, 8, 16, 32], "num_stages": [2, 4]}),
    key=["n", "d"],
)
@triton.jit
def _softmax_no_cache_bwd_triton(
    X,
    DO,
    DX,
    n: tl.constexpr,
    d: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_n = tl.program_id(0)
    # compute offset
    offset_n = off_n * d
    # mask
    d_mask = tl.arange(0, BLOCK) < d

    # compute
    x_block_ptr = X + offset_n + tl.arange(0, BLOCK)
    do_block_ptr = DO + offset_n + tl.arange(0, BLOCK)
    dx_block_ptr = DX + offset_n + tl.arange(0, BLOCK)

    x = tl.load(x_block_ptr, mask=d_mask, other=-float("inf"))
    # for stable
    x_minus_max = x - tl.max(x, axis=0)
    # softmax
    numerator = tl.exp(x_minus_max)
    denominator = tl.sum(numerator)
    o = numerator / denominator

    do = tl.load(do_block_ptr, mask=d_mask, other=0)
    # scalar
    c = tl.sum(o * do, axis=0)
    dx = o * do - c * o

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=d_mask)


class SoftmaxNoCacheTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, dim=-1):
        o = softmax_no_cache_fwd_triton(x, dim)

        ctx.save_for_backward(x)
        ctx.dim = dim

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x = ctx.saved_tensors[0]
        dim = ctx.dim

        dx = softmax_no_cache_bwd_triton(x, do, dim)

        return dx, None


def softmax_no_cache_fwd_triton(x, dim=-1):
    if dim != -1:
        x = x.transpose(dim, -1).contiguous()

    shape = x.shape
    n = torch.prod(torch.tensor(shape[:-1])).item()
    d = x.shape[-1]
    BLOCK = triton.next_power_of_2(d)
    o = torch.empty_like(x)

    grid = (n,)
    _softmax_no_cache_fwd_triton[grid](
        x,
        o,
        n,
        d,
        BLOCK,
    )

    if dim != -1:
        o = o.transpose(dim, -1).contiguous()

    return o


def softmax_no_cache_bwd_triton(o, do, dim=-1):
    if dim != -1:
        do = do.transpose(dim, -1).contiguous()
        o = o.transpose(dim, -1).contiguous()

    shape = o.shape
    n = torch.prod(torch.tensor(shape[:-1])).item()
    d = o.shape[-1]
    BLOCK = triton.next_power_of_2(d)
    dx = torch.empty_like(o)

    grid = (n,)
    _softmax_no_cache_bwd_triton[grid](o, do, dx, n, d, BLOCK)

    if dim != -1:
        dx = dx.transpose(dim, -1).contiguous()
        o = o.transpose(dim, -1).contiguous()

    return dx


def softmax_no_cache_triton(x, dim=-1):
    return SoftmaxNoCacheTriton.apply(x, dim)


if __name__ == "__main__":
    # unit test
    dtype = torch.bfloat16
    device = torch.cuda.current_device()

    b, n, d = 8, 128, 64
    x = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((b, n, d), dtype=dtype, device=device)
    dim = -1

    o = softmax_no_cache_triton(x, dim)
    o.backward(do)
