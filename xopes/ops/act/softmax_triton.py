import torch
import triton
import triton.language as tl

from xopes.utils import contiguous


@triton.jit
def _softmax_fwd_triton(
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
    x = tl.load(x_block_ptr, mask=d_mask, other=-float("inf")).to(tl.float32)
    # for stable
    x_minus_max = x - tl.max(x, axis=0)
    # softmax
    numerator = tl.exp(x_minus_max)
    denominator = tl.sum(x_minus_max)
    o = numerator / denominator

    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=d_mask)


@triton.jit
def _softmax_bwd_triton(
    O,
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
    o_block_ptr = O + offset_n + tl.arange(0, BLOCK)
    do_block_ptr = DO + offset_n + tl.arange(0, BLOCK)
    dx_block_ptr = DX + offset_n + tl.arange(0, BLOCK)
    o = tl.load(o_block_ptr, mask=d_mask, other=0).to(tl.float32)
    do = tl.load(do_block_ptr, mask=d_mask, other=0).to(tl.float32)
    # scalar
    c = tl.sum(o * do, axis=0)
    dx = o * do - c * o

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=d_mask)


class SoftmaxTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, dim=-1):
        if dim != -1:
            x = x.transpose(dim, -1).contiguous()

        o = softmax_fwd_triton(x)

        # save first
        ctx.save_for_backward(o)
        ctx.dim = dim

        if dim != -1:
            o = o.transpose(dim, -1).contiguous()

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        o = ctx.saved_tensors[0]
        dim = ctx.dim
        if dim != -1:
            do = do.transpose(dim, -1).contiguous()

        dx = softmax_bwd_triton(o, do)
        if dim != -1:
            dx = dx.transpose(dim, -1).contiguous()

        return dx, None


def softmax_fwd_triton(x):
    shape = x.shape
    n = torch.prod(torch.tensor(shape[:-1])).item()
    d = x.shape[-1]
    BLOCK = triton.next_power_of_2(d)
    o = torch.empty_like(x)

    grid = (n,)
    _softmax_fwd_triton[grid](
        x,
        o,
        n,
        d,
        BLOCK,
    )

    return o


def softmax_bwd_triton(o, do):
    shape = o.shape
    n = torch.prod(torch.tensor(shape[:-1])).item()
    d = o.shape[-1]
    BLOCK = triton.next_power_of_2(d)
    dx = torch.empty_like(o)

    grid = (n,)
    _softmax_bwd_triton[grid](o, do, dx, n, d, BLOCK)

    return dx


def softmax_triton(x, dim=-1):
    return SoftmaxTriton.apply(x, dim)


if __name__ == "__main__":
    # unit test
    dtype = torch.bfloat16
    device = torch.cuda.current_device()

    b, n, d = 8, 128, 64
    x = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((b, n, d), dtype=dtype, device=device)
    dim = -1

    o = softmax_triton(x, dim)
    o.backward(do)
