import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs, transpose


@triton.autotune(
    generate_configs({"num_warps": [1, 2, 4, 8, 16, 32], "num_stages": [2, 4]}),
    key=["N", "D"],
)
@triton.jit
def _softmax_fwd_triton(
    X,
    O,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_n = tl.program_id(0)
    # compute offset
    offset_n = off_n * D
    # mask
    mask_d = tl.arange(0, BLOCK) < D

    # compute
    x_block_ptr = X + offset_n + tl.arange(0, BLOCK)
    o_block_ptr = O + offset_n + tl.arange(0, BLOCK)
    x = tl.load(x_block_ptr, mask=mask_d, other=-float("inf"))
    # for stable
    x_minus_max = x - tl.max(x, axis=0)
    # softmax
    numerator = tl.exp(x_minus_max)
    denominator = tl.sum(numerator)
    o = numerator / denominator

    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask_d)


@triton.autotune(
    generate_configs({"num_warps": [1, 2, 4, 8, 16, 32], "num_stages": [2, 4]}),
    key=["N", "D"],
)
@triton.jit
def _softmax_bwd_triton(
    O,
    DO,
    DX,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_n = tl.program_id(0)
    # compute offset
    offset_n = off_n * D
    # mask
    mask_d = tl.arange(0, BLOCK) < D

    # compute
    o_block_ptr = O + offset_n + tl.arange(0, BLOCK)
    do_block_ptr = DO + offset_n + tl.arange(0, BLOCK)
    dx_block_ptr = DX + offset_n + tl.arange(0, BLOCK)
    o = tl.load(o_block_ptr, mask=mask_d, other=0)
    do = tl.load(do_block_ptr, mask=mask_d, other=0)
    # scalar
    c = tl.sum(o * do, axis=0)
    dx = o * do - c * o

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=mask_d)


class SoftmaxTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, dim=-1):
        o = softmax_fwd_triton(
            x=x,
            dim=dim,
        )

        ctx.save_for_backward(o)
        ctx.dim = dim

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        o = ctx.saved_tensors[0]
        dim = ctx.dim

        dx = softmax_bwd_triton(
            o=o,
            do=do,
            dim=dim,
        )

        return dx, None


def softmax_fwd_triton(x, dim=-1):
    if dim != -1:
        x = transpose(x, dim, -1)

    shape = x.shape
    n = torch.prod(torch.tensor(shape[:-1])).item()
    d = x.shape[-1]
    BLOCK = triton.next_power_of_2(d)
    o = torch.empty_like(x)

    grid = (n,)
    _softmax_fwd_triton[grid](
        X=x,
        O=o,
        N=n,
        D=d,
        BLOCK=BLOCK,
    )

    if dim != -1:
        o = transpose(o, dim, -1)

    return o


def softmax_bwd_triton(o, do, dim=-1):
    if dim != -1:
        do = transpose(do, dim, -1)
        o = transpose(o, dim, -1)

    shape = o.shape
    n = torch.prod(torch.tensor(shape[:-1])).item()
    d = o.shape[-1]
    BLOCK = triton.next_power_of_2(d)
    dx = torch.empty_like(o)

    grid = (n,)
    _softmax_bwd_triton[grid](
        O=o,
        DO=do,
        DX=dx,
        N=n,
        D=d,
        BLOCK=BLOCK,
    )

    if dim != -1:
        dx = transpose(dx, dim, -1)
        o = transpose(o, dim, -1)

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
