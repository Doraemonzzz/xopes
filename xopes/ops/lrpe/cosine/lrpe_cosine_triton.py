import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8]}),
    key=["h", "n", "d"],
)
@triton.jit
def _lrpe_cosine_fwd_triton(
    X,
    Theta,
    O,
    offset: tl.constexpr,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_n = tl.program_id(2)
    # compute offset
    offset_x = off_b * h * n * d + off_h * n * d + off_n * d
    offset_theta = off_h * d
    offset_o = off_b * h * n * 2 * d + off_h * n * 2 * d + off_n * 2 * d

    x_block_ptr = X + offset_x + tl.arange(0, d)
    theta_block_ptr = Theta + offset_theta + tl.arange(0, d)
    o_cos_block_ptr = O + offset_o + tl.arange(0, d)
    o_sin_block_ptr = O + offset_o + d + tl.arange(0, d)

    x = tl.load(x_block_ptr).to(tl.float32)
    theta = tl.load(theta_block_ptr).to(tl.float32) * (off_n + offset)
    o_cos = x * tl.cos(theta)
    o_sin = x * tl.sin(theta)

    tl.store(o_cos_block_ptr, o_cos.to(o_cos_block_ptr.dtype.element_ty))
    tl.store(o_sin_block_ptr, o_sin.to(o_cos_block_ptr.dtype.element_ty))


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8]}),
    key=["h", "n", "d"],
)
@triton.jit
def _lrpe_cosine_bwd_triton(
    X,
    Theta,
    DO,
    DX,
    offset: tl.constexpr,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_n = tl.program_id(2)
    # compute offset
    offset_x = off_b * h * n * d + off_h * n * d + off_n * d
    offset_theta = off_h * d
    offset_o = off_b * h * n * 2 * d + off_h * n * 2 * d + off_n * 2 * d

    theta_block_ptr = Theta + offset_theta + tl.arange(0, d)
    dx_block_ptr = DX + offset_x + tl.arange(0, d)
    do_cos_block_ptr = DO + offset_o + tl.arange(0, d)
    do_sin_block_ptr = DO + offset_o + d + tl.arange(0, d)

    do_cos = tl.load(do_cos_block_ptr).to(tl.float32)
    do_sin = tl.load(do_sin_block_ptr).to(tl.float32)

    theta = tl.load(theta_block_ptr).to(tl.float32) * (off_n + offset)
    dx = do_cos * tl.cos(theta) + do_sin * tl.sin(theta)

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty))


class LrpeCosineTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, theta, offset=0):
        o = lrpe_cosine_fwd_triton(x, theta, offset)
        ctx.save_for_backward(x, theta)
        ctx.offset = offset

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, theta = ctx.saved_tensors
        offset = ctx.offset
        dx = lrpe_cosine_bwd_triton(x, theta, do, offset)

        return dx, None, None


def lrpe_cosine_fwd_triton(x, theta, offset=0):
    b, h, n, d = x.shape
    o = torch.empty(b, h, n, 2 * d, dtype=x.dtype, device=x.device)

    def grid(meta):
        return (b, h, n)

    _lrpe_cosine_fwd_triton[grid](x, theta, o, offset, b, h, n, d)

    return o


def lrpe_cosine_bwd_triton(x, theta, do, offset=0):
    b, h, n, d = x.shape

    dx = torch.empty_like(x)

    def grid(meta):
        return (b, h, n)

    _lrpe_cosine_bwd_triton[grid](x, theta, do, dx, offset, b, h, n, d)

    return dx


def lrpe_cosine_triton(x, theta, offset=0):
    # x: b, h, n, d
    # theta: h, d
    return LrpeCosineTriton.apply(x, theta, offset)


if __name__ == "__main__":
    # unit test
    b, h, n, d = 2, 8, 128, 64
    dtype = torch.float32
    device = torch.cuda.current_device()
    x = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    theta = torch.randn((h, d), dtype=dtype, device=device)
    do = torch.randn((b, h, n, 2 * d), dtype=dtype, device=device)

    o = lrpe_cosine_triton(x, theta)
    o.backward(do)
