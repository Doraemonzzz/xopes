import torch
import triton
import triton.language as tl

from xopes.utils import contiguous


@triton.jit
def _lrpe_cosine_fwd(
    X,
    Theta,
    O,
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
    theta = tl.load(theta_block_ptr).to(tl.float32) * off_n
    o_cos = x * tl.cos(theta)
    o_sin = x * tl.sin(theta)

    tl.store(o_cos_block_ptr, o_cos.to(o_cos_block_ptr.dtype.element_ty))
    tl.store(o_sin_block_ptr, o_sin.to(o_cos_block_ptr.dtype.element_ty))


@triton.jit
def _lrpe_cosine_bwd(
    X,
    Theta,
    DO,
    DX,
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

    theta = tl.load(theta_block_ptr).to(tl.float32) * off_n
    dx = do_cos * tl.cos(theta) + do_sin * tl.sin(theta)

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty))


class LrpeCosine(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, theta):
        b, h, n, d = x.shape
        o = torch.empty(b, h, n, 2 * d, dtype=x.dtype, device=x.device)

        grid = (b, h, n)
        _lrpe_cosine_fwd[grid](x, theta, o, b, h, n, d)

        ctx.save_for_backward(x, theta)

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, theta = ctx.saved_tensors
        b, h, n, d = x.shape

        dx = torch.empty_like(x)

        grid = (b, h, n)
        _lrpe_cosine_bwd[grid](x, theta, do, dx, b, h, n, d)

        return dx, None


def lrpe_cosine_triton(x, theta):
    # x: b, h, n, d
    # theta: h, d
    return LrpeCosine.apply(x, theta)


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
