import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs, next_power_of_two


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8]}),
    key=["h", "n", "d", "m"],
)
@triton.jit
def _md_lrpe_cosine_cache_fwd(
    X,
    Theta,
    O,
    ThetaCache,
    Shape,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    m: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_n = tl.program_id(2)
    # compute offset
    offset_x = off_b * h * n * d + off_h * n * d + off_n * d
    offset_theta = off_h * e
    offset_o = off_b * h * n * 2 * d + off_h * n * 2 * d + off_n * 2 * d
    offset_d = m * e
    offset_theta_cache = off_h * n * d + off_n * d

    # compute from the last theta block
    x_block_ptr = X + offset_x + offset_d + tl.arange(0, e)
    theta_block_ptr = Theta + offset_theta + tl.arange(0, e)
    o_cos_block_ptr = O + offset_o + offset_d + tl.arange(0, e)
    o_sin_block_ptr = O + offset_o + offset_d + d + tl.arange(0, e)
    theta_cache_block_ptr = ThetaCache + offset_theta_cache + offset_d + tl.arange(0, e)
    # triton only support load block at least 16 elements, use this to get shape
    shape_block_ptr = Shape + m + tl.arange(0, 16)
    shape_mask = tl.arange(0, 16) < 1

    c = off_n
    offset = 0
    theta_ = tl.load(theta_block_ptr).to(tl.float32)
    for i in range(m):
        # update block ptr
        shape_block_ptr -= 1
        x_block_ptr -= e
        o_cos_block_ptr -= e
        o_sin_block_ptr -= e
        offset_d -= e
        theta_cache_block_ptr -= e
        mask = (offset_d + tl.arange(0, e)) < d

        # compute dim
        dim = tl.sum(tl.load(shape_block_ptr, mask=shape_mask, other=0).to(tl.int32))
        offset = c % dim
        c = c // dim

        # compute
        x = tl.load(x_block_ptr, mask=mask, other=0).to(tl.float32)
        theta = theta_ * offset
        o_cos = x * tl.cos(theta)
        o_sin = x * tl.sin(theta)

        # save
        tl.store(o_cos_block_ptr, o_cos.to(o_cos_block_ptr.dtype.element_ty), mask=mask)
        tl.store(o_sin_block_ptr, o_sin.to(o_sin_block_ptr.dtype.element_ty), mask=mask)
        tl.store(
            theta_cache_block_ptr,
            theta.to(theta_cache_block_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8]}),
    key=["h", "n", "d", "m"],
)
@triton.jit
def _md_lrpe_cosine_cache_bwd(
    X,
    Theta,
    ThetaCache,
    DO,
    DX,
    Shape,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    m: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_n = tl.program_id(2)
    # compute offset
    offset_x = off_b * h * n * d + off_h * n * d + off_n * d
    off_h * e
    offset_o = off_b * h * n * 2 * d + off_h * n * 2 * d + off_n * 2 * d
    offset_d = m * e
    offset_theta_cache = off_h * n * d + off_n * d

    # compute from the last theta block
    theta_cache_block_ptr = ThetaCache + offset_theta_cache + offset_d + tl.arange(0, e)
    dx_block_ptr = DX + offset_x + offset_d + tl.arange(0, e)
    do_cos_block_ptr = DO + offset_o + offset_d + tl.arange(0, e)
    do_sin_block_ptr = DO + offset_o + offset_d + d + tl.arange(0, e)
    # triton only support load block at least 16 elements, use this to get shape
    shape_block_ptr = Shape + m + tl.arange(0, 16)
    tl.arange(0, 16) < 1

    for i in range(m):
        # update block ptr
        shape_block_ptr -= 1
        dx_block_ptr -= e
        do_cos_block_ptr -= e
        do_sin_block_ptr -= e
        offset_d -= e
        theta_cache_block_ptr -= e
        mask = (offset_d + tl.arange(0, e)) < d

        # compute
        theta = tl.load(theta_cache_block_ptr, mask=mask, other=0).to(tl.float32)
        do_cos = tl.load(do_cos_block_ptr, mask=mask, other=0).to(tl.float32)
        do_sin = tl.load(do_sin_block_ptr, mask=mask, other=0).to(tl.float32)
        dx = do_cos * tl.cos(theta) + do_sin * tl.sin(theta)

        tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=mask)


class MdLrpeCosineCache(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, theta, shape):
        b, h, d = x.shape[0], x.shape[1], x.shape[-1]
        e = theta.shape[-1]
        n = torch.prod(shape).item()
        m = len(shape)

        output_shape = list(x.shape)
        output_shape[-1] *= 2

        o = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        theta_cache = torch.empty((h, n, d), dtype=torch.float32, device=theta.device)

        def grid(meta):
            return (b, h, n)

        _md_lrpe_cosine_cache_fwd[grid](
            x, theta, o, theta_cache, shape, b, h, n, d, e, m
        )

        ctx.save_for_backward(x, theta, shape, theta_cache)

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, theta, shape, theta_cache = ctx.saved_tensors
        b, h, d = x.shape[0], x.shape[1], x.shape[-1]
        e = theta.shape[-1]
        n = torch.prod(shape).item()
        m = len(shape)

        dx = torch.empty_like(x)

        def grid(meta):
            return (b, h, n)

        _md_lrpe_cosine_cache_bwd[grid](
            x, theta, theta_cache, do, dx, shape, b, h, n, d, e, m
        )

        return dx, None, None


def md_lrpe_cosine_cache_triton(x, theta, shape=None):
    # x: b, h, ..., d
    # theta: h, next_power_of_two((d + len(shape) - 1) // len(shape))
    if shape is None:
        shape = x.shape[2:-1]
    assert theta.shape[-1] == next_power_of_two(
        theta.shape[-1]
    ), "theta.shape[-1] must be power of 2"
    shape = torch.tensor(shape, dtype=torch.int32, device=x.device)
    return MdLrpeCosineCache.apply(x, theta, shape)


if __name__ == "__main__":
    # unit test
    shape = tuple([2, 8, 128, 128, 64])
    h = shape[1]
    d = shape[-1]
    m = len(shape) - 3
    e = next_power_of_two((d + m - 1) // m)
    dtype = torch.float32
    device = torch.cuda.current_device()
    x = (torch.randn(shape, dtype=dtype, device=device)).requires_grad_()
    theta = torch.randn((h, e), dtype=dtype, device=device)
    shape = shape[:-1] + (shape[-1] * 2,)
    do = torch.randn(shape, dtype=dtype, device=device)

    o = md_lrpe_cosine_cache_triton(x, theta)
    o.backward(do)
