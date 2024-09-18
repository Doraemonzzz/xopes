import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs, next_power_of_two


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8]}),
    key=["h", "n", "d", "m"],
)
@triton.jit
def _md_lrpe_cosine_cache_fwd_triton(
    X,
    Theta,
    O,
    ThetaCache,
    Shape,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    l: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    m: tl.constexpr,
    BLOCK: tl.constexpr,
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
    x_block_ptr = X + offset_x + offset_d + tl.arange(0, BLOCK)
    theta_block_ptr = Theta + offset_theta + tl.arange(0, BLOCK)
    o_cos_block_ptr = O + offset_o + offset_d + tl.arange(0, BLOCK)
    o_sin_block_ptr = O + offset_o + offset_d + d + tl.arange(0, BLOCK)
    theta_cache_block_ptr = (
        ThetaCache + offset_theta_cache + offset_d + tl.arange(0, BLOCK)
    )
    # triton only support load block at least 16 elements, use this to get shape
    shape_block_ptr = Shape + m + tl.arange(0, 16)
    shape_mask = tl.arange(0, 16) < 1
    # mask
    e_mask = tl.arange(0, BLOCK) < e

    c = off_n - l
    offset = 0

    n_mask = c >= 0
    theta_ = tl.load(theta_block_ptr, mask=e_mask & n_mask[None], other=0).to(
        tl.float32
    )
    # this is equivalent to:
    # if off_n >= l:
    #     theta_ = tl.load(theta_block_ptr).to(tl.float32)
    # else:
    #     # concat((x, 0)) = concat(x * cos(0), x * sin(0))
    #     theta_ = tl.zeros((e,), dtype=tl.float32)

    for i in range(m):
        # update block ptr
        shape_block_ptr -= 1
        x_block_ptr -= e
        o_cos_block_ptr -= e
        o_sin_block_ptr -= e
        offset_d -= e
        theta_cache_block_ptr -= e
        mask = ((offset_d + tl.arange(0, BLOCK)) < d) & e_mask

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
def _md_lrpe_cosine_cache_bwd_triton(
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
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_n = tl.program_id(2)
    # compute offset
    offset_x = off_b * h * n * d + off_h * n * d + off_n * d
    offset_o = off_b * h * n * 2 * d + off_h * n * 2 * d + off_n * 2 * d
    offset_theta_cache = off_h * n * d + off_n * d

    # compute in parallel
    theta_cache_block_ptr = ThetaCache + offset_theta_cache + tl.arange(0, BLOCK)
    dx_block_ptr = DX + offset_x + tl.arange(0, BLOCK)
    do_cos_block_ptr = DO + offset_o + tl.arange(0, BLOCK)
    do_sin_block_ptr = DO + offset_o + d + tl.arange(0, BLOCK)
    # mask
    d_mask = tl.arange(0, BLOCK) < d

    # compute
    theta = tl.load(theta_cache_block_ptr, mask=d_mask, other=0).to(tl.float32)
    do_cos = tl.load(do_cos_block_ptr, mask=d_mask, other=0).to(tl.float32)
    do_sin = tl.load(do_sin_block_ptr, mask=d_mask, other=0).to(tl.float32)
    dx = do_cos * tl.cos(theta) + do_sin * tl.sin(theta)

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=d_mask)


class MdLrpeCosineCacheTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, theta, shape, l=0):
        o, theta_cache = md_lrpe_cosine_cache_fwd_triton(x, theta, shape, l)

        ctx.save_for_backward(x, theta, shape, theta_cache)
        ctx.l = l

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, theta, shape, theta_cache = ctx.saved_tensors
        l = ctx.l

        dx = md_lrpe_cosine_cache_bwd_triton(x, theta, theta_cache, do, shape, l)

        return dx, None, None, None


def md_lrpe_cosine_cache_fwd_triton(x, theta, shape, l=0):
    b, h, n, d = x.shape
    e = theta.shape[-1]
    m = len(shape)

    output_shape = list(x.shape)
    output_shape[-1] *= 2

    o = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    theta_cache = torch.empty((h, n, d), dtype=torch.float32, device=theta.device)
    BLOCK = next_power_of_two(e)

    def grid(meta):
        return (b, h, n)

    _md_lrpe_cosine_cache_fwd_triton[grid](
        x, theta, o, theta_cache, shape, b, h, n, l, d, e, m, BLOCK
    )

    return o, theta_cache


def md_lrpe_cosine_cache_bwd_triton(x, theta, theta_cache, do, shape, l=0):
    b, h, n, d = x.shape
    e = theta.shape[-1]
    m = len(shape)

    dx = torch.empty_like(x)
    BLOCK = next_power_of_two(d)

    def grid(meta):
        return (b, h, n)

    _md_lrpe_cosine_cache_bwd_triton[grid](
        x, theta, theta_cache, do, dx, shape, b, h, n, d, e, m, BLOCK
    )

    return dx


def md_lrpe_cosine_cache_triton(x, theta, shape, l=0):
    # x: b, h, n, d; n = l + prod(shape)
    # theta: h, e; e >= round(d + len(shape) - 1) // len(shape))
    # shape: n1, ... , nm
    # l: we do not do lrpe cosine on the first l tokens
    shape = torch.tensor(shape, dtype=torch.int32, device=x.device)
    assert (
        theta.shape[-1] * len(shape) >= x.shape[-1]
    ), "dim of theta should be larger than round(d + len(shape) - 1) // len(shape))"

    return MdLrpeCosineCacheTriton.apply(x, theta, shape, l)


if __name__ == "__main__":
    from einops import pack

    # unit test
    shape = tuple([2, 8, 8, 8, 8, 64])
    l = 2
    b = shape[0]
    h = shape[1]
    d = shape[-1]
    m = len(shape) - 3
    e = (d + m - 1) // m
    dtype = torch.float32
    device = torch.cuda.current_device()

    x = torch.randn(shape, dtype=dtype, device=device)
    x, ps_x = pack([x], "b h * d")
    if l > 0:
        token = torch.randn((b, h, l, d), dtype=dtype, device=device)
        x = torch.cat([token, x], dim=-2)
    x = x.requires_grad_()

    theta = torch.randn((h, e), dtype=dtype, device=device)
    shape = shape[:-1] + (shape[-1] * 2,)

    do = torch.randn(shape, dtype=dtype, device=device)
    do, ps_do = pack([do], "b h * d")
    if l > 0:
        do_token = torch.randn((b, h, l, 2 * d), dtype=dtype, device=device)
        do = torch.cat([do_token, do], dim=-2)

    o = md_lrpe_cosine_cache_triton(x, theta, shape=shape[2:-1])
    o.backward(do)
