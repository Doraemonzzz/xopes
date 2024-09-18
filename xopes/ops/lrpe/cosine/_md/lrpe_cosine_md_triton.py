import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs, next_power_of_two


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8]}),
    key=["h", "n", "d", "m"],
)
@triton.jit
def _lrpe_cosine_md_fwd_triton(
    X,
    Theta,
    O,
    Shape,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    l: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    m: tl.constexpr,
    ACT: tl.constexpr,
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

    # compute from the last theta block
    x_block_ptr = X + offset_x + offset_d + tl.arange(0, BLOCK)
    theta_block_ptr = Theta + offset_theta + tl.arange(0, BLOCK)
    o_cos_block_ptr = O + offset_o + offset_d + tl.arange(0, BLOCK)
    o_sin_block_ptr = O + offset_o + offset_d + d + tl.arange(0, BLOCK)
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
    #     theta_ = tl.load(theta_block_ptr, mask=e_mask, other=0).to(tl.float32)
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
        mask = ((offset_d + tl.arange(0, BLOCK)) < d) & e_mask

        # compute dim
        dim = tl.sum(tl.load(shape_block_ptr, mask=shape_mask, other=0).to(tl.int32))
        offset = c % dim
        c = c // dim

        # compute
        if ACT != "none":
            if ACT == "softmax":
                value = -float("inf")
            else:
                value = 0

        x = tl.load(x_block_ptr, mask=mask, other=value).to(tl.float32)
        if ACT != "none":
            if ACT == "relu":
                x = tl.where(x >= 0, x, 0)
            elif ACT == "sigmoid":
                x = tl.sigmoid(x)
            elif ACT == "silu":
                x = x * tl.sigmoid(x)
            elif ACT == "softmax":
                # for stable
                x_minus_max = x - tl.max(x, axis=0)
                # softmax
                numerator = tl.exp(x_minus_max)
                denominator = tl.sum(numerator)
                x = numerator / denominator

        theta = theta_ * offset
        o_cos = x * tl.cos(theta)
        o_sin = x * tl.sin(theta)

        # save
        tl.store(o_cos_block_ptr, o_cos.to(o_cos_block_ptr.dtype.element_ty), mask=mask)
        tl.store(o_sin_block_ptr, o_sin.to(o_sin_block_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8]}),
    key=["h", "n", "d", "m"],
)
@triton.jit
def _lrpe_cosine_md_bwd_triton(
    X,
    Theta,
    DO,
    DX,
    Shape,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    l: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    m: tl.constexpr,
    ACT: tl.constexpr,
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

    # compute from the last theta block
    theta_block_ptr = Theta + offset_theta + tl.arange(0, BLOCK)
    dx_block_ptr = DX + offset_x + offset_d + tl.arange(0, BLOCK)
    x_block_ptr = X + offset_x + offset_d + tl.arange(0, BLOCK)
    do_cos_block_ptr = DO + offset_o + offset_d + tl.arange(0, BLOCK)
    do_sin_block_ptr = DO + offset_o + offset_d + d + tl.arange(0, BLOCK)
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
        dx_block_ptr -= e
        x_block_ptr -= e
        do_cos_block_ptr -= e
        do_sin_block_ptr -= e
        offset_d -= e
        mask = ((offset_d + tl.arange(0, BLOCK)) < d) & e_mask

        # compute dim
        dim = tl.sum(tl.load(shape_block_ptr, mask=shape_mask, other=0).to(tl.int32))
        offset = c % dim
        c = c // dim

        # compute
        do_cos = tl.load(do_cos_block_ptr, mask=mask, other=0).to(tl.float32)
        do_sin = tl.load(do_sin_block_ptr, mask=mask, other=0).to(tl.float32)
        theta = theta_ * offset
        dx = do_cos * tl.cos(theta) + do_sin * tl.sin(theta)

        if ACT != "none":
            if ACT == "softmax":
                value = -float("inf")
            else:
                value = 0

            x = tl.load(x_block_ptr, mask=mask, other=value).to(tl.float32)

            if ACT == "relu":
                dx = tl.where(x >= 0, dx, 0)
            elif ACT == "sigmoid":
                sigmoid = tl.sigmoid(x)
                dx = dx * sigmoid * (1 - sigmoid)
            elif ACT == "silu":
                sigmoid = tl.sigmoid(x)
                dx = dx * sigmoid * (1 + x * (1 - sigmoid))
            elif ACT == "softmax":
                # for stable
                x_minus_max = x - tl.max(x, axis=0)
                # softmax
                numerator = tl.exp(x_minus_max)
                denominator = tl.sum(numerator)
                o = numerator / denominator

                # scalar
                s = tl.sum(o * dx, axis=0)
                dx = o * dx - s * o

        tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=mask)


class LrpeCosineMdTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, theta, shape, l=0, act="none", dim=None):
        o = lrpe_cosine_md_fwd_triton(x, theta, shape, l, act, dim)

        ctx.save_for_backward(x, theta, shape)
        ctx.l = l
        ctx.act = act
        ctx.dim = dim

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, theta, shape = ctx.saved_tensors
        l = ctx.l
        act = ctx.act
        dim = ctx.dim

        dx = lrpe_cosine_md_bwd_triton(x, theta, do, shape, l, act, dim)

        return dx, None, None, None, None, None


def lrpe_cosine_md_fwd_triton(x, theta, shape, l=0, act="none", dim=None):
    assert dim in [-1, None], "dim must in [-1, None]"

    b, h, n, d = x.shape
    e = theta.shape[-1]
    m = len(shape)

    output_shape = list(x.shape)
    output_shape[-1] *= 2

    o = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    BLOCK = next_power_of_two(e)

    def grid(meta):
        return (b, h, n)

    _lrpe_cosine_md_fwd_triton[grid](
        x, theta, o, shape, b, h, n, l, d, e, m, act, BLOCK
    )

    return o


def lrpe_cosine_md_bwd_triton(x, theta, do, shape, l=0, act="none", dim=None, **kwargs):
    assert dim in [-1, None], "dim must in [-1, None]"

    b, h, n, d = x.shape
    e = theta.shape[-1]
    m = len(shape)

    dx = torch.empty_like(x)
    BLOCK = next_power_of_two(e)

    def grid(meta):
        return (b, h, n)

    _lrpe_cosine_md_bwd_triton[grid](
        x, theta, do, dx, shape, b, h, n, l, d, e, m, act, BLOCK
    )

    return dx


def lrpe_cosine_md_triton(x, theta, shape, l=0, act="none", dim=None, **kwargs):
    # x: b, h, n, d; n = l + prod(shape)
    # theta: h, e; e >= round(d + len(shape) - 1) // len(shape))
    # shape: n1, ... , nm
    # l: we do not do lrpe cosine on the first l tokens
    shape = torch.tensor(shape, dtype=torch.int32, device=x.device)
    assert (
        theta.shape[-1] * len(shape) >= x.shape[-1]
    ), "dim of theta should be larger than round(d + len(shape) - 1) // len(shape))"

    return LrpeCosineMdTriton.apply(x, theta, shape, l, act, dim)


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
    act = "silu"
    dim = -1

    o = lrpe_cosine_md_triton(x, theta, shape=shape[2:-1], act=act, dim=dim)
    o.backward(do)
