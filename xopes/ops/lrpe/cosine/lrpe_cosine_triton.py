import torch
import triton
import triton.language as tl

from xopes.ops.act import act_bwd
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
    ACT: tl.constexpr,
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
    ACT: tl.constexpr,
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

    if ACT != "none":
        x_block_ptr = X + offset_x + tl.arange(0, d)
        x = tl.load(x_block_ptr).to(tl.float32)
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
            c = tl.sum(o * dx, axis=0)
            dx = o * dx - c * o

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty))


##### act over sequence
@triton.autotune(
    generate_configs(
        {
            "BLOCK_N": [16, 32, 64, 128],
            "BLOCK_D": [16, 32, 64, 128],
            "num_warps": [2, 4, 8],
        }
    ),
    key=["h", "n", "d"],
)
@triton.jit
def _lrpe_cosine_fwd_seq_triton(
    X,
    Theta,
    O,
    offset: tl.constexpr,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_d = tl.program_id(2)
    # compute offset
    offset_d = off_d * BLOCK_D
    offset_x = off_b * h * n * d + off_h * n * d + offset_d
    offset_theta = off_h * d + offset_d
    offset_o = off_b * h * n * 2 * d + off_h * n * 2 * d + offset_d
    # compute block ptr
    x_block_ptr = (
        X
        + offset_x
        + tl.arange(0, BLOCK_N)[:, None] * d
        + tl.arange(0, BLOCK_D)[None, :]
    )
    # mask
    d_mask = (offset_d + tl.arange(0, BLOCK_D)) < d

    if ACT == "softmax":
        value = -float("inf")
    else:
        value = 0

    # get stat
    x_max = tl.full([BLOCK_D], value, dtype=tl.float32)
    denominator = tl.full([BLOCK_D], 0, dtype=tl.float32)
    for i in range(tl.cdiv(n, BLOCK_N)):
        n_mask = (i * BLOCK_N + tl.arange(0, BLOCK_N)) < n
        x = tl.load(x_block_ptr, mask=n_mask[:, None] & d_mask[None, :], other=value)

        x_block_max = tl.max(x, axis=0)
        x_max_ = tl.where(x_block_max > x_max, x_block_max, x_max)
        # sum(exp(xi - a)) + exp(x - a) = exp(b - a) * sum(exp(xi - b)) + exp(x - b)
        x_exp = tl.exp(x - x_max_)
        lambda_ = tl.exp(x_max - x_max_)
        denominator = lambda_ * denominator + tl.sum(x_exp, axis=0)
        x_max = x_max_

        x_block_ptr += BLOCK_N * d

    # compute block ptr
    theta_block_ptr = Theta + offset_theta + tl.arange(0, BLOCK_D)[None, :]
    x_block_ptr = (
        X
        + offset_x
        + tl.arange(0, BLOCK_N)[:, None] * d
        + tl.arange(0, BLOCK_D)[None, :]
    )
    o_cos_block_ptr = (
        O
        + offset_o
        + tl.arange(0, BLOCK_N)[:, None] * 2 * d
        + tl.arange(0, BLOCK_D)[None, :]
    )
    o_sin_block_ptr = (
        O
        + offset_o
        + d
        + tl.arange(0, BLOCK_N)[:, None] * 2 * d
        + tl.arange(0, BLOCK_D)[None, :]
    )
    array = tl.arange(0, BLOCK_N)
    theta_ = tl.load(theta_block_ptr).to(tl.float32)

    for i in range(tl.cdiv(n, BLOCK_N)):
        n_mask = array < n
        x = tl.load(x_block_ptr, mask=n_mask[:, None] & d_mask[None, :], other=value)

        if ACT == "softmax":
            # for stable
            x_minus_max = x - x_max
            # softmax
            numerator = tl.exp(x_minus_max)
            x = numerator / denominator

        theta = theta_ * (array[:, None] + offset)
        o_cos = x * tl.cos(theta)
        o_sin = x * tl.sin(theta)

        tl.store(o_cos_block_ptr, o_cos.to(o_cos_block_ptr.dtype.element_ty))
        tl.store(o_sin_block_ptr, o_sin.to(o_cos_block_ptr.dtype.element_ty))

        x_block_ptr += BLOCK_N * d
        array += BLOCK_N
        o_cos_block_ptr += BLOCK_N * 2 * d
        o_sin_block_ptr += BLOCK_N * 2 * d


class FusedActLrpeCosineTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, theta, offset=0, act="none", dim=None):
        o = lrpe_cosine_fwd_triton(x, theta, offset, act, dim)
        ctx.save_for_backward(x, theta)
        ctx.offset = offset
        ctx.act = act
        ctx.dim = dim

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, theta = ctx.saved_tensors
        offset = ctx.offset
        act = ctx.act
        dim = ctx.dim
        dx = lrpe_cosine_bwd_triton(x, theta, do, offset, act, dim)

        return dx, None, None, None, None


def lrpe_cosine_fwd_triton(x, theta, offset=0, act="none", dim=None):
    assert dim in [-1, -2, None], "dim must be -1, -2, None"

    b, h, n, d = x.shape
    o = torch.empty(b, h, n, 2 * d, dtype=x.dtype, device=x.device)

    if dim in [-1, None] or act == "none":

        def grid(meta):
            return (b, h, n)

        _lrpe_cosine_fwd_triton[grid](x, theta, o, offset, b, h, n, d, act)
    else:
        assert act in ["softmax"]

        def grid(meta):
            return (b, h, triton.cdiv(d, meta["BLOCK_D"]))

        _lrpe_cosine_fwd_seq_triton[grid](x, theta, o, offset, b, h, n, d, act)

    return o


def lrpe_cosine_bwd_triton(x, theta, do, offset=0, act="none", dim=None):
    assert dim in [-1, -2, None], "dim must be -1, -2, None"
    b, h, n, d = x.shape
    dx = torch.empty_like(x)

    def grid(meta):
        return (b, h, n)

    _lrpe_cosine_bwd_triton[grid](x, theta, do, dx, offset, b, h, n, d, act)

    if dim == -2:
        dx = act_bwd(x, dx, act, dim)

    return dx


def lrpe_cosine_triton(x, theta, offset=0, act="none", dim=None):
    # x: b, h, n, d
    # theta: h, d
    return FusedActLrpeCosineTriton.apply(x, theta, offset, act, dim)


if __name__ == "__main__":
    # unit test
    b, h, n, d = 2, 8, 128, 64
    dtype = torch.float32
    device = torch.cuda.current_device()
    x = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    theta = torch.randn((h, d), dtype=dtype, device=device)
    do = torch.randn((b, h, n, 2 * d), dtype=dtype, device=device)
    act = "silu"
    dim = -1

    o = lrpe_cosine_triton(x, theta, act=act, dim=dim)
    o.backward(do)
