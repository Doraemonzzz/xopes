import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs(
        {
            "BLOCK_N": [16, 32, 64, 128],
            "BLOCK_D": [16, 32, 64, 128],
            "num_warps": [2, 4, 8],
        }
    ),
    key=["n", "d"],
)
@triton.jit
def _lrpe_cosine_1d_bp_fwd_triton(
    X,
    Theta,
    O,
    X_STAT1,
    X_STAT2,
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
    if ACT != "none":
        if ACT == "softmax":
            x_max = tl.full([BLOCK_D], value, dtype=tl.float32)
            denominator = tl.full([BLOCK_D], 0, dtype=tl.float32)
            for i in range(tl.cdiv(n, BLOCK_N)):
                n_mask = (i * BLOCK_N + tl.arange(0, BLOCK_N)) < n
                x = tl.load(
                    x_block_ptr, mask=n_mask[:, None] & d_mask[None, :], other=value
                )

                x_block_max = tl.max(x, axis=0)
                x_max_ = tl.where(x_block_max > x_max, x_block_max, x_max)
                # sum(exp(xi - a)) + exp(x - a) = exp(b - a) * sum(exp(xi - b)) + exp(x - b)
                x_exp = tl.exp(x - x_max_)
                lambda_ = tl.exp(x_max - x_max_)
                denominator = lambda_ * denominator + tl.sum(x_exp, axis=0)
                x_max = x_max_

                x_block_ptr += BLOCK_N * d

            # save
            x_stat1_block_ptr = (
                X_STAT1 + off_b * h * d + off_h * d + offset_d + tl.arange(0, BLOCK_D)
            )
            x_stat2_block_ptr = (
                X_STAT2 + off_b * h * d + off_h * d + offset_d + tl.arange(0, BLOCK_D)
            )

            tl.store(
                x_stat1_block_ptr,
                x_max.to(x_stat1_block_ptr.dtype.element_ty),
                mask=d_mask,
            )
            tl.store(
                x_stat2_block_ptr,
                denominator.to(x_stat2_block_ptr.dtype.element_ty),
                mask=d_mask,
            )

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
    theta_ = tl.load(theta_block_ptr, mask=d_mask[None, :], other=0).to(tl.float32)

    for i in range(tl.cdiv(n, BLOCK_N)):
        n_mask = array < n
        mask = n_mask[:, None] & d_mask[None, :]
        x = tl.load(x_block_ptr, mask=mask, other=0).to(tl.float32)

        if ACT != "none":
            if ACT == "relu":
                x = tl.where(x >= 0, x, 0)
            elif ACT == "sigmoid":
                x = tl.sigmoid(x)
            elif ACT == "silu":
                x = x * tl.sigmoid(x)
            elif ACT == "softmax":
                # for stable
                x_minus_max = x - x_max
                # softmax
                numerator = tl.exp(x_minus_max)
                x = numerator / denominator

        theta = theta_ * (array[:, None] + offset)
        o_cos = x * tl.cos(theta)
        o_sin = x * tl.sin(theta)

        tl.store(o_cos_block_ptr, o_cos.to(o_cos_block_ptr.dtype.element_ty), mask=mask)
        tl.store(o_sin_block_ptr, o_sin.to(o_cos_block_ptr.dtype.element_ty), mask=mask)

        x_block_ptr += BLOCK_N * d
        array += BLOCK_N
        o_cos_block_ptr += BLOCK_N * 2 * d
        o_sin_block_ptr += BLOCK_N * 2 * d


@triton.autotune(
    generate_configs(
        {
            "BLOCK_N": [16, 32, 64, 128],
            "BLOCK_D": [16, 32, 64, 128],
            "num_warps": [2, 4, 8],
        }
    ),
    key=["n", "d"],
)
@triton.jit
def _lrpe_cosine_1d_bp_bwd_triton(
    X,
    Theta,
    DO,
    DX,
    X_STAT1,
    X_STAT2,
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
    theta_block_ptr = Theta + offset_theta + tl.arange(0, BLOCK_D)[None, :]
    dx_block_ptr = (
        DX
        + offset_x
        + tl.arange(0, BLOCK_N)[:, None] * d
        + tl.arange(0, BLOCK_D)[None, :]
    )
    do_cos_block_ptr = (
        DO
        + offset_o
        + tl.arange(0, BLOCK_N)[:, None] * 2 * d
        + tl.arange(0, BLOCK_D)[None, :]
    )
    do_sin_block_ptr = (
        DO
        + offset_o
        + d
        + tl.arange(0, BLOCK_N)[:, None] * 2 * d
        + tl.arange(0, BLOCK_D)[None, :]
    )
    array = tl.arange(0, BLOCK_N)
    # mask
    d_mask = (offset_d + tl.arange(0, BLOCK_D)) < d

    theta_ = tl.load(theta_block_ptr, mask=d_mask[None, :], other=0).to(tl.float32)

    if ACT == "softmax":  # compute c first
        x_block_ptr = (
            X
            + offset_x
            + tl.arange(0, BLOCK_N)[:, None] * d
            + tl.arange(0, BLOCK_D)[None, :]
        )
        x_stat1_block_ptr = (
            X_STAT1 + off_b * h * d + off_h * d + offset_d + tl.arange(0, BLOCK_D)
        )
        x_stat2_block_ptr = (
            X_STAT2 + off_b * h * d + off_h * d + offset_d + tl.arange(0, BLOCK_D)
        )
        x_max = tl.load(x_stat1_block_ptr, mask=d_mask, other=0).to(tl.float32)
        denominator = tl.load(x_stat2_block_ptr, mask=d_mask, other=1).to(tl.float32)

        c = tl.zeros([BLOCK_D], dtype=tl.float32)

        for i in range(tl.cdiv(n, BLOCK_N)):
            n_mask = array < n
            mask = n_mask[:, None] & d_mask[None, :]

            do_cos = tl.load(do_cos_block_ptr, mask=mask, other=0).to(tl.float32)
            do_sin = tl.load(do_sin_block_ptr, mask=mask, other=0).to(tl.float32)

            theta = theta_ * (array[:, None] + offset)
            dx = do_cos * tl.cos(theta) + do_sin * tl.sin(theta)

            x = tl.load(x_block_ptr, mask=mask, other=0).to(tl.float32)
            # for stable
            x_minus_max = x - x_max
            # softmax
            numerator = tl.exp(x_minus_max)
            o = numerator / denominator

            # scalar
            c += tl.sum(o * dx, axis=0)

            x_block_ptr += BLOCK_N * d
            array += BLOCK_N
            do_cos_block_ptr += BLOCK_N * 2 * d
            do_sin_block_ptr += BLOCK_N * 2 * d

        # reinit
        do_cos_block_ptr = (
            DO
            + offset_o
            + tl.arange(0, BLOCK_N)[:, None] * 2 * d
            + tl.arange(0, BLOCK_D)[None, :]
        )
        do_sin_block_ptr = (
            DO
            + offset_o
            + d
            + tl.arange(0, BLOCK_N)[:, None] * 2 * d
            + tl.arange(0, BLOCK_D)[None, :]
        )
        array = tl.arange(0, BLOCK_N)

    for i in range(tl.cdiv(n, BLOCK_N)):
        n_mask = array < n
        mask = n_mask[:, None] & d_mask[None, :]

        do_cos = tl.load(do_cos_block_ptr, mask=mask, other=0).to(tl.float32)
        do_sin = tl.load(do_sin_block_ptr, mask=mask, other=0).to(tl.float32)

        theta = theta_ * (array[:, None] + offset)
        dx = do_cos * tl.cos(theta) + do_sin * tl.sin(theta)

        if ACT != "none":
            x_block_ptr = (
                X
                + offset_x
                + i * BLOCK_N * d
                + tl.arange(0, BLOCK_N)[:, None] * d
                + tl.arange(0, BLOCK_D)[None, :]
            )
            x = tl.load(x_block_ptr, mask=mask, other=0).to(tl.float32)
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
                x_minus_max = x - x_max
                # softmax
                numerator = tl.exp(x_minus_max)
                o = numerator / denominator
                # scalar
                dx = o * dx - c * o

        tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=mask)

        dx_block_ptr += BLOCK_N * d
        array += BLOCK_N
        do_cos_block_ptr += BLOCK_N * 2 * d
        do_sin_block_ptr += BLOCK_N * 2 * d


class LrpeCosine1dBpTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, theta, offset=0, act="none", dim=None):
        o, x_stat1, x_stat2 = lrpe_cosine_1d_bp_fwd_triton(x, theta, offset, act, dim)

        ctx.save_for_backward(x, theta, x_stat1, x_stat2)
        ctx.offset = offset
        ctx.act = act
        ctx.dim = dim

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, theta, x_stat1, x_stat2 = ctx.saved_tensors
        offset = ctx.offset
        act = ctx.act
        dim = ctx.dim

        dx = lrpe_cosine_1d_bp_bwd_triton(
            x, theta, do, x_stat1, x_stat2, offset, act, dim
        )

        return dx, None, None, None, None


def lrpe_cosine_1d_bp_fwd_triton(x, theta, offset=0, act="none", dim=None, **kwargs):
    assert dim in [-2, None], "dim must in [-2, None]"

    b, h, n, d = x.shape
    o = torch.empty(b, h, n, 2 * d, dtype=x.dtype, device=x.device)
    x_stat1 = torch.empty(b, h, d, dtype=x.dtype, device=x.device)
    x_stat2 = torch.empty(b, h, d, dtype=x.dtype, device=x.device)

    def grid(meta):
        return (b, h, triton.cdiv(d, meta["BLOCK_D"]))

    _lrpe_cosine_1d_bp_fwd_triton[grid](
        x, theta, o, x_stat1, x_stat2, offset, b, h, n, d, act
    )

    return o, x_stat1, x_stat2


def lrpe_cosine_1d_bp_bwd_triton(
    x, theta, do, x_stat1, x_stat2, offset=0, act="none", dim=None, **kwargs
):
    assert dim in [-2, None], "dim must in [-2, None]"

    b, h, n, d = x.shape
    dx = torch.empty_like(x)

    def grid(meta):
        return (b, h, triton.cdiv(d, meta["BLOCK_D"]))

    _lrpe_cosine_1d_bp_bwd_triton[grid](
        x, theta, do, dx, x_stat1, x_stat2, offset, b, h, n, d, act
    )

    return dx


def lrpe_cosine_1d_bp_triton(x, theta, offset=0, act="none", dim=None, **kwargs):
    # x: b, h, n, d
    # theta: h, d
    assert dim in [-2, None], "dim must in [-2, None]"
    return LrpeCosine1dBpTriton.apply(x, theta, offset, act, dim)


if __name__ == "__main__":
    # unit test
    b, h, n, d = 2, 8, 128, 64
    dtype = torch.float32
    device = torch.cuda.current_device()
    x = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    theta = torch.randn((h, d), dtype=dtype, device=device)
    do = torch.randn((b, h, n, 2 * d), dtype=dtype, device=device)
    act = "silu"
    dim = -2

    o = lrpe_cosine_1d_bp_triton(x, theta, act=act, dim=dim)
    o.backward(do)
