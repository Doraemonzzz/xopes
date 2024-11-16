import torch
import triton
import triton.language as tl

from xopes.utils import ACT_SET, contiguous, generate_configs, next_power_of_two


@triton.autotune(
    generate_configs({"BLOCK_N": [16, 32, 64, 128], "num_warps": [2, 4, 8]}),
    key=["h", "n", "d", "m"],
)
@triton.jit
def _lrpe_cosine_md_bp_fwd_triton(
    X,
    Theta,
    O,
    Shape,
    ThetaCache,
    X_STAT1,
    X_STAT2,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    l: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    m: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    # compute offset
    offset_x = off_b * h * n * d + off_h * n * d
    offset_theta = off_h * e
    offset_o = off_b * h * n * 2 * d + off_h * n * 2 * d
    offset_d = m * e
    offset_theta_cache = off_h * n * d + l * d

    if ACT == "softmax":
        value = -float("inf")
    else:
        value = 0

    # get stat
    # for softmax act, we should compute max and denominator first
    if ACT == "softmax":
        # mask
        d_mask = tl.arange(0, BLOCK_D) < d

        x_block_ptr_ = (
            X
            + offset_x
            + tl.arange(0, BLOCK_N)[:, None] * d
            + tl.arange(0, BLOCK_D)[None, :]
        )
        x_max = tl.full([BLOCK_D], value, dtype=tl.float32)
        denominator = tl.full([BLOCK_D], 0, dtype=tl.float32)

        for i in range(tl.cdiv(n, BLOCK_N)):
            n_mask = (i * BLOCK_N + tl.arange(0, BLOCK_N)) < n
            x_ = tl.load(
                x_block_ptr_, mask=n_mask[:, None] & d_mask[None, :], other=value
            )

            x_block_max = tl.max(x_, axis=0)
            x_max_ = tl.where(x_block_max > x_max, x_block_max, x_max)
            # sum(exp(xi - a)) + exp(x - a) = exp(b - a) * sum(exp(xi - b)) + exp(x - b)
            x_exp = tl.exp(x_ - x_max_)
            lambda_ = tl.exp(x_max - x_max_)
            denominator = lambda_ * denominator + tl.sum(x_exp, axis=0)
            x_max = x_max_

            x_block_ptr_ += BLOCK_N * d

        # save
        x_stat1_block_ptr = X_STAT1 + off_b * h * d + off_h * d + tl.arange(0, BLOCK_D)
        x_stat2_block_ptr = X_STAT2 + off_b * h * d + off_h * d + tl.arange(0, BLOCK_D)

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

    # compute the first l element
    if l > 0:
        offset_theta_cache_l = off_h * n * d
        theta_cache_block_ptr_l = (
            ThetaCache
            + offset_theta_cache_l
            + tl.arange(0, BLOCK_L)[:, None]
            + tl.arange(0, BLOCK_D)[None, :]
        )

        x_block_ptr_l = (
            X
            + offset_x
            + tl.arange(0, BLOCK_L)[:, None] * d
            + tl.arange(0, BLOCK_D)[None, :]
        )
        o_cos_block_ptr_l = (
            O
            + offset_o
            + tl.arange(0, BLOCK_L)[:, None] * 2 * d
            + tl.arange(0, BLOCK_D)[None, :]
        )
        o_sin_block_ptr_l = (
            O
            + offset_o
            + d
            + tl.arange(0, BLOCK_L)[:, None] * 2 * d
            + tl.arange(0, BLOCK_D)[None, :]
        )
        ld_mask = (tl.arange(0, BLOCK_L) < l)[:, None] & (
            tl.arange(0, BLOCK_D)[None, :] < d
        )
        x_l = tl.load(x_block_ptr_l, mask=ld_mask, other=0)

        if ACT != "none":
            if ACT == "relu":
                x_l = tl.where(x_l >= 0, x_l, 0)
            elif ACT == "sigmoid":
                x_l = tl.sigmoid(x_l)
            elif ACT == "silu":
                x_l = x_l * tl.sigmoid(x_l)
            elif ACT == "softmax":
                # for stable
                x_l_minus_max = x_l - x_max
                # softmax
                numerator_l = tl.exp(x_l_minus_max)
                x_l = numerator_l / denominator

        zero = tl.zeros([BLOCK_L, BLOCK_D], dtype=x_l.dtype)
        # save
        tl.store(
            o_cos_block_ptr_l, x_l.to(o_cos_block_ptr_l.dtype.element_ty), mask=ld_mask
        )
        tl.store(
            o_sin_block_ptr_l, zero.to(o_sin_block_ptr_l.dtype.element_ty), mask=ld_mask
        )
        tl.store(
            theta_cache_block_ptr_l,
            zero.to(theta_cache_block_ptr_l.dtype.element_ty),
            mask=ld_mask,
        )

    # compute from the last theta block
    x_block_ptr = (
        X
        + offset_x
        + l * d
        + offset_d
        + tl.arange(0, BLOCK_N)[:, None] * d
        + tl.arange(0, BLOCK_E)[None, :]
    )
    theta_block_ptr = Theta + offset_theta + tl.arange(0, BLOCK_E)[None, :]
    o_cos_block_ptr = (
        O
        + offset_o
        + 2 * l * d
        + offset_d
        + tl.arange(0, BLOCK_N)[:, None] * 2 * d
        + tl.arange(0, BLOCK_E)[None, :]
    )
    o_sin_block_ptr = (
        O
        + offset_o
        + 2 * l * d
        + offset_d
        + d
        + tl.arange(0, BLOCK_N)[:, None] * 2 * d
        + tl.arange(0, BLOCK_E)[None, :]
    )
    # triton only support load block at least 16 elements, use this to get shape
    shape_mask = tl.arange(0, 16) < 1
    # mask
    e_mask = tl.arange(0, BLOCK_E) < e

    theta_ = tl.load(theta_block_ptr, mask=e_mask[None, :], other=0).to(tl.float32)
    array = tl.arange(0, BLOCK_N)
    theta_cache_block_ptr = (
        ThetaCache
        + offset_theta_cache
        + offset_d
        + tl.arange(0, BLOCK_N)[:, None] * e
        + tl.arange(0, BLOCK_E)[None, :]
    )

    for i in range(tl.cdiv(n - l, BLOCK_N)):
        n_mask = array < n - l  # !!! important
        c = array[:, None]
        offset_d = m * e
        # triton only support load block at least 16 elements, use this to get shape
        shape_block_ptr = Shape + m + tl.arange(0, 16)
        if ACT == "softmax":
            x_max_block_ptr = (
                X_STAT1
                + off_b * h * d
                + off_h * d
                + offset_d
                + tl.arange(0, BLOCK_E)[None, :]
            )
            denominator_block_ptr = (
                X_STAT2
                + off_b * h * d
                + off_h * d
                + offset_d
                + tl.arange(0, BLOCK_E)[None, :]
            )

        for j in range(m):
            # update block ptr
            shape_block_ptr -= 1
            x_block_ptr -= e
            o_cos_block_ptr -= e
            o_sin_block_ptr -= e
            offset_d -= e
            theta_cache_block_ptr -= e

            de_mask = ((offset_d + tl.arange(0, BLOCK_E)) < d) & e_mask
            mask = n_mask[:, None] & de_mask[None, :]

            # compute dim
            dim = tl.sum(
                tl.load(shape_block_ptr, mask=shape_mask, other=0).to(tl.int32)
            )
            offset = c % dim
            c = c // dim

            x = tl.load(x_block_ptr, mask=mask, other=value).to(tl.float32)
            if ACT != "none":
                if ACT == "relu":
                    x = tl.where(x >= 0, x, 0)
                elif ACT == "sigmoid":
                    x = tl.sigmoid(x)
                elif ACT == "silu":
                    x = x * tl.sigmoid(x)
                elif ACT == "softmax":
                    x_max_block_ptr -= e
                    denominator_block_ptr -= e
                    x_max_ = tl.load(
                        x_max_block_ptr, mask=de_mask[None, :], other=0
                    ).to(tl.float32)
                    denominator_ = tl.load(
                        denominator_block_ptr, mask=de_mask[None, :], other=1
                    ).to(tl.float32)
                    # for stable
                    x_minus_max_ = x - x_max_
                    # softmax
                    numerator_ = tl.exp(x_minus_max_)
                    x = numerator_ / denominator_

            theta = theta_ * offset
            o_cos = x * tl.cos(theta)
            o_sin = x * tl.sin(theta)

            # save
            tl.store(
                o_cos_block_ptr, o_cos.to(o_cos_block_ptr.dtype.element_ty), mask=mask
            )
            tl.store(
                o_sin_block_ptr, o_sin.to(o_sin_block_ptr.dtype.element_ty), mask=mask
            )
            if i == 0:
                tl.store(
                    theta_cache_block_ptr,
                    theta.to(theta_cache_block_ptr.dtype.element_ty),
                    mask=mask,
                )

        x_block_ptr += BLOCK_N * d + e * m
        array += BLOCK_N
        o_cos_block_ptr += BLOCK_N * 2 * d + e * m
        o_sin_block_ptr += BLOCK_N * 2 * d + e * m


@triton.autotune(
    generate_configs({"BLOCK_N": [16, 32, 64, 128], "num_warps": [2, 4, 8]}),
    key=["h", "n", "d", "m"],
)
@triton.jit
def _lrpe_cosine_md_bp_bwd_triton(
    X,
    Theta,
    DO,
    DX,
    Shape,
    ThetaCache,
    X_STAT1,
    X_STAT2,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    l: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    m: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    # compute offset
    offset_x = off_b * h * n * d + off_h * n * d
    offset_o = off_b * h * n * 2 * d + off_h * n * 2 * d
    offset_theta_cache = off_h * n * d
    # compute block ptr
    theta_block_ptr = (
        ThetaCache
        + offset_theta_cache
        + tl.arange(0, BLOCK_N)[:, None] * d
        + tl.arange(0, BLOCK_D)[None, :]
    )
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
    d_mask = tl.arange(0, BLOCK_D) < d

    if ACT == "softmax":  # compute c first
        x_block_ptr = (
            X
            + offset_x
            + tl.arange(0, BLOCK_N)[:, None] * d
            + tl.arange(0, BLOCK_D)[None, :]
        )
        x_stat1_block_ptr = X_STAT1 + off_b * h * d + off_h * d + tl.arange(0, BLOCK_D)
        x_stat2_block_ptr = X_STAT2 + off_b * h * d + off_h * d + tl.arange(0, BLOCK_D)
        x_max = tl.load(x_stat1_block_ptr, mask=d_mask, other=0).to(tl.float32)
        denominator = tl.load(x_stat2_block_ptr, mask=d_mask, other=1).to(tl.float32)

        c = tl.zeros([BLOCK_D], dtype=tl.float32)

        for i in range(tl.cdiv(n, BLOCK_N)):
            n_mask = array < n
            mask = n_mask[:, None] & d_mask[None, :]

            do_cos = tl.load(do_cos_block_ptr, mask=mask, other=0).to(tl.float32)
            do_sin = tl.load(do_sin_block_ptr, mask=mask, other=0).to(tl.float32)
            theta = tl.load(theta_block_ptr, mask=mask, other=0).to(tl.float32)

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
            theta_block_ptr += BLOCK_N * d

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
        theta_block_ptr = (
            ThetaCache
            + offset_theta_cache
            + tl.arange(0, BLOCK_N) * d
            + tl.arange(0, BLOCK_D)[None, :]
        )

    for i in range(tl.cdiv(n, BLOCK_N)):
        n_mask = array < n
        mask = n_mask[:, None] & d_mask[None, :]

        do_cos = tl.load(do_cos_block_ptr, mask=mask, other=0).to(tl.float32)
        do_sin = tl.load(do_sin_block_ptr, mask=mask, other=0).to(tl.float32)
        theta = tl.load(theta_block_ptr, mask=mask, other=0).to(tl.float32)

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
        theta_block_ptr += BLOCK_N * d


class LrpeCosineMdBpTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, theta, shape, l=0, act="none", dim=None):
        o, theta_cache, x_stat1, x_stat2 = lrpe_cosine_md_bp_fwd_triton(
            x, theta, shape, l, act, dim
        )

        ctx.save_for_backward(x, theta, shape, theta_cache, x_stat1, x_stat2)
        ctx.l = l
        ctx.act = act
        ctx.dim = dim

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, theta, shape, theta_cache, x_stat1, x_stat2 = ctx.saved_tensors
        l = ctx.l
        act = ctx.act
        dim = ctx.dim

        dx = lrpe_cosine_md_bp_bwd_triton(
            x, theta, do, shape, theta_cache, x_stat1, x_stat2, l, act, dim
        )

        return dx, None, None, None, None, None


def lrpe_cosine_md_bp_fwd_triton(x, theta, shape, l=0, act="none", dim=None):
    assert act in ACT_SET, f"act: {act} not in {ACT_SET}"
    assert dim in [-2, None], "dim must in [-2, None]"

    b, h, n, d = x.shape
    e = theta.shape[-1]
    m = len(shape)

    output_shape = list(x.shape)
    output_shape[-1] *= 2

    o = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    theta_cache = torch.empty((h, n, d), dtype=torch.float32, device=theta.device)
    # max
    x_stat1 = torch.empty(b, h, d, dtype=x.dtype, device=x.device)
    # denominator
    x_stat2 = torch.empty(b, h, d, dtype=x.dtype, device=x.device)

    BLOCK_D = next_power_of_two(d)
    BLOCK_E = next_power_of_two(e)
    BLOCK_L = next_power_of_two(l) if l > 0 else 0

    def grid(meta):
        return (b, h)

    _lrpe_cosine_md_bp_fwd_triton[grid](
        x,
        theta,
        o,
        shape,
        theta_cache,
        x_stat1,
        x_stat2,
        b,
        h,
        n,
        l,
        d,
        e,
        m,
        act,
        BLOCK_D,
        BLOCK_E,
        BLOCK_L,
    )

    return o, theta_cache, x_stat1, x_stat2


def lrpe_cosine_md_bp_bwd_triton(
    x,
    theta,
    do,
    shape,
    theta_cache,
    x_stat1,
    x_stat2,
    l=0,
    act="none",
    dim=None,
    **kwargs,
):
    assert act in ACT_SET, f"act: {act} not in {ACT_SET}"
    assert dim in [-2, None], "dim must in [-2, None]"

    b, h, n, d = x.shape
    e = theta.shape[-1]
    m = len(shape)

    dx = torch.empty_like(x)
    BLOCK_D = next_power_of_two(d)
    BLOCK_E = next_power_of_two(e)
    BLOCK_L = next_power_of_two(l) if l > 0 else 0

    def grid(meta):
        return (b, h)

    _lrpe_cosine_md_bp_bwd_triton[grid](
        x,
        theta,
        do,
        dx,
        shape,
        theta_cache,
        x_stat1,  # max
        x_stat2,  # denominator
        b,
        h,
        n,
        l,
        d,
        e,
        m,
        act,
        BLOCK_D,
        BLOCK_E,
        BLOCK_L,
    )

    return dx


def lrpe_cosine_md_bp_triton(x, theta, shape, l=0, act="none", dim=None, **kwargs):
    # x: b, h, n, d; n = l + prod(shape)
    # theta: h, e; e >= round(d + len(shape) - 1) // len(shape))
    # shape: n1, ... , nm
    # l: we do not do lrpe cosine on the first l tokens
    assert act in ACT_SET, f"act: {act} not in {ACT_SET}"
    assert dim in [-2, None], "dim must in [-2, None]"
    shape = torch.tensor(shape, dtype=torch.int32, device=x.device)
    assert (
        theta.shape[-1] * len(shape) >= x.shape[-1]
    ), "dim of theta should be larger than round(d + len(shape) - 1) // len(shape))"

    return LrpeCosineMdBpTriton.apply(x, theta, shape, l, act, dim)


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
    dim = -2

    o = lrpe_cosine_md_bp_triton(x, theta, shape=shape[2:-1], act=act, dim=dim)
    o.backward(do)
