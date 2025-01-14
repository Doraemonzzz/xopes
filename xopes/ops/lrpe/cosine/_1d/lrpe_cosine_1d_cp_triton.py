import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs(
        {
            "BLOCK_N": [16, 32, 64, 128],
            "BLOCK_D": [16, 32, 64, 128],
            "num_warps": [2, 4, 8, 16, 32],
        }
    ),
    # key=["N"],
    key=["N", "H", "D", "ACT"],
)
@triton.jit
def _lrpe_cosine_1d_cp_fwd_triton(
    X,  # B N H D
    THETA,  # H D / H 1 / 1 D
    O,  # B N H 2D
    X_STAT1,  # B H D
    X_STAT2,  # B H D
    OFFSET: tl.constexpr,
    THETA_TYPE: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    H_T: tl.constexpr,
    H_D: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_d = tl.program_id(2)
    # compute offset
    offset_d = off_d * BLOCK_D
    offset_x = off_b * N * H * D + off_h * D + offset_d
    if THETA_TYPE == 1:  # H D
        offset_theta = off_h * D + offset_d
    elif THETA_TYPE == 2:  # H 1
        offset_theta = off_h
    elif THETA_TYPE == 3:  # 1 D
        offset_theta = offset_d
    offset_o = off_b * N * H * 2 * D + off_h * 2 * D + offset_d
    # compute block ptr
    x_block_ptr = (
        X
        + offset_x
        + tl.arange(0, BLOCK_N)[:, None] * H * D
        + tl.arange(0, BLOCK_D)[None, :]
    )
    # mask
    mask_d = (offset_d + tl.arange(0, BLOCK_D)) < D

    if ACT == "softmax":
        value = -float("inf")
    else:
        value = 0

    # get stat
    if ACT != "none":
        if ACT == "softmax":
            x_max = tl.full([BLOCK_D], value, dtype=tl.float32)
            denominator = tl.full([BLOCK_D], 0, dtype=tl.float32)
            for i in range(tl.cdiv(N, BLOCK_N)):
                mask_n = (i * BLOCK_N + tl.arange(0, BLOCK_N)) < N
                x = tl.load(
                    x_block_ptr, mask=mask_n[:, None] & mask_d[None, :], other=value
                )

                x_block_max = tl.max(x, axis=0)
                x_max_ = tl.maximum(x_block_max, x_max)
                # sum(exp(xi - a)) + exp(x - a) = exp(b - a) * sum(exp(xi - b)) + exp(x - b)
                x_exp = tl.exp(x - x_max_)
                lambda_ = tl.exp(x_max - x_max_)
                denominator = lambda_ * denominator + tl.sum(x_exp, axis=0)
                x_max = x_max_

                x_block_ptr += BLOCK_N * H * D

            # save
            x_stat1_block_ptr = (
                X_STAT1 + off_b * H * D + off_h * D + offset_d + tl.arange(0, BLOCK_D)
            )
            x_stat2_block_ptr = (
                X_STAT2 + off_b * H * D + off_h * D + offset_d + tl.arange(0, BLOCK_D)
            )

            tl.store(
                x_stat1_block_ptr,
                x_max.to(x_stat1_block_ptr.dtype.element_ty),
                mask=mask_d,
            )
            tl.store(
                x_stat2_block_ptr,
                denominator.to(x_stat2_block_ptr.dtype.element_ty),
                mask=mask_d,
            )

    # compute block ptr
    x_block_ptr = (
        X
        + offset_x
        + tl.arange(0, BLOCK_N)[:, None] * H * D
        + tl.arange(0, BLOCK_D)[None, :]
    )

    o_cos_block_ptr = (
        O
        + offset_o
        + tl.arange(0, BLOCK_N)[:, None] * H * 2 * D
        + tl.arange(0, BLOCK_D)[None, :]
    )
    o_sin_block_ptr = (
        O
        + offset_o
        + D
        + tl.arange(0, BLOCK_N)[:, None] * H * 2 * D
        + tl.arange(0, BLOCK_D)[None, :]
    )
    array = tl.arange(0, BLOCK_N)
    if H_D != 1:
        theta_block_ptr = THETA + offset_theta + tl.arange(0, BLOCK_D)[None, :]
        theta_ = tl.load(theta_block_ptr, mask=mask_d[None, :], other=0).to(tl.float32)
    else:  # scalar version
        theta_block_ptr = THETA + offset_theta + tl.arange(0, 1)[None, :]
        theta_ = tl.load(theta_block_ptr).to(tl.float32)[None, :]

    for i in range(tl.cdiv(N, BLOCK_N)):
        mask_n = array < N
        mask = mask_n[:, None] & mask_d[None, :]
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

        theta = theta_ * (array[:, None] + OFFSET)
        o_cos = x * tl.cos(theta)
        o_sin = x * tl.sin(theta)

        tl.store(o_cos_block_ptr, o_cos.to(o_cos_block_ptr.dtype.element_ty), mask=mask)
        tl.store(o_sin_block_ptr, o_sin.to(o_cos_block_ptr.dtype.element_ty), mask=mask)

        x_block_ptr += BLOCK_N * H * D
        array += BLOCK_N
        o_cos_block_ptr += BLOCK_N * H * 2 * D
        o_sin_block_ptr += BLOCK_N * H * 2 * D


@triton.autotune(
    generate_configs(
        {
            "BLOCK_N": [16, 32, 64, 128],
            "BLOCK_D": [16, 32, 64, 128],
            "num_warps": [2, 4, 8, 16, 32],
        }
    ),
    key=["N", "H", "D", "ACT"],
)
@triton.jit
def _lrpe_cosine_1d_cp_bwd_triton(
    X,  # B N H D
    THETA,  # H D / H 1 / 1 D
    DO,  # B N H 2D
    DX,  # B N H D
    X_STAT1,  # B H D
    X_STAT2,  # B H D
    OFFSET: tl.constexpr,
    THETA_TYPE: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    H_T: tl.constexpr,
    H_D: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_d = tl.program_id(2)
    # compute offset
    offset_d = off_d * BLOCK_D
    offset_x = off_b * N * H * D + off_h * D + offset_d
    if THETA_TYPE == 1:  # H D
        offset_theta = off_h * D + offset_d
    elif THETA_TYPE == 2:  # H 1
        offset_theta = off_h
    elif THETA_TYPE == 3:  # 1 D
        offset_theta = offset_d
    offset_o = off_b * N * H * 2 * D + off_h * 2 * D + offset_d
    dx_block_ptr = (
        DX
        + offset_x
        + tl.arange(0, BLOCK_N)[:, None] * H * D
        + tl.arange(0, BLOCK_D)[None, :]
    )
    do_cos_block_ptr = (
        DO
        + offset_o
        + tl.arange(0, BLOCK_N)[:, None] * H * 2 * D
        + tl.arange(0, BLOCK_D)[None, :]
    )
    do_sin_block_ptr = (
        DO
        + offset_o
        + D
        + tl.arange(0, BLOCK_N)[:, None] * 2 * D
        + tl.arange(0, BLOCK_D)[None, :]
    )
    array = tl.arange(0, BLOCK_N)
    # mask
    mask_d = (offset_d + tl.arange(0, BLOCK_D)) < D
    if H_D != 1:
        theta_block_ptr = THETA + offset_theta + tl.arange(0, BLOCK_D)[None, :]
        theta_ = tl.load(theta_block_ptr, mask=mask_d[None, :], other=0).to(tl.float32)
    else:  # scalar version
        theta_block_ptr = THETA + offset_theta + tl.arange(0, 1)[None, :]
        theta_ = tl.load(theta_block_ptr).to(tl.float32)[None, :]

    if ACT == "softmax":  # compute c first
        x_block_ptr = (
            X
            + offset_x
            + tl.arange(0, BLOCK_N)[:, None] * H * D
            + tl.arange(0, BLOCK_D)[None, :]
        )
        x_stat1_block_ptr = (
            X_STAT1 + off_b * N * H * D + off_h * D + offset_d + tl.arange(0, BLOCK_D)
        )
        x_stat2_block_ptr = (
            X_STAT2 + off_b * N * H * D + off_h * D + offset_d + tl.arange(0, BLOCK_D)
        )
        x_max = tl.load(x_stat1_block_ptr, mask=mask_d, other=0).to(tl.float32)
        denominator = tl.load(x_stat2_block_ptr, mask=mask_d, other=1).to(tl.float32)

        c = tl.zeros([BLOCK_D], dtype=tl.float32)

        for i in range(tl.cdiv(N, BLOCK_N)):
            n_mask = array < N
            mask = n_mask[:, None] & mask_d[None, :]

            do_cos = tl.load(do_cos_block_ptr, mask=mask, other=0).to(tl.float32)
            do_sin = tl.load(do_sin_block_ptr, mask=mask, other=0).to(tl.float32)

            theta = theta_ * (array[:, None] + OFFSET)
            dx = do_cos * tl.cos(theta) + do_sin * tl.sin(theta)

            x = tl.load(x_block_ptr, mask=mask, other=0).to(tl.float32)
            # for stable
            x_minus_max = x - x_max
            # softmax
            numerator = tl.exp(x_minus_max)
            o = numerator / denominator

            # scalar
            c += tl.sum(o * dx, axis=0)

            x_block_ptr += BLOCK_N * H * D
            array += BLOCK_N
            do_cos_block_ptr += BLOCK_N * H * 2 * D
            do_sin_block_ptr += BLOCK_N * H * 2 * D

        # reinit
        do_cos_block_ptr = (
            DO
            + offset_o
            + tl.arange(0, BLOCK_N)[:, None] * H * 2 * D
            + tl.arange(0, BLOCK_D)[None, :]
        )
        do_sin_block_ptr = (
            DO
            + offset_o
            + D
            + tl.arange(0, BLOCK_N)[:, None] * H * 2 * D
            + tl.arange(0, BLOCK_D)[None, :]
        )
        array = tl.arange(0, BLOCK_N)

    for i in range(tl.cdiv(N, BLOCK_N)):
        mask_n = array < N
        mask = mask_n[:, None] & mask_d[None, :]

        do_cos = tl.load(do_cos_block_ptr, mask=mask, other=0).to(tl.float32)
        do_sin = tl.load(do_sin_block_ptr, mask=mask, other=0).to(tl.float32)

        theta = theta_ * (array[:, None] + OFFSET)
        dx = do_cos * tl.cos(theta) + do_sin * tl.sin(theta)

        if ACT != "none":
            x_block_ptr = (
                X
                + offset_x
                + i * BLOCK_N * H * D
                + tl.arange(0, BLOCK_N)[:, None] * H * D
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

        dx_block_ptr += BLOCK_N * H * D
        array += BLOCK_N
        do_cos_block_ptr += BLOCK_N * H * 2 * D
        do_sin_block_ptr += BLOCK_N * H * 2 * D


class LrpeCosine1dCpTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, theta, offset=0, act="none", dim=None):
        o, x_stat1, x_stat2 = lrpe_cosine_1d_cp_fwd_triton(
            x=x, theta=theta, offset=offset, act=act, dim=dim
        )

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

        dx = lrpe_cosine_1d_cp_bwd_triton(
            x=x,
            theta=theta,
            do=do,
            x_stat1=x_stat1,
            x_stat2=x_stat2,
            offset=offset,
            act=act,
            dim=dim,
        )

        return dx, None, None, None, None


def lrpe_cosine_1d_cp_fwd_triton(x, theta, offset=0, act="none", dim=None, **kwargs):
    b, n, h, d = x.shape
    h_t, h_d = theta.shape
    # When h_d != d, we need to pad the theta with zeros, this makes the kernel much simpler
    if h_d != 1 and h_d != d:
        theta = F.pad(theta, (0, 0, 0, d - h_d))

    if h_t != 1 and h_t != h:  # H D
        theta_type = 1
    elif h_d == 1:  # H 1
        theta_type = 2
    else:  # D 1
        theta_type = 3

    # update shape
    h_t, h_d = theta.shape

    o = torch.empty(b, n, h, 2 * d, dtype=x.dtype, device=x.device)
    x_stat1 = torch.empty(b, h, d, dtype=x.dtype, device=x.device)
    x_stat2 = torch.empty(b, h, d, dtype=x.dtype, device=x.device)

    def grid(meta):
        return (b, h, triton.cdiv(d, meta["BLOCK_D"]))

    _lrpe_cosine_1d_cp_fwd_triton[grid](
        X=x,
        THETA=theta,
        O=o,
        X_STAT1=x_stat1,
        X_STAT2=x_stat2,
        OFFSET=offset,
        THETA_TYPE=theta_type,
        B=b,
        N=n,
        H=h,
        D=d,
        H_T=h_t,
        H_D=h_d,
        ACT=act,
    )

    return o, x_stat1, x_stat2


def lrpe_cosine_1d_cp_bwd_triton(
    x, theta, do, x_stat1, x_stat2, offset=0, act="none", dim=None, **kwargs
):
    b, n, h, d = x.shape
    h_t, h_d = theta.shape
    # When h_d != d, we need to pad the theta with zeros, this makes the kernel much simpler
    if h_d != 1 and h_d != d:
        theta = F.pad(theta, (0, 0, 0, d - h_d))

    if h_t != 1 and h_t != h:  # H D
        theta_type = 1
    elif h_d == 1:  # H 1
        theta_type = 2
    else:  # D 1
        theta_type = 3

    # update shape
    h_t, h_d = theta.shape

    dx = torch.empty_like(x)

    def grid(meta):
        return (b, h, triton.cdiv(d, meta["BLOCK_D"]))

    _lrpe_cosine_1d_cp_bwd_triton[grid](
        X=x,
        THETA=theta,
        DO=do,
        DX=dx,
        X_STAT1=x_stat1,
        X_STAT2=x_stat2,
        OFFSET=offset,
        THETA_TYPE=theta_type,
        B=b,
        N=n,
        H=h,
        D=d,
        H_T=h_t,
        H_D=h_d,
        ACT=act,
    )

    return dx


def lrpe_cosine_1d_cp_triton(x, theta, offset=0, act="none", dim=None, **kwargs):
    """
    Apply Lrpe Cosine 1d on the last dimension of x, loop over chunk.

    Args:
        x: Input tensor of shape (B, N, H, D)
        theta: Tensor of shape (H, E) or (H, 1) or (1, E)
        offset: Offset for the index
        act: Activation function before apply lrpe cosine
        dim: Dimension to apply the operation on

    Returns:
        output: Tensor of shape (B, N, H, 2 * D)
    """
    assert dim in [-3, None], "dim must in [-3, None]"
    return LrpeCosine1dCpTriton.apply(x, theta, offset, act, dim)


if __name__ == "__main__":
    # unit test
    b, n, h, d = 2, 128, 8, 64
    dtype = torch.float32
    device = torch.cuda.current_device()
    x = (torch.randn((b, n, h, d), dtype=dtype, device=device)).requires_grad_()
    theta = torch.randn((h, d), dtype=dtype, device=device)
    do = torch.randn((b, n, h, 2 * d), dtype=dtype, device=device)
    act = "silu"
    dim = -3

    o = lrpe_cosine_1d_cp_triton(x, theta, act=act, dim=dim)
    o.backward(do)
