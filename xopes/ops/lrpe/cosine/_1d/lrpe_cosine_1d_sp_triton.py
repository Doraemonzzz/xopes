import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs, next_power_of_two


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8, 16, 32]}),
    key=["N", "H", "D", "ACT"],
)
@triton.jit
def _lrpe_cosine_1d_sp_fwd_triton(
    X,  # B N H D
    THETA,  # H D / H 1 / 1 D
    O,  # B N H 2D
    OFFSET: tl.constexpr,
    THETA_TYPE: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    H_T: tl.constexpr,
    H_D: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    off_h = tl.program_id(2)
    # compute offset
    offset_x = off_b * N * H * D + off_n * H * D + off_h * D
    if THETA_TYPE == 1:  # H D
        offset_theta = off_h * D
    elif THETA_TYPE == 2:  # H 1
        offset_theta = off_h
    elif THETA_TYPE == 3:  # 1 D
        offset_theta = 0
    C = 2 * D
    offset_o = off_b * N * H * C + off_n * H * C + off_h * C
    # mask
    mask_d = tl.arange(0, BLOCK_D) < D

    x_block_ptr = X + offset_x + tl.arange(0, BLOCK_D)
    if H_D != 1:
        theta_block_ptr = THETA + offset_theta + tl.arange(0, BLOCK_D)
    else:  # scalar version
        theta_block_ptr = THETA + offset_theta + tl.arange(0, 1)
    o_cos_block_ptr = O + offset_o + tl.arange(0, BLOCK_D)
    o_sin_block_ptr = O + offset_o + D + tl.arange(0, BLOCK_D)

    if ACT == "softmax":
        value = -float("inf")
    else:
        value = 0

    x = tl.load(x_block_ptr, mask=mask_d, other=value).to(tl.float32)

    if ACT != "none":
        if ACT == "relu":
            x = tl.where(x >= 0, x, 0)
        elif ACT == "sigmoid":
            x = tl.sigmoid(x)
        elif ACT == "silu":
            x = x * tl.sigmoid(x)
        elif ACT == "softmax":
            x_max = tl.max(x, axis=0)
            # for stable
            x_minus_max = x - x_max
            # softmax
            numerator = tl.exp(x_minus_max)
            denominator = tl.sum(numerator)
            x = numerator / denominator

    if H_D != 1:
        theta = tl.load(theta_block_ptr, mask=mask_d, other=0).to(tl.float32) * (
            off_n + OFFSET
        )
    else:
        theta = tl.load(theta_block_ptr).to(tl.float32) * (off_n + OFFSET)
    o_cos = x * tl.cos(theta)
    o_sin = x * tl.sin(theta)

    tl.store(o_cos_block_ptr, o_cos.to(o_cos_block_ptr.dtype.element_ty), mask=mask_d)
    tl.store(o_sin_block_ptr, o_sin.to(o_sin_block_ptr.dtype.element_ty), mask=mask_d)


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8, 16, 32]}),
    key=["N", "H", "D", "ACT"],
)
@triton.jit
def _lrpe_cosine_1d_sp_bwd_triton(
    X,
    THETA,
    DO,
    DX,
    OFFSET: tl.constexpr,
    THETA_TYPE: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    H_T: tl.constexpr,
    H_D: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    off_h = tl.program_id(2)
    # compute offset
    offset_x = off_b * N * H * D + off_n * H * D + off_h * D
    if THETA_TYPE == 1:  # H D
        offset_theta = off_h * D
    elif THETA_TYPE == 2:  # H 1
        offset_theta = off_h
    elif THETA_TYPE == 3:  # 1 D
        offset_theta = 0
    C = 2 * D
    offset_o = off_b * N * H * C + off_n * H * C + off_h * C
    # mask
    mask_d = tl.arange(0, BLOCK_D) < D

    theta_block_ptr = THETA + offset_theta + tl.arange(0, BLOCK_D)
    dx_block_ptr = DX + offset_x + tl.arange(0, BLOCK_D)
    if H_D != 1:
        theta_block_ptr = THETA + offset_theta + tl.arange(0, BLOCK_D)
    else:  # scalar version
        theta_block_ptr = THETA + offset_theta + tl.arange(0, 1)
    do_cos_block_ptr = DO + offset_o + tl.arange(0, BLOCK_D)
    do_sin_block_ptr = DO + offset_o + D + tl.arange(0, BLOCK_D)

    # load
    do_cos = tl.load(do_cos_block_ptr, mask=mask_d, other=0).to(tl.float32)
    do_sin = tl.load(do_sin_block_ptr, mask=mask_d, other=0).to(tl.float32)
    if H_D != 1:
        theta = tl.load(theta_block_ptr, mask=mask_d, other=0).to(tl.float32) * (
            off_n + OFFSET
        )
    else:
        theta = tl.load(theta_block_ptr).to(tl.float32) * (off_n + OFFSET)
    dx = do_cos * tl.cos(theta) + do_sin * tl.sin(theta)

    if ACT != "none":
        if ACT == "softmax":
            value = -float("inf")
        else:
            value = 0

        x_block_ptr = X + offset_x + tl.arange(0, BLOCK_D)
        x = tl.load(x_block_ptr, mask=mask_d, other=value).to(tl.float32)

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

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=mask_d)


class LrpeCosine1dSpTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, theta, offset=0, act="none", dim=None):
        o = lrpe_cosine_1d_sp_fwd_triton(
            x=x,
            theta=theta,
            offset=offset,
            act=act,
            dim=dim,
        )

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

        dx = lrpe_cosine_1d_sp_bwd_triton(
            x=x,
            theta=theta,
            do=do,
            offset=offset,
            act=act,
            dim=dim,
        )

        return dx, None, None, None, None


def lrpe_cosine_1d_sp_fwd_triton(
    x: torch.Tensor,
    theta: torch.Tensor,
    offset: int = 0,
    act: str = "none",
    dim: int = None,
    **kwargs
):
    assert dim in [-1, None], "dim must in [-1, None]"

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
    BLOCK_D = next_power_of_two(d)

    def grid(meta):
        return (b, n, h)

    _lrpe_cosine_1d_sp_fwd_triton[grid](
        X=x,
        THETA=theta,
        O=o,
        OFFSET=offset,
        THETA_TYPE=theta_type,
        B=b,
        N=n,
        H=h,
        D=d,
        H_T=h_t,
        H_D=h_d,
        ACT=act,
        BLOCK_D=BLOCK_D,
    )

    return o


def lrpe_cosine_1d_sp_bwd_triton(
    x, theta, do, offset=0, start_dim=0, act="none", dim=None, **kwargs
):
    assert dim in [-1, None], "dim must in [-1, None]"

    b, h, n, d = x.shape
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
    BLOCK_D = next_power_of_two(d)

    def grid(meta):
        return (b, n, h)

    _lrpe_cosine_1d_sp_bwd_triton[grid](
        X=x,
        THETA=theta,
        DO=do,
        DX=dx,
        OFFSET=offset,
        THETA_TYPE=theta_type,
        B=b,
        H=h,
        N=n,
        D=d,
        H_T=h_t,
        H_D=h_d,
        ACT=act,
        BLOCK_D=BLOCK_D,
    )

    return dx


def lrpe_cosine_1d_sp_triton(
    x: torch.Tensor,
    theta: torch.Tensor,
    offset: int = 0,
    e: int = -1,
    act: str = "none",
    dim: int = None,
    **kwargs
):
    """
    Apply Lrpe Cosine 1d on the last dimension of x, parallel over sequence.

    Args:
        x: Input tensor of shape (B, N, H, D)
        theta: Tensor of shape (H, E) or (H, 1) or (1, E)
        offset: Offset for the index
        act: Activation function before apply lrpe cosine
        dim: Dimension to apply the operation on

    Returns:
        output: Tensor of shape (B, N, H, 2 * D)
    """
    assert dim in [-1, None], "dim must in [-1, None]"
    return LrpeCosine1dSpTriton.apply(x, theta, offset, act, dim)


if __name__ == "__main__":
    # unit test
    b, n, h, d = 2, 128, 8, 64
    dtype = torch.float32
    device = torch.cuda.current_device()
    x = (torch.randn((b, n, h, d), dtype=dtype, device=device)).requires_grad_()
    theta = torch.randn((h, d), dtype=dtype, device=device)
    do = torch.randn((b, n, h, 2 * d), dtype=dtype, device=device)
    act = "silu"
    dim = -1

    o = lrpe_cosine_1d_sp_triton(x, theta, act=act, dim=dim)
    o.backward(do)
