import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs, next_power_of_two


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8, 16, 32]}),
    key=["h", "n", "d"],
)
@triton.jit
def _lrpe_cosine_1d_sp_fwd_triton(
    X,  # B N H D
    Theta,  # H D / H / D
    O,  # B N H 2D
    OFFSET: tl.constexpr,
    START_DIM: tl.constexpr,
    THETA_TYPE: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_ID: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_n = tl.program_id(2)
    # compute offset
    offset_x = off_b * h * n * d + off_h * n * d + off_n * d
    if THETA_TYPE == 1:  # H D
        offset_theta = off_h * d
    elif THETA_TYPE == 2:  # H
        offset_theta = off_h
    elif THETA_TYPE == 3:  # D
        offset_theta = 0
    offset_o = off_b * h * n * 2 * d + off_h * n * 2 * d + off_n * 2 * d
    # mask
    tl.arange(0, BLOCK_D) < d
    mask_id = tl.arange(0, BLOCK_ID) < START_DIM

    X + offset_x + START_DIM + tl.arange(0, BLOCK_ID)
    X + offset_x + START_DIM + d + tl.arange(0, BLOCK_D)
    x_block_ptr = X + offset_x + tl.arange(0, BLOCK_D)
    theta_block_ptr = Theta + offset_theta + tl.arange(0, BLOCK_D)
    o_cos_block_ptr = O + offset_o + START_DIM + tl.arange(0, BLOCK_D)
    o_sin_block_ptr = O + offset_o + START_DIM + d + tl.arange(0, BLOCK_D)

    if ACT == "softmax":
        value = -float("inf")
    else:
        value = 0

    x = tl.load(x_block_ptr, mask=d_mask, other=value).to(tl.float32)
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

    theta = tl.load(theta_block_ptr, mask=d_mask, other=0).to(tl.float32) * (
        off_n + OFFSET
    )
    o_cos = x * tl.cos(theta)
    o_sin = x * tl.sin(theta)

    if START_DIM > 0:
        o_id_block_ptr = O + offset_o + tl.arange(0, BLOCK_ID)
        tl.where(mask_id, o_cos, o_sin)
        tl.store(
            o_id_block_ptr, o_cos.to(o_id_block_ptr.dtype.element_ty), mask=mask_id
        )

    tl.store(o_cos_block_ptr, o_cos.to(o_cos_block_ptr.dtype.element_ty), mask=d_mask)
    tl.store(o_sin_block_ptr, o_sin.to(o_sin_block_ptr.dtype.element_ty), mask=d_mask)


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8]}),
    key=["h", "n", "d"],
)
@triton.jit
def _lrpe_cosine_1d_sp_bwd_triton(
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
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_n = tl.program_id(2)
    # compute offset
    offset_x = off_b * h * n * d + off_h * n * d + off_n * d
    offset_theta = off_h * d
    offset_o = off_b * h * n * 2 * d + off_h * n * 2 * d + off_n * 2 * d
    # mask
    d_mask = tl.arange(0, BLOCK) < d

    theta_block_ptr = Theta + offset_theta + tl.arange(0, BLOCK)
    dx_block_ptr = DX + offset_x + tl.arange(0, BLOCK)
    do_cos_block_ptr = DO + offset_o + tl.arange(0, BLOCK)
    do_sin_block_ptr = DO + offset_o + d + tl.arange(0, BLOCK)

    do_cos = tl.load(do_cos_block_ptr, mask=d_mask, other=0).to(tl.float32)
    do_sin = tl.load(do_sin_block_ptr, mask=d_mask, other=0).to(tl.float32)

    theta = tl.load(theta_block_ptr, mask=d_mask, other=0).to(tl.float32) * (
        off_n + offset
    )
    dx = do_cos * tl.cos(theta) + do_sin * tl.sin(theta)

    if ACT != "none":
        if ACT == "softmax":
            value = -float("inf")
        else:
            value = 0

        x_block_ptr = X + offset_x + tl.arange(0, BLOCK)
        x = tl.load(x_block_ptr, mask=d_mask, other=value).to(tl.float32)

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

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=d_mask)


class LrpeCosine1dSpTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, theta, offset=0, start_dim=0, act="none", dim=None):
        o = lrpe_cosine_1d_sp_fwd_triton(
            x=x,
            theta=theta,
            offset=offset,
            start_dim=start_dim,
            act=act,
            dim=dim,
        )

        ctx.save_for_backward(x, theta)
        ctx.offset = offset
        ctx.start_dim = start_dim
        ctx.act = act
        ctx.dim = dim

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, theta = ctx.saved_tensors
        offset = ctx.offset
        start_dim = ctx.start_dim
        act = ctx.act
        dim = ctx.dim

        dx = lrpe_cosine_1d_sp_bwd_triton(x, theta, do, offset, start_dim, act, dim)

        return dx, None, None, None, None


def lrpe_cosine_1d_sp_fwd_triton(
    x: torch.Tensor,
    theta: torch.Tensor,
    offset: int = 0,
    start_dim: int = 0,
    act: str = "none",
    dim: int = None,
    **kwargs
):
    assert dim in [-1, None], "dim must in [-1, None]"

    b, n, h, d = x.shape
    o = torch.empty(b, n, h, 2 * d, dtype=x.dtype, device=x.device)
    BLOCK_ID = next_power_of_two(start_dim)
    BLOCK_D = next_power_of_two(d)

    def grid(meta):
        return (b, n, h)

    if len(theta.shape) == 2:
        theta_type = 1
    elif len(theta.shape) == 1:
        if theta.shape[0] == h:
            theta_type = 2
        elif theta.shape[0] == d:
            theta_type = 3

    _lrpe_cosine_1d_sp_fwd_triton[grid](
        X=x,
        Theta=theta,
        O=o,
        OFFSET=offset,
        START_DIM=start_dim,
        THETA_TYPE=theta_type,
        B=b,
        H=h,
        N=n,
        D=d,
        ACT=act,
        BLOCK_ID=BLOCK_ID,
        BLOCK_D=BLOCK_D,
    )

    return o


def lrpe_cosine_1d_sp_bwd_triton(
    x, theta, do, offset=0, start_dim=0, act="none", dim=None, **kwargs
):
    assert dim in [-1, None], "dim must in [-1, None]"

    b, h, n, d = x.shape
    dx = torch.empty_like(x)
    BLOCK = next_power_of_two(d)

    def grid(meta):
        return (b, h, n)

    _lrpe_cosine_1d_sp_bwd_triton[grid](
        x, theta, do, dx, offset, b, h, n, d, act, BLOCK
    )

    return dx


def lrpe_cosine_1d_sp_triton(
    x: torch.Tensor,
    theta: torch.Tensor,
    offset: int = 0,
    start_dim: int = 0,
    act: str = "none",
    dim: int = None,
    **kwargs
):
    """
    Apply Lrpe Cosine 1d on the last dimension of x using Triton, parallel on the sequence dimension.

    Args:
        x: Input tensor of shape (B, N, H, D)
        theta: Tensor of shape (H, D) or (H) or (D)
        offset: Offset for the index
        start_dim: Start dimension to apply the operation on
        act: Activation function before apply lrpe cosine
        dim: Dimension to apply the operation on

    Returns:
        output: Tensor of shape (B, N, H, start_dim + 2 * (D - start_dim))

    Examples:
        [:start_dim], [start_dim:d] -> [:start_dim], [start_dim:d] * cos, [start_dim:d] * sin
    """
    assert dim in [-1, None], "dim must in [-1, None]"
    return LrpeCosine1dSpTriton.apply(x, theta, offset, start_dim, act, dim)


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

    o = lrpe_cosine_1d_sp_triton(x, theta, act=act, dim=dim)
    o.backward(do)
