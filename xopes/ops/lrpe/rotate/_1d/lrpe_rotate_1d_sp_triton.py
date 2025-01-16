import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from xopes.ops.act import act_fn
from xopes.utils import contiguous, generate_configs, next_power_of_two


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8, 16, 32]}),
    key=["N", "H", "D", "ACT"],
)
@triton.jit
def _lrpe_rotate_1d_sp_fwd_triton(
    X,  # B N H D
    THETA,  # H D / 1 D
    O,  # B N H D
    OFFSET: tl.constexpr,
    THETA_TYPE: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    H_T: tl.constexpr,
    D_T: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    off_h = tl.program_id(2)

    # compute offset
    offset_x = off_b * N * H * D + off_n * H * D + off_h * D
    if THETA_TYPE == 1:  # H D/2
        offset_theta = off_h * D_T
    elif THETA_TYPE == 2:  # 1 D/2
        offset_theta = 0
    offset_o = off_b * N * H * D + off_n * H * D + off_h * D

    # mask
    array_d = tl.arange(0, BLOCK_D)
    mask_d1 = array_d < D_T
    mask_d2 = (D_T + array_d) < D

    x1_block_ptr = X + offset_x + tl.arange(0, BLOCK_D)
    x2_block_ptr = X + offset_x + D_T + tl.arange(0, BLOCK_D)
    o1_block_ptr = O + offset_o + tl.arange(0, BLOCK_D)
    o2_block_ptr = O + offset_o + D_T + tl.arange(0, BLOCK_D)
    theta_ptr = THETA + offset_theta + tl.arange(0, BLOCK_D)

    # load values
    if ACT == "softmax":
        value = -float("inf")
    else:
        value = 0

    x1 = tl.load(x1_block_ptr, mask=mask_d1, other=value).to(tl.float32)
    x2 = tl.load(x2_block_ptr, mask=mask_d2, other=value).to(tl.float32)

    if ACT != "none":
        if ACT == "relu":
            x1 = tl.where(x1 >= 0, x1, 0)
            x2 = tl.where(x2 >= 0, x2, 0)
        elif ACT == "sigmoid":
            x1 = tl.sigmoid(x1)
            x2 = tl.sigmoid(x2)
        elif ACT == "silu":
            x1 = x1 * tl.sigmoid(x1)
            x2 = x2 * tl.sigmoid(x2)
        elif ACT == "softmax":
            x1_max = tl.max(x1, axis=0)
            x2_max = tl.max(x2, axis=0)
            x_max = tl.maximum(x1_max, x2_max)
            # for stable
            x1_minus_max = x1 - x_max
            x2_minus_max = x2 - x_max
            # softmax
            numerator1 = tl.exp(x1_minus_max)
            numerator2 = tl.exp(x2_minus_max)
            denominator = tl.sum(numerator1) + tl.sum(numerator2)
            x1 = numerator1 / denominator
            x2 = numerator2 / denominator

    # load and compute rotation
    theta = tl.load(theta_ptr, mask=mask_d1, other=0).to(tl.float32) * (off_n + OFFSET)
    cos = tl.cos(theta)
    sin = tl.sin(theta)
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos

    tl.store(o1_block_ptr, o1.to(o1_block_ptr.dtype.element_ty), mask=mask_d1)
    tl.store(o2_block_ptr, o2.to(o2_block_ptr.dtype.element_ty), mask=mask_d2)


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8, 16, 32]}),
    key=["N", "H", "D", "ACT"],
)
@triton.jit
def _lrpe_rotate_1d_sp_bwd_triton(
    X,  # B N H D
    THETA,  # H D / H 1 / 1 D
    DO,  # B N H 2D
    DX,  # B N H D
    OFFSET: tl.constexpr,
    THETA_TYPE: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    H_T: tl.constexpr,
    D_T: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    off_h = tl.program_id(2)
    # compute offset
    offset_x = off_b * N * H * D + off_n * H * D + off_h * D
    if THETA_TYPE == 1:  # H D/2
        offset_theta = off_h * D_T
    elif THETA_TYPE == 2:  # 1 D/2
        offset_theta = 0
    offset_o = off_b * N * H * D + off_n * H * D + off_h * D

    # mask
    array_d = tl.arange(0, BLOCK_D)
    mask_d1 = array_d < D_T
    mask_d2 = (D_T + array_d) < D

    do1_block_ptr = DO + offset_o + tl.arange(0, BLOCK_D)
    do2_block_ptr = DO + offset_o + D_T + tl.arange(0, BLOCK_D)
    dx1_block_ptr = DX + offset_x + tl.arange(0, BLOCK_D)
    dx2_block_ptr = DX + offset_x + D_T + tl.arange(0, BLOCK_D)
    theta_block_ptr = THETA + offset_theta + tl.arange(0, BLOCK_D)

    # load and compute rotation
    theta = -tl.load(theta_block_ptr, mask=mask_d1, other=0).to(tl.float32) * (
        off_n + OFFSET
    )
    do1 = tl.load(do1_block_ptr, mask=mask_d1, other=0).to(tl.float32)
    do2 = tl.load(do2_block_ptr, mask=mask_d2, other=0).to(tl.float32)
    cos = tl.cos(theta)
    sin = tl.sin(theta)
    dx1 = do1 * cos - do2 * sin
    dx2 = do1 * sin + do2 * cos

    if ACT != "none":
        if ACT == "softmax":
            value = -float("inf")
        else:
            value = 0

        x1_block_ptr = X + offset_x + tl.arange(0, BLOCK_D)
        x2_block_ptr = X + offset_x + D_T + tl.arange(0, BLOCK_D)
        x1 = tl.load(x1_block_ptr, mask=mask_d1, other=value).to(tl.float32)
        x2 = tl.load(x2_block_ptr, mask=mask_d2, other=value).to(tl.float32)

        if ACT == "relu":
            dx1 = tl.where(x1 >= 0, dx1, 0)
            dx2 = tl.where(x2 >= 0, dx2, 0)
        elif ACT == "sigmoid":
            sigmoid1 = tl.sigmoid(x1)
            dx1 = dx1 * sigmoid1 * (1 - sigmoid1)
            sigmoid2 = tl.sigmoid(x2)
            dx2 = dx2 * sigmoid2 * (1 - sigmoid2)
        elif ACT == "silu":
            sigmoid1 = tl.sigmoid(x1)
            dx1 = dx1 * sigmoid1 * (1 + x1 * (1 - sigmoid1))
            sigmoid2 = tl.sigmoid(x2)
            dx2 = dx2 * sigmoid2 * (1 + x2 * (1 - sigmoid2))
        elif ACT == "softmax":
            x1_max = tl.max(x1, axis=0)
            x2_max = tl.max(x2, axis=0)
            x_max = tl.maximum(x1_max, x2_max)
            # for stable
            x1_minus_max = x1 - x_max
            x2_minus_max = x2 - x_max
            # softmax
            numerator1 = tl.exp(x1_minus_max)
            numerator2 = tl.exp(x2_minus_max)
            denominator = tl.sum(numerator1) + tl.sum(numerator2)
            o1 = numerator1 / denominator
            o2 = numerator2 / denominator

            # scalar
            c = tl.sum(o1 * dx1, axis=0) + tl.sum(o2 * dx2, axis=0)
            dx1 = o1 * dx1 - c * o1
            dx2 = o2 * dx2 - c * o2

    tl.store(dx1_block_ptr, dx1.to(dx1_block_ptr.dtype.element_ty), mask=mask_d1)
    tl.store(dx2_block_ptr, dx2.to(dx2_block_ptr.dtype.element_ty), mask=mask_d2)


class LrpeRotate1dSpTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, theta, offset=0, act="none", dim=None):
        has_head = len(x.shape) != 3
        if not has_head:  # b n d -> b n h d
            assert theta.shape[0] == 1, "theta must be (1, E)"
            x = x.unsqueeze(-2)

        o = lrpe_rotate_1d_sp_fwd_triton(
            x=x,
            theta=theta,
            offset=offset,
            act=act,
            dim=dim,
        )

        if not has_head:
            o = o.squeeze(-2)

        ctx.save_for_backward(x, theta)
        ctx.offset = offset
        ctx.act = act
        ctx.dim = dim
        ctx.has_head = has_head

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, theta = ctx.saved_tensors
        offset = ctx.offset
        act = ctx.act
        dim = ctx.dim
        has_head = ctx.has_head

        dx = lrpe_rotate_1d_sp_bwd_triton(
            x=x,
            theta=theta,
            do=do,
            offset=offset,
            act=act,
            dim=dim,
        )

        if not has_head:
            dx = dx.squeeze(-2)

        return dx, None, None, None, None


def lrpe_rotate_1d_sp_fwd_triton(
    x: torch.Tensor,
    theta: torch.Tensor,
    offset: int = 0,
    act: str = "none",
    dim: int = None,
    **kwargs
):
    b, n, h, d = x.shape
    h_t, d_t = theta.shape
    # When d_t != d // 2, we need to pad the theta with zeros, this makes the kernel much simpler
    if d_t != 1 and d_t != d // 2:
        theta = F.pad(theta, (0, 0, 0, d // 2 - d_t))

    if h_t != 1 and d_t != 1:  # H D/2
        theta_type = 1
    else:  # 1 D/2
        theta_type = 2

    # update shape
    h_t, d_t = theta.shape

    o = torch.empty(b, n, h, d, dtype=x.dtype, device=x.device)
    BLOCK_D = next_power_of_two(d_t)

    def grid(meta):
        return (b, n, h)

    _lrpe_rotate_1d_sp_fwd_triton[grid](
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
        D_T=d_t,
        ACT=act,
        BLOCK_D=BLOCK_D,
    )

    return o


def lrpe_rotate_1d_sp_bwd_triton(
    x, theta, do, offset=0, start_dim=0, act="none", dim=None, **kwargs
):
    b, n, h, d = x.shape
    h_t, d_t = theta.shape
    # When d_t != d // 2, we need to pad the theta with zeros, this makes the kernel much simpler
    if d_t != 1 and d_t != d // 2:
        theta = F.pad(theta, (0, 0, 0, d // 2 - d_t))

    if h_t != 1 and d_t != 1:  # H D/2
        theta_type = 1
    else:  # 1 D/2
        theta_type = 2

    # update shape
    h_t, d_t = theta.shape

    dx = torch.empty_like(x)
    BLOCK_D = next_power_of_two(d_t)

    def grid(meta):
        return (b, n, h)

    _lrpe_rotate_1d_sp_bwd_triton[grid](
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
        D_T=d_t,
        ACT=act,
        BLOCK_D=BLOCK_D,
    )

    return dx


def lrpe_rotate_1d_sp_triton(
    x: torch.Tensor,
    theta: torch.Tensor,
    offset: int = 0,
    act: str = "none",
    dim: int = None,
    **kwargs
) -> torch.Tensor:
    """
    Apply Lrpe Rotate (i.e. RoPE) 1d on the last dimension of x, parallel over sequence.

    Args:
        x: Input tensor of shape (B, N, H, D) or (B, N, D)
        theta: Tensor of shape (H, E) or (1, E), E <= D/2
        offset: Offset for the index
        act: Activation function before apply rotation
        dim: Dimension to apply the operation on, choose from [None, -1, 1]

    Returns:
        output: Tensor of shape (B, N, H, D)
    """
    assert dim in [None, -1, 1], "dim must in [None, -1, 1]"
    if act == "softmax" and dim == 1:  # softmax over sequence
        x = act_fn(x, act=act, dim=dim)
        # important: set act to none, because we dont need to apply softmax in kernel
        act = "none"
    return LrpeRotate1dSpTriton.apply(x, theta, offset, act, dim)


if __name__ == "__main__":
    # unit test
    b, n, h, d = 2, 128, 8, 64
    dtype = torch.float32
    device = torch.cuda.current_device()
    x = (torch.randn((b, n, h, d), dtype=dtype, device=device)).requires_grad_()
    theta = torch.randn((h, d // 2), dtype=dtype, device=device)
    do = torch.randn((b, n, h, d), dtype=dtype, device=device)

    o = lrpe_rotate_1d_sp_triton(x, theta, offset=0, act="none", dim=None)
    o.backward(do)
