import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import repeat

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
    if THETA_TYPE == 1:  # H D
        offset_theta = off_h * D
    elif THETA_TYPE == 2:  # 1 D
        offset_theta = 0
    offset_o = off_b * N * H * D + off_n * H * D + off_h * D

    # mask
    array = tl.arange(0, BLOCK_D)
    mask_d = array < D

    x_block_ptr = X + offset_x + tl.arange(0, BLOCK_D)
    theta_ptr = THETA + offset_theta + tl.arange(0, BLOCK_D)
    o_block_ptr = O + offset_o + tl.arange(0, BLOCK_D)

    # load values
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

    # load and compute rotation
    theta_cos = tl.load(theta_ptr, mask=mask_d, other=0).to(tl.float32) * (
        off_n + OFFSET
    )
    theta_sin = tl.where(array % 2 == 0, -theta_cos, theta_cos)
    x_cos = x
    # 0 1 2 3 4 5 (base)
    # 1 0 1 0 1 0 (+ 1 mod 2)
    # 2 0 2 0 2 0 (* 2)
    # 2 1 4 3 6 5 (add base)
    # 1 0 3 2 5 4 (-1)
    index = array + (array + 1) % 2 * 2 - 1
    x_sin = tl.permute(x, index)

    o = x_cos * tl.cos(theta_cos) + x_sin * tl.sin(theta_sin)

    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask_d)


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
    if THETA_TYPE == 1:  # H D
        offset_theta = off_h * D
    elif THETA_TYPE == 2:  # 1 D
        offset_theta = 0
    offset_o = off_b * N * H * D + off_n * H * D + off_h * D

    # mask
    array = tl.arange(0, BLOCK_D)
    mask_d = array < D

    THETA + offset_theta + tl.arange(0, BLOCK_D)
    dx_block_ptr = DX + offset_x + tl.arange(0, BLOCK_D)
    THETA + offset_theta + tl.arange(0, BLOCK_D)
    do_block_ptr = DO + offset_o + tl.arange(0, BLOCK_D)

    # load and compute rotation
    theta_cos = tl.load(theta_ptr, mask=mask_d, other=0).to(tl.float32) * (
        off_n + OFFSET
    )
    theta_sin = tl.where(array % 2 == 0, theta_cos, -theta_cos)
    do = tl.load(do_block_ptr, mask=mask_d, other=0).to(tl.float32)
    do_cos = do
    # 0 1 2 3 4 5 (base)
    # 1 0 1 0 1 0 (+ 1 mod 2)
    # 2 0 2 0 2 0 (* 2)
    # 2 1 4 3 6 5 (add base)
    # 1 0 3 2 5 4 (-1)
    index = array + (array + 1) % 2 * 2 - 1
    do_sin = tl.permute(do, index)

    dx = do_cos * tl.cos(theta_cos) + do_sin * tl.sin(theta_sin)

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

    # x1 x2 x3 -> x1 x1 x2 x2 x3 x3
    theta = repeat(x, "h d -> h (d g)", g=2)

    if h_t != 1 and d_t != 1:  # H D
        theta_type = 1
    else:  # 1 D
        theta_type = 2

    # update shape
    h_t, d_t = theta.shape

    o = torch.empty(b, n, h, d, dtype=x.dtype, device=x.device)
    BLOCK_D = next_power_of_two(d)

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

    # x1 x2 x3 -> x1 x1 x2 x2 x3 x3
    theta = repeat(x, "h d -> h (d g)", g=2)

    if h_t != 1 and d_t != 1:  # H D
        theta_type = 1
    else:  # 1 D
        theta_type = 2

    # update shape
    h_t, d_t = theta.shape

    dx = torch.empty_like(x)
    BLOCK_D = next_power_of_two(d)

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
        x = F.softmax(x, dim=dim)
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
