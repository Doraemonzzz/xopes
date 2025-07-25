import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from xopes.ops.act import act_fn
from xopes.utils import contiguous, generate_configs, next_power_of_two


@triton.autotune(
    generate_configs(
        {
            "num_warps": [2, 4, 8, 16, 32],
            "BLOCK_N": [4, 8, 16, 32, 64, 128],
        }
    ),
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
    D_T: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    off_h = tl.program_id(2)

    # compute offset
    offset_n = off_n * BLOCK_N
    array_n = offset_n + tl.arange(0, BLOCK_N)
    offset_x = off_b * N * H * D + array_n[:, None] * H * D + off_h * D
    if THETA_TYPE == 1:  # H D
        offset_theta = off_h * D
    elif THETA_TYPE == 2:  # H 1
        offset_theta = off_h
    elif THETA_TYPE == 3:  # 1 D
        offset_theta = 0
    C = 2 * D
    offset_o = off_b * N * H * C + array_n[:, None] * H * C + off_h * C

    # mask
    mask_n = array_n < N
    mask_d = tl.arange(0, BLOCK_D) < D
    mask = mask_n[:, None] & mask_d[None, :]

    x_block_ptr = X + offset_x + tl.arange(0, BLOCK_D)[None, :]
    if D_T != 1:
        theta_block_ptr = THETA + offset_theta + tl.arange(0, BLOCK_D)[None, :]
    else:  # scalar version
        theta_block_ptr = THETA + offset_theta + tl.arange(0, 1)[None, :]
    o_cos_block_ptr = O + offset_o + tl.arange(0, BLOCK_D)[None, :]
    o_sin_block_ptr = O + offset_o + D + tl.arange(0, BLOCK_D)[None, :]

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
            x_minus_max = x - tl.max(x, axis=-1, keep_dims=True)
            # softmax
            numerator = tl.exp(x_minus_max)
            denominator = tl.sum(numerator, axis=-1, keep_dims=True)
            x = numerator / denominator

    if D_T != 1:
        theta = tl.load(theta_block_ptr, mask=mask_d[None, :], other=0).to(
            tl.float32
        ) * (OFFSET + array_n[:, None])
    else:
        theta = tl.load(theta_block_ptr).to(tl.float32) * (OFFSET + array_n[:, None])
    o_cos = x * tl.cos(theta)
    o_sin = x * tl.sin(theta)

    tl.store(o_cos_block_ptr, o_cos.to(o_cos_block_ptr.dtype.element_ty), mask=mask)
    tl.store(o_sin_block_ptr, o_sin.to(o_sin_block_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    generate_configs(
        {
            "num_warps": [2, 4, 8, 16, 32],
            "BLOCK_N": [4, 8, 16, 32, 64, 128],
        }
    ),
    key=["N", "H", "D", "ACT"],
)
@triton.jit
def _lrpe_cosine_1d_sp_bwd_triton(
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
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    off_h = tl.program_id(2)

    # compute offset
    offset_n = off_n * BLOCK_N
    array_n = offset_n + tl.arange(0, BLOCK_N)
    offset_x = off_b * N * H * D + array_n[:, None] * H * D + off_h * D
    if THETA_TYPE == 1:  # H D
        offset_theta = off_h * D
    elif THETA_TYPE == 2:  # H 1
        offset_theta = off_h
    elif THETA_TYPE == 3:  # 1 D
        offset_theta = 0
    C = 2 * D
    offset_o = off_b * N * H * C + array_n[:, None] * H * C + off_h * C

    # mask
    mask_n = array_n < N
    mask_d = tl.arange(0, BLOCK_D) < D
    mask = mask_n[:, None] & mask_d[None, :]

    theta_block_ptr = THETA + offset_theta + tl.arange(0, BLOCK_D)[None, :]
    dx_block_ptr = DX + offset_x + tl.arange(0, BLOCK_D)[None, :]
    if D_T != 1:
        theta_block_ptr = THETA + offset_theta + tl.arange(0, BLOCK_D)[None, :]
    else:  # scalar version
        theta_block_ptr = THETA + offset_theta + tl.arange(0, 1)[None, :]
    do_cos_block_ptr = DO + offset_o + tl.arange(0, BLOCK_D)[None, :]
    do_sin_block_ptr = DO + offset_o + D + tl.arange(0, BLOCK_D)[None, :]

    # load
    do_cos = tl.load(do_cos_block_ptr, mask=mask, other=0).to(tl.float32)
    do_sin = tl.load(do_sin_block_ptr, mask=mask, other=0).to(tl.float32)
    if D_T != 1:
        theta = tl.load(theta_block_ptr, mask=mask_d[None, :], other=0).to(
            tl.float32
        ) * (OFFSET + array_n[:, None])
    else:
        theta = tl.load(theta_block_ptr).to(tl.float32) * (OFFSET + array_n[:, None])
    dx = do_cos * tl.cos(theta) + do_sin * tl.sin(theta)

    if ACT != "none":
        if ACT == "softmax":
            value = -float("inf")
        else:
            value = 0

        x_block_ptr = X + offset_x + tl.arange(0, BLOCK_D)[None, :]
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
            x_minus_max = x - tl.max(x, axis=-1, keep_dims=True)
            # softmax
            numerator = tl.exp(x_minus_max)
            denominator = tl.sum(numerator, axis=-1, keep_dims=True)
            o = numerator / denominator

            # scalar
            c = tl.sum(o * dx, axis=-1, keep_dims=True)
            dx = o * dx - c * o

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=mask)


class LrpeCosine1dSpTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, theta, offset=0, act="none", dim=None):
        has_head = len(x.shape) != 3
        if not has_head:  # b n d -> b n h d
            assert theta.shape[0] == 1, "theta must be (1, E)"
            x = x.unsqueeze(-2)
        shape = x.shape

        o = lrpe_cosine_1d_sp_fwd_triton(
            x=x,
            theta=theta,
            offset=offset,
            act=act,
            dim=dim,
        )

        if not has_head:  # b n d -> b n h d
            o = o.squeeze(-2)

        if act in [
            "none",
        ]:
            x = None

        ctx.save_for_backward(x, theta)
        ctx.offset = offset
        ctx.act = act
        ctx.dim = dim
        ctx.has_head = has_head
        ctx.shape = shape

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, theta = ctx.saved_tensors
        offset = ctx.offset
        act = ctx.act
        dim = ctx.dim
        has_head = ctx.has_head
        shape = ctx.shape

        dx = lrpe_cosine_1d_sp_bwd_triton(
            x=x,
            theta=theta,
            do=do,
            offset=offset,
            act=act,
            dim=dim,
            shape=shape,
        )

        if not has_head:
            dx = dx.squeeze(-2)

        return dx, None, None, None, None


def lrpe_cosine_1d_sp_fwd_triton(
    x: torch.Tensor,
    theta: torch.Tensor,
    offset: int = 0,
    act: str = "none",
    dim: int = None,
    **kwargs
):
    b, n, h, d = x.shape
    h_t, d_t = theta.shape
    # When d_t != d, we need to pad the theta with zeros, this makes the kernel much simpler
    if d_t != 1 and d_t != d:
        theta = F.pad(theta, (0, 0, 0, d - d_t))

    if h_t != 1 and d_t != 1:  # H D
        theta_type = 1
    elif d_t == 1:  # H 1
        theta_type = 2
    else:  # 1 D
        theta_type = 3

    # update shape
    h_t, d_t = theta.shape

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
        D_T=d_t,
        ACT=act,
        BLOCK_D=BLOCK_D,
    )

    return o


def lrpe_cosine_1d_sp_bwd_triton(
    x, theta, do, offset=0, start_dim=0, act="none", dim=None, shape=None, **kwargs
):
    b, n, h, d = shape
    h_t, d_t = theta.shape
    # When d_t != d, we need to pad the theta with zeros, this makes the kernel much simpler
    if d_t != 1 and d_t != d:
        theta = F.pad(theta, (0, 0, 0, d - d_t))

    if h_t != 1 and d_t != 1:  # H D
        theta_type = 1
    elif d_t == 1:  # H 1
        theta_type = 2
    else:  # D 1
        theta_type = 3

    # update shape
    h_t, d_t = theta.shape

    dx = torch.empty((b, n, h, d), dtype=do.dtype, device=do.device)
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
        D_T=d_t,
        ACT=act,
        BLOCK_D=BLOCK_D,
    )

    return dx


def lrpe_cosine_1d_sp_triton(
    x: torch.Tensor,
    theta: torch.Tensor,
    offset: int = 0,
    act: str = "none",
    dim: int = None,
    **kwargs
):
    """
    Apply Lrpe Cosine 1d on the last dimension of x, parallel over sequence.

    Args:
        x: Input tensor of shape (B, N, H, D) or (B, N, D)
        theta: Tensor of shape (H, E) or (H, 1) or (1, E)
        offset: Offset for the index
        act: Activation function before apply lrpe cosine
        dim: Dimension to apply the operation on, choose from [None, -1, 1]

    Returns:
        output: Tensor of shape (B, N, H, 2 * D)
    """
    assert dim in [None, -1, 1], "dim must in [None, -1, 1]"
    if act == "softmax" and dim == 1:  # softmax over sequence
        x = act_fn(x, act=act, dim=dim)
        # important: set act to none, because we dont need to apply softmax in kernel
        act = "none"
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
