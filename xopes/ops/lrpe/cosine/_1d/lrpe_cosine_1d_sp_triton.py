import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs, next_power_of_two


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8, 16, 32]}),
    key=["D", "E", "ACT"],
)
@triton.jit
def _lrpe_cosine_1d_sp_fwd_triton(
    X,  # B N H D
    Theta,  # H D / H / D
    O,  # B N H (D + E)
    OFFSET: tl.constexpr,
    THETA_TYPE: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ID: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    off_h = tl.program_id(2)
    # compute offset
    offset_x = off_b * N * H * D + off_n * H * D + off_h * D
    if THETA_TYPE == 1:  # H E
        offset_theta = off_h * E
    elif THETA_TYPE == 2:  # H 1
        offset_theta = off_h
    elif THETA_TYPE == 3:  # 1 E
        offset_theta = 0
    C = D + E
    offset_o = off_b * N * H * C + off_n * H * C + off_h * C
    # mask
    mask_e = tl.arange(0, BLOCK_E) < E
    mask_d = tl.arange(0, BLOCK_D) < D
    mask_id = tl.arange(0, BLOCK_ID) < D - E

    x_block_ptr = X + offset_x + tl.arange(0, BLOCK_D)
    theta_block_ptr = Theta + offset_theta + tl.arange(0, BLOCK_E)
    o_cos_block_ptr = O + offset_o + tl.arange(0, BLOCK_E)
    o_sin_block_ptr = O + offset_o + e + tl.arange(0, BLOCK_E)

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

    index_e = tl.arange(0, BLOCK_E)
    x_lrpe = tl.where(mask_e, tl.gather(x, index_e, axis=0), 0)

    theta = tl.load(theta_block_ptr, mask=mask_e, other=0).to(tl.float32) * (
        off_n + OFFSET
    )
    o_cos = x_lrpe * tl.cos(theta)
    o_sin = x_lrpe * tl.sin(theta)

    tl.store(o_cos_block_ptr, o_cos.to(o_cos_block_ptr.dtype.element_ty), mask=mask_d)
    tl.store(o_sin_block_ptr, o_sin.to(o_sin_block_ptr.dtype.element_ty), mask=mask_d)

    if E != D:
        index_d = E + tl.arange(0, BLOCK_ID)
        x_id = tl.where(mask_id, tl.gather(x, index_d, axis=0), 0)
        o_id_block_ptr = O + offset_o + 2 * e + tl.arange(0, BLOCK_ID)
        tl.store(o_id_block_ptr, x_id.to(o_id_block_ptr.dtype.element_ty), mask=mask_id)


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8]}),
    key=["D", "E", "ACT"],
)
@triton.jit
def _lrpe_cosine_1d_sp_bwd_triton(
    X,
    Theta,
    DO,
    DX,
    OFFSET: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ID: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    off_h = tl.program_id(2)
    # compute offset
    offset_x = off_b * N * H * D + off_n * H * D + off_h * D
    if THETA_TYPE == 1:  # H E
        offset_theta = off_h * E
    elif THETA_TYPE == 2:  # H 1
        offset_theta = off_h
    elif THETA_TYPE == 3:  # 1 E
        offset_theta = 0
    C = D + E
    offset_o = off_b * N * H * C + off_n * H * C + off_h * C
    # mask
    mask_e = tl.arange(0, BLOCK_E) < E
    tl.arange(0, BLOCK_D) < D
    mask_id = tl.arange(0, BLOCK_ID) < D - E

    theta_block_ptr = Theta + offset_theta + tl.arange(0, BLOCK_E)
    dx_block_ptr = DX + offset_x + tl.arange(0, BLOCK_D)
    do_cos_block_ptr = DO + offset_o + tl.arange(0, BLOCK_E)
    do_sin_block_ptr = DO + offset_o + e + tl.arange(0, BLOCK_E)

    do_cos = tl.load(do_cos_block_ptr, mask=mask_e, other=0).to(tl.float32)
    do_sin = tl.load(do_sin_block_ptr, mask=mask_e, other=0).to(tl.float32)
    theta = tl.load(theta_block_ptr, mask=mask_e, other=0).to(tl.float32) * (
        off_n + OFFSET
    )
    dx_lrpe = do_cos * tl.cos(theta) + do_sin * tl.sin(theta)

    if E != D:
        do_id_block_ptr = DO + offset_o + 2 * e + tl.arange(0, BLOCK_ID)
        dx_id = tl.load(do_id_block_ptr, mask=mask_id, other=0).to(tl.float32)

    if ACT != "none":
        if ACT == "softmax":
            -float("inf")
        else:
            pass

        # x_block_ptr = X + offset_x + tl.arange(0, BLOCK_D)
        # x = tl.load(x_block_ptr, mask=mask_d, other=value).to(tl.float32)
        # x_lrpe_block_ptr = X + offset_x + tl.arange(0, BLOCK_E)
        x_lrpe_block_ptr = X + offset_x + tl.arange(0, BLOCK_E)
        x_lrpe = tl.load(x_lrpe_block_ptr, mask=mask_e, other=0).to(tl.float32)

        if ACT == "relu":
            dx_lrpe = tl.where(x_lrpe >= 0, dx_lrpe, 0)
        elif ACT == "sigmoid":
            sigmoid = tl.sigmoid(x_lrpe)
            dx_lrpe = dx_lrpe * sigmoid * (1 - sigmoid)
        elif ACT == "silu":
            sigmoid = tl.sigmoid(x_lrpe)
            dx_lrpe = dx_lrpe * sigmoid * (1 + x_lrpe * (1 - sigmoid))
        elif ACT == "softmax":
            # for stable
            x_minus_max = x_lrpe - tl.max(x_lrpe, axis=0)
            # softmax
            numerator = tl.exp(x_minus_max)
            denominator = tl.sum(numerator)
            o = numerator / denominator

            # scalar
            c = tl.sum(o * dx_lrpe, axis=0)
            dx_lrpe = o * dx_lrpe - c * o

        if E != D:
            x_id_block_ptr = X + offset_x + 2 * e + tl.arange(0, BLOCK_ID)
            x_id = tl.load(x_id_block_ptr, mask=mask_id, other=0).to(tl.float32)

            if ACT == "relu":
                dx_id = tl.where(x_id >= 0, dx_id, 0)
            elif ACT == "sigmoid":
                sigmoid = tl.sigmoid(x_id)
                dx_id = dx_id * sigmoid * (1 - sigmoid)
            elif ACT == "silu":
                sigmoid = tl.sigmoid(x_id)
                dx_id = dx_id * sigmoid * (1 + x_id * (1 - sigmoid))

    do_id_block_ptr = DO + offset_o + 2 * e + tl.arange(0, BLOCK_ID)
    dx_id = tl.load(do_id_block_ptr, mask=mask_id, other=0).to(tl.float32)

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=d_mask)


class LrpeCosine1dSpTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, theta, offset=0, e=-1, act="none", dim=None):
        # TODO
        if e != -1:
            theta = torch.cat(
                [theta[..., :e], torch.zeros(h, d - e, device=theta.device)], dim=-1
            )

        o = lrpe_cosine_1d_sp_fwd_triton(
            x=x,
            theta=theta,
            offset=offset,
            e=e,
            act=act,
            dim=dim,
        )

        ctx.save_for_backward(x, theta)
        ctx.offset = offset
        ctx.e = e
        ctx.act = act
        ctx.dim = dim

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, theta = ctx.saved_tensors
        offset = ctx.offset
        e = ctx.e
        act = ctx.act
        dim = ctx.dim

        dx = lrpe_cosine_1d_sp_bwd_triton(x, theta, do, offset, e, act, dim)

        return dx, None, None, None, None


def lrpe_cosine_1d_sp_fwd_triton(
    x: torch.Tensor,
    theta: torch.Tensor,
    offset: int = 0,
    e: int = -1,
    act: str = "none",
    dim: int = None,
    **kwargs
):
    assert dim in [-1, None], "dim must in [-1, None]"

    b, n, h, d = x.shape
    h_t, h_d = theta.shape
    if h_t == h and h_d > 1:
        theta_type = 1
    elif h_d == 1:
        theta_type = 2
    else:
        theta_type = 3
    o = torch.empty(b, n, h, d + e, dtype=x.dtype, device=x.device)
    BLOCK_E = next_power_of_two(e)
    BLOCK_D = next_power_of_two(d)
    if d != e:
        BLOCK_ID = next_power_of_two(d - e)
    else:
        BLOCK_ID = 1

    def grid(meta):
        return (b, n, h)

    _lrpe_cosine_1d_sp_fwd_triton[grid](
        X=x,
        Theta=theta,
        O=o,
        OFFSET=offset,
        THETA_TYPE=theta_type,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        ACT=act,
        BLOCK_E=BLOCK_E,
        BLOCK_D=BLOCK_D,
        BLOCK_ID=BLOCK_ID,
    )

    return o


def lrpe_cosine_1d_sp_bwd_triton(
    x, theta, do, offset=0, start_dim=0, act="none", dim=None, **kwargs
):
    assert dim in [-1, None], "dim must in [-1, None]"

    b, h, n, d = x.shape
    h_t, h_d = theta.shape
    if h_t == h and h_d > 1:
        pass
    elif h_d == 1:
        pass
    else:
        pass
    dx = torch.empty_like(x)
    BLOCK_E = next_power_of_two(e)
    BLOCK_D = next_power_of_two(d)
    BLOCK_ID = next_power_of_two(d - e)

    def grid(meta):
        return (b, n, h)

    _lrpe_cosine_1d_sp_bwd_triton[grid](
        X=x,
        Theta=theta,
        DO=do,
        DX=dx,
        OFFSET=offset,
        B=b,
        H=h,
        N=n,
        D=d,
        E=e,
        ACT=act,
        BLOCK_E=BLOCK_E,
        BLOCK_D=BLOCK_D,
        BLOCK_ID=BLOCK_ID,
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
    Apply Lrpe Cosine 1d on the last dimension of x using Triton, parallel on the sequence dimension.

    Args:
        x: Input tensor of shape (B, N, H, D)
        theta: Tensor of shape (H, E) or (H, 1) or (1, E)
        offset: Offset for the index
        e: Number of elements to apply the operation on
        act: Activation function before apply lrpe cosine
        dim: Dimension to apply the operation on

    Returns:
        output: Tensor of shape (B, N, H, 2 * D)
    """
    assert dim in [-1, None], "dim must in [-1, None]"
    return LrpeCosine1dSpTriton.apply(x, theta, offset, e, act, dim)


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
