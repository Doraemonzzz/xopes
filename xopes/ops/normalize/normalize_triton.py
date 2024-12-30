import torch
import triton
import triton.language as tl
from einops import rearrange

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16],
        }
    ),
    key=["G", "E"],
)
@triton.jit
def _normalize_fwd(
    X,  # B G E
    WEIGHT,  # G E
    BIAS,  # G E
    RESIDUAL,  # B G E
    MEAN,  # B G
    SIGMA,  # B G
    O,  # B G E
    C: tl.constexpr,
    EPS: tl.constexpr,
    USE_WEIGHT: tl.constexpr,
    USE_BIAS: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    USE_MEAN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    B: tl.constexpr,
    G: tl.constexpr,
    E: tl.constexpr,
    D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_g = tl.program_id(1)
    # compute offset
    offset_xro = off_b * G * E + off_g * E
    offset_wb = off_g * E
    offset_ms = off_b * G + off_g
    # mask
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    array_d < D
    mask_e = array_e < E
    # compute block ptr
    x_block_ptr = X + offset_xro + tl.arange(0, BLOCK_E)
    o_block_ptr = O + offset_xro + tl.arange(0, BLOCK_E)

    sigma_block_ptr = SIGMA + offset_ms

    # load and compute
    x = tl.load(x_block_ptr, mask=mask_e, other=0.0).to(tl.float32)

    if USE_RESIDUAL:
        r_block_ptr = RESIDUAL + offset_xro + tl.arange(0, BLOCK_E)
        r = tl.load(r_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        x = x + r

    if USE_MEAN:
        mean_block_ptr = MEAN + offset_ms
        mean = tl.sum(x, axis=0) / E
        tl.store(mean_block_ptr, mean.to(mean_block_ptr.dtype.element_ty))
        x = x - mean

    sigma = tl.sqrt(tl.sum(x * x, axis=0) + EPS)
    x = C * x / sigma

    if USE_WEIGHT:
        w_block_ptr = WEIGHT + offset_wb + tl.arange(0, BLOCK_E)
        weight = tl.load(w_block_ptr, mask=mask_e, other=1.0).to(tl.float32)
        x = x * weight

    if USE_BIAS:
        b_block_ptr = BIAS + offset_wb + tl.arange(0, BLOCK_E)
        bias = tl.load(b_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        x = x + bias

    tl.store(o_block_ptr, x.to(o_block_ptr.dtype.element_ty), mask=mask_e)
    tl.store(sigma_block_ptr, sigma.to(sigma_block_ptr.dtype.element_ty))


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16],
        }
    ),
    key=["G", "E"],
)
@triton.jit
def _normalize_bwd(
    X,  # B G E
    WEIGHT,  # G E
    BIAS,  # G E
    RESIDUAL,  # B G E
    MEAN,  # B G
    SIGMA,  # B G
    DX,  # B G E
    DW,  # B G E
    DB,  # B G E
    DO,  # B G E
    C: tl.constexpr,
    EPS: tl.constexpr,
    USE_WEIGHT: tl.constexpr,
    USE_BIAS: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    USE_MEAN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    B: tl.constexpr,
    G: tl.constexpr,
    E: tl.constexpr,
    D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_g = tl.program_id(1)
    # compute offset
    offset_xr = off_b * G * E + off_g * E
    offset_wb = off_g * E
    offset_ms = off_b * G + off_g
    # mask
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    array_d < D
    mask_e = array_e < E
    # compute block ptr
    x_block_ptr = X + offset_xr + tl.arange(0, BLOCK_E)
    sigma_block_ptr = SIGMA + offset_ms
    do_block_ptr = DO + offset_xr + tl.arange(0, BLOCK_E)
    dx_block_ptr = DX + offset_xr + tl.arange(0, BLOCK_E)

    # load and compute
    x = tl.load(x_block_ptr, mask=mask_e, other=0.0)
    do = tl.load(do_block_ptr, mask=mask_e, other=0.0)

    if USE_RESIDUAL:
        r_block_ptr = RESIDUAL + offset_xr + tl.arange(0, BLOCK_E)
        r = tl.load(r_block_ptr, mask=mask_e, other=0.0)
        x = x + r

    if USE_MEAN:
        mean_block_ptr = MEAN + offset_ms
        mean = tl.load(mean_block_ptr)
        x = x - mean

    sigma = tl.load(sigma_block_ptr)
    r = x / sigma
    dr = do * C

    if USE_WEIGHT:
        w_block_ptr = WEIGHT + offset_wb + tl.arange(0, BLOCK_E)
        dw_block_ptr = DW + offset_xr + tl.arange(0, BLOCK_E)
        dw = do * C * r
        tl.store(dw_block_ptr, dw.to(dw_block_ptr.dtype.element_ty), mask=mask_e)
        weight = tl.load(w_block_ptr, mask=mask_e, other=0.0)
        dr = dr * weight

    # if USE_BIAS:
    #     db_block_ptr = DB + offset_xr + tl.arange(0, BLOCK_E)
    #     db = do
    #     tl.store(db_block_ptr, db.to(db_block_ptr.dtype.element_ty), mask=mask_e)

    dx = 1 / sigma * (dr - r * tl.sum(r * dr, axis=0))

    if USE_MEAN:
        dx = dx - tl.sum(dx, axis=0) / E

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=mask_e)


class NormalizeTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        x,
        weight=None,
        bias=None,
        residual=None,
        c=1.0,
        eps=1e-5,
        use_mean=False,
        num_groups=1,
    ):
        use_weight = weight is not None
        use_bias = bias is not None
        use_residual = residual is not None

        # catch eps being too small if the tensors are fp16
        if x.dtype == torch.float16:
            eps = max(eps, 1.6e-5)

        # allocate output
        o = torch.empty_like(x).contiguous()

        # reshape input data into 2D tensor
        x_ = x.reshape(-1, x.shape[-1])
        b, d = x_.shape
        x_ = rearrange(x_, "... (g e) -> ... g e", g=num_groups).contiguous()
        e = x_.shape[-1]
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_E = min(MAX_FUSED_SIZE, triton.next_power_of_2(e))
        BLOCK_D = min(MAX_FUSED_SIZE, triton.next_power_of_2(d))
        if d > BLOCK_D:
            raise RuntimeError("Normalize doesn't support feature dim >= 64KB.")

        # allocate sigma, they'll be used in the backward pass
        sigma = torch.empty((b, num_groups), dtype=torch.float32, device=x.device)
        if use_mean:
            mean = torch.empty((b, num_groups), dtype=torch.float32, device=x.device)
        else:
            mean = None

        grid = (b, num_groups)
        _normalize_fwd[grid](
            X=x_,
            WEIGHT=weight,
            BIAS=bias,
            RESIDUAL=residual,
            MEAN=mean,
            SIGMA=sigma,
            O=o,
            C=c,
            EPS=eps,
            USE_WEIGHT=use_weight,
            USE_BIAS=use_bias,
            USE_RESIDUAL=use_residual,
            USE_MEAN=use_mean,
            BLOCK_D=BLOCK_D,
            BLOCK_E=BLOCK_E,
            B=b,
            G=num_groups,
            E=e,
            D=d,
        )

        ctx.save_for_backward(x, weight, bias, residual, mean, sigma)
        ctx.c = c
        ctx.eps = eps
        ctx.USE_MEAN = use_mean
        ctx.USE_WEIGHT = use_weight
        ctx.USE_BIAS = use_bias
        ctx.USE_RESIDUAL = use_residual
        ctx.BLOCK_E = BLOCK_E
        ctx.BLOCK_D = BLOCK_D
        ctx.num_groups = num_groups

        return o.reshape_as(x)

    @staticmethod
    @contiguous
    def backward(
        ctx,
        do,
        weight=None,
        bias=None,
        residual=None,
        c=1.0,
        eps=1e-5,
        use_mean=False,
        num_groups=1,
    ):
        x, weight, bias, residual, mean, sigma = ctx.saved_tensors
        c = ctx.c
        eps = ctx.eps
        use_mean = ctx.USE_MEAN
        use_weight = ctx.USE_WEIGHT
        use_bias = ctx.USE_BIAS
        use_residual = ctx.USE_RESIDUAL
        BLOCK_E = ctx.BLOCK_E
        BLOCK_D = ctx.BLOCK_D
        num_groups = ctx.num_groups

        # reshape input data into 2D tensor
        x_ = x.reshape(-1, x.shape[-1])
        do = do.reshape(-1, do.shape[-1]).contiguous()
        b, d = x_.shape
        x_ = rearrange(x_, "... (g e) -> ... g e", g=num_groups).contiguous()
        e = x_.shape[-1]

        dx = torch.empty_like(x).contiguous()

        if use_weight:
            dw = torch.empty((b, d), dtype=torch.float32, device=x.device).contiguous()
        else:
            dw = None

        if use_bias:
            db = torch.empty((b, d), dtype=torch.float32, device=x.device).contiguous()
        else:
            db = None

        grid = (b, num_groups)
        _normalize_bwd[grid](
            X=x_,
            WEIGHT=weight,
            BIAS=bias,
            RESIDUAL=residual,
            MEAN=mean,
            SIGMA=sigma,
            DX=dx,
            DW=dw,
            DB=db,
            DO=do,
            C=c,
            EPS=eps,
            USE_WEIGHT=use_weight,
            USE_BIAS=use_bias,
            USE_RESIDUAL=use_residual,
            USE_MEAN=use_mean,
            BLOCK_D=BLOCK_D,
            BLOCK_E=BLOCK_E,
            B=b,
            G=num_groups,
            E=e,
            D=d,
        )

        if use_residual:
            dr = dx
        else:
            dr = None

        if use_weight:
            dw = dw.sum(0)

        if use_bias:
            db = do.sum(0)

        return dx, dw, db, dr, None, None, None, None


def normalize_triton(
    x,
    weight=None,
    bias=None,
    residual=None,
    c=1.0,
    eps=1e-5,
    use_mean=False,
    num_groups=1,
):
    assert (
        x.shape[-1] % num_groups == 0
    ), "The last dimension of x must be divisible by num_groups"
    return NormalizeTriton.apply(
        x, weight, bias, residual, c, eps, use_mean, num_groups
    )


if __name__ == "__main__":
    b, d = 2, 512
    num_groups = 4
    dtype = torch.float32
    x = torch.randn((b, d), dtype=dtype).cuda()
    weight = torch.randn((num_groups), dtype=dtype).cuda()
    bias = torch.randn((num_groups), dtype=dtype).cuda()
    residual = torch.randn((b, d), dtype=dtype).cuda()
    o = normalize_triton(
        x, weight, bias, residual, c=1.0, eps=1e-5, use_mean=True, num_groups=num_groups
    )
    print(o.shape)
