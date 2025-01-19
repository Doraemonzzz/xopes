from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
        }
    ),
    key=["B", "V"],
)
@triton.jit
def _ce_fwd_parallel(
    Z,  # B V
    Y,  # B
    LSE,  # B G
    S,  # B G
    ZK,  # B
    IGNORE_INDEX: tl.constexpr,
    LABEL_SMOOTHING: tl.constexpr,
    USE_LABEL_SMOOTHING: tl.constexpr,
    B: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_g = tl.program_id(1)
    # compute offset
    offset_z = off_b * V + off_g * BLOCK_V
    offset_ls = off_b * G + off_g
    offset_y = off_b
    # compute block ptr
    zy_block_ptr = Z + offset_z
    z_block_ptr = zy_block_ptr + tl.arange(0, BLOCK_V)
    y_block_ptr = (
        Y + offset_y
    )  # since we need to use y as a scalar, we don't need to use block_ptr
    lse_block_ptr = LSE + offset_ls + tl.arange(0, 1)
    if USE_LABEL_SMOOTHING:
        s_block_ptr = S + offset_ls + tl.arange(0, 1)
    zk_block_ptr = ZK + off_b
    array = tl.arange(0, BLOCK_V)
    # mask
    mask = (off_g * BLOCK_V + array) < V

    # get label
    y = tl.load(y_block_ptr)

    s = tl.full([1], 0, dtype=tl.float32)

    if y == IGNORE_INDEX:
        lse = tl.full([1], -float("inf"), dtype=tl.float32)
        tl.store(lse_block_ptr, lse.to(lse_block_ptr.dtype.element_ty))

        if USE_LABEL_SMOOTHING:
            tl.store(s_block_ptr, s.to(s_block_ptr.dtype.element_ty))

        if off_g == 0:
            zk = 0.0
            tl.store(zk_block_ptr, zk.to(zk_block_ptr.dtype.element_ty))
    else:
        z = tl.load(z_block_ptr, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(z)
        lse = tl.log(tl.sum(tl.exp(z - m), keep_dims=True)) + m
        if USE_LABEL_SMOOTHING:
            z_ = tl.where(mask, z, 0.0).to(z.dtype)
            s += tl.sum(z_)
            tl.store(s_block_ptr, s.to(s_block_ptr.dtype.element_ty))

        tl.store(lse_block_ptr, lse.to(lse_block_ptr.dtype.element_ty))

        if off_g == 0:
            y_offset = y
            zk = tl.load(zy_block_ptr + y_offset)
            tl.store(zk_block_ptr, zk.to(zk_block_ptr.dtype.element_ty))


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
        }
    ),
    key=["B", "V"],
)
@triton.jit
def _ce_fwd_reduce(
    O,  # B
    ZK,  # B
    LSE,  # B G
    S,  # B G
    LABEL_SMOOTHING: tl.constexpr,
    USE_LABEL_SMOOTHING: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    off_b = tl.program_id(0)
    # compute offset
    offset_ls = off_b * G
    offset_zo = off_b
    # compute block ptr
    lse_block_ptr = LSE + offset_ls + tl.arange(0, BLOCK_G)
    lse_output_block_ptr = LSE + offset_ls
    zk_block_ptr = ZK + offset_zo + tl.arange(0, 1)
    o_block_ptr = O + offset_zo + tl.arange(0, 1)
    # mask
    mask_g = tl.arange(0, BLOCK_G) < G
    # load
    lse = tl.load(lse_block_ptr, mask=mask_g, other=-float("inf")).to(tl.float32)
    zk = tl.load(zk_block_ptr).to(tl.float32)

    # compute global lse and sum
    # Important!!! Check whether lse is all -inf
    all_inf = tl.max(lse) == -float("inf")
    m = tl.max(lse)
    # If all -inf, set lse to 0
    if all_inf:
        lse_ = 0.0
    else:
        lse_ = tl.log(tl.sum(tl.exp(lse - m))) + m

    if USE_LABEL_SMOOTHING:
        s_block_ptr = S + offset_ls + tl.arange(0, BLOCK_G)
        s_ = tl.load(s_block_ptr, mask=mask_g, other=0.0).to(tl.float32)
        s = tl.sum(s_)
    else:
        s = 0.0

    o = (-(1 - LABEL_SMOOTHING) * zk + lse_ - (LABEL_SMOOTHING / V) * s) / N
    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty))
    tl.store(lse_output_block_ptr, lse_.to(lse_output_block_ptr.dtype.element_ty))


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_V": [1024, 2048, 4096, 8192, 16384, 32768, 65536],
        }
    ),
    key=["B", "V"],
)
@triton.jit
def _ce_bwd(
    Z,  # B V
    Y,  # B
    LSE,  # B
    DZ,  # B V
    IGNORE_INDEX: tl.constexpr,
    LABEL_SMOOTHING: tl.constexpr,
    USE_LABEL_SMOOTHING: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    V: tl.constexpr,
    # G: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    tl.cdiv(V, BLOCK_V)
    off_b = tl.program_id(0)
    off_g = tl.program_id(1)
    # compute offset
    offset_z = off_b * V
    offset_ylse = off_b
    # compute block ptr
    array = off_g * BLOCK_V + tl.arange(0, BLOCK_V)
    z_block_ptr = Z + offset_z + array
    y_block_ptr = Y + offset_ylse
    dz_block_ptr = DZ + offset_z + array
    lse_block_ptr = LSE + offset_ylse
    # mask
    mask = array < V

    z = tl.load(z_block_ptr, mask=mask, other=-float("inf"))
    y = tl.load(y_block_ptr)
    lse = tl.load(lse_block_ptr)
    p = tl.exp(z - lse)
    c = -LABEL_SMOOTHING / V
    dz = tl.where(array == y, -1 + LABEL_SMOOTHING + p + c, p + c) / N
    # When y is IGNORE_INDEX, set dz to 0
    dz = tl.where(y == IGNORE_INDEX, 0.0, dz)
    tl.store(dz_block_ptr, dz.to(dz_block_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK": [1024, 2048, 4096, 8192, 16384, 32768, 65536],
        }
    ),
    key=["B"],
)
@triton.jit
def _ewbo_mul_fwd(
    X,  # (B)
    Y,  # ()
    O,  # (B)
    B: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(0).to(tl.int64)  # avoid overflow

    # compute offset
    offset_b = off_b * BLOCK
    array = tl.arange(0, BLOCK)
    offset_xo = offset_b
    mask = (offset_b + array) < B

    # compute block ptr
    x_block_ptr = X + offset_xo + array
    y_block_ptr = Y
    o_block_ptr = O + offset_xo + array

    # Load data
    y = tl.load(y_block_ptr).to(tl.float32)
    x = tl.load(x_block_ptr, mask=mask, other=0).to(tl.float32)
    o = x * y

    # Store result
    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask)


def ewbo_mul_fwd(x, y):
    b = x.numel()

    def grid(meta):
        return (triton.cdiv(b, meta["BLOCK"]),)

    _ewbo_mul_fwd[grid](X=x, Y=y, O=x, B=b)


class LinearCrossEntropyTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        x,
        y,
        w,
        bias=None,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
    ):
        b, d = x.shape
        v, d = w.shape

        # Allocate output
        o = torch.empty((b,), dtype=torch.float32, device=x.device)
        # dx = torch.empty((b, d), dtype=torch.float32, device=x.device)
        dx = torch.empty_like(x)
        dw = torch.zeros((v, d), dtype=torch.float32, device=x.device)
        if bias is not None:
            db = torch.zeros((v,), dtype=torch.float32, device=x.device)
        else:
            db = None

        # Use at most B D memory size, so we should set NUM_CHUNKS to V / D
        if d < v:
            num_chunks = 1
            chunk_size = b
        else:
            num_chunks = min(8, triton.cdiv(v, d))
            chunk_size = triton.next_power_of_2(triton.cdiv(b, num_chunks))
            num_chunks = triton.cdiv(b, chunk_size)

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, b)

            xi = x[start:end]  # c d
            yi = y[start:end]  # c
            zi = F.linear(xi, w, bias)  # c v
            oi = o[start:end]  # c

            ##### start compute loss and gradients #####
            c, v = zi.shape

            if reduction == "mean":
                n = max(yi.ne(ignore_index).sum().item(), 1)  # avoid all IGNORE_INDEX
            else:
                n = 1

            # TODO: tune the parameters
            MAX_BLOCK_SIZE = 8192
            BLOCK_V = min(triton.next_power_of_2(v), MAX_BLOCK_SIZE)
            g = triton.cdiv(v, BLOCK_V)
            BLOCK_G = triton.next_power_of_2(g)
            lse = torch.empty((c, g), dtype=torch.float32, device=zi.device)
            zk = torch.empty((c), dtype=torch.float32, device=zi.device)
            use_label_smoothing = label_smoothing > 0
            if use_label_smoothing:
                s = torch.empty((c, g), dtype=torch.float32, device=zi.device)
            else:
                s = None

            grid = (c, g)

            _ce_fwd_parallel[grid](
                Z=zi,
                Y=yi,
                LSE=lse,
                S=s,
                ZK=zk,
                IGNORE_INDEX=ignore_index,
                LABEL_SMOOTHING=label_smoothing,
                USE_LABEL_SMOOTHING=use_label_smoothing,
                B=c,
                V=v,
                G=g,
                BLOCK_V=BLOCK_V,
            )

            grid = (c,)
            _ce_fwd_reduce[grid](
                O=oi,
                ZK=zk,
                LSE=lse,
                S=s,
                LABEL_SMOOTHING=label_smoothing,
                USE_LABEL_SMOOTHING=use_label_smoothing,
                N=n,
                B=c,
                V=v,
                G=g,
                BLOCK_G=BLOCK_G,
            )

            lse = lse[:, 0].contiguous()

            def grid(meta):
                return (c, triton.cdiv(v, meta["BLOCK_V"]))

            _ce_bwd[grid](
                Z=zi,
                Y=yi,
                LSE=lse,
                DZ=zi,  # use inplace operation to compute gradients
                IGNORE_INDEX=ignore_index,
                LABEL_SMOOTHING=label_smoothing,
                USE_LABEL_SMOOTHING=use_label_smoothing,
                N=n,
                B=c,
                V=v,
            )
            ##### end compute loss and gradients #####

            dx[start:end] = F.linear(zi, w.T)
            # dw += F.linear(zi.T, xi.T)
            dw += zi.T @ xi
            if bias is not None:
                db += zi.sum(dim=0)

        if reduction in ["mean", "sum"]:
            o = o.sum()

        ctx.save_for_backward(
            dx.to(x.dtype),
            dw.to(w.dtype),
            db.to(bias.dtype) if bias is not None else db,
        )

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        dx, dw, db = ctx.saved_tensors
        if torch.ne(do, torch.tensor(1.0, device=do.device)):
            # dx = dx * do
            # dw = dw * do
            # if db is not None:
            #     db = db * do
            ewbo_mul_fwd(dx, do)
            ewbo_mul_fwd(dw, do)
            if db is not None:
                ewbo_mul_fwd(db, do)

        return dx, None, dw, db, None, None, None


def linear_cross_entropy_triton(
    x: torch.Tensor,  # (B D)
    y: torch.Tensor,  # (B)
    W: torch.Tensor,  # (V D)
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Computes linear cross entropy using Triton.

    Args:
        x: Input tensor of shape (B, D)
        y: Target indices of shape (B)
        W: Weight matrix of shape (V, D)
        bias: Optional bias tensor of shape (V)
        ignore_index: Target indices to ignore
        reduction: Reduction method ("mean", "sum",)
        label_smoothing: Label smoothing factor

    Returns:
        Loss tensor
    """
    return LinearCrossEntropyTriton.apply(
        x, y, W, bias, ignore_index, reduction, label_smoothing
    )


if __name__ == "__main__":
    # Test code
    b, d, v = 2, 512, 1000
    dtype = torch.float32
    x = torch.randn((b, d), dtype=dtype).cuda()
    w = torch.randn((v, d), dtype=dtype).cuda()
    y = torch.randint(0, v, (b,)).cuda()
    o = linear_cross_entropy_triton(x, y, w)
    print(o.shape)
