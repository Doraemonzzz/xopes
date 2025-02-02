from typing import Optional

import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_B": [32, 64, 128],
            "BLOCK_D1": [32, 64, 128],
            "BLOCK_D2": [32, 64, 128],
        }
    ),
    key=["B", "D1", "D2"],
)
@triton.jit
def _gate_linear_fwd(
    X1,  # B D1
    X2,  # B D1
    WEIGHT,  # D2 D1
    BIAS,  # D2
    RESIDUAL,  # B D2
    O,  # B D2
    USE_BIAS: tl.constexpr,
    USE_RESIDUAL: tl.constexpr,
    B: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    ACT: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_d2 = tl.program_id(1)
    # compute offset
    offset_b = off_b * BLOCK_B
    offset_d2 = off_d2 * BLOCK_D2
    # compute block pointer
    array_b = offset_b + tl.arange(0, BLOCK_B)
    array_d1 = tl.arange(0, BLOCK_D1)
    array_d2 = offset_d2 + tl.arange(0, BLOCK_D2)
    # BLOCK_B, BLOCK_D1
    x1_block_ptr = X1 + array_b[:, None] * D1 + array_d1[None, :]
    x2_block_ptr = X2 + array_b[:, None] * D1 + array_d1[None, :]
    # BLOCK_D1, BLOCK_D2
    weight_block_ptr = WEIGHT + array_d2[None, :] * D1 + array_d1[:, None]
    if USE_BIAS:
        bias_block_ptr = BIAS + array_d2
    if USE_RESIDUAL:
        residual_block_ptr = RESIDUAL + array_b[:, None] * D2 + array_d2[None, :]
    o_block_ptr = O + array_b[:, None] * D2 + array_d2[None, :]

    mask_b = array_b < B
    mask_d2 = array_d2 < D2
    mask = mask_b[:, None] & mask_d2[None, :]
    acc = tl.zeros((BLOCK_B, BLOCK_D2), dtype=tl.float32)

    for i in range(tl.cdiv(D1, BLOCK_D1)):
        mask_d1 = array_d1 < D1
        x1 = tl.load(x1_block_ptr, mask=mask_b[:, None] & mask_d1[None, :], other=0.0)
        x2 = tl.load(x2_block_ptr, mask=mask_b[:, None] & mask_d1[None, :], other=0.0)
        weight = tl.load(
            weight_block_ptr, mask=mask_d1[:, None] & mask_d2[None, :], other=0.0
        )

        # act
        if ACT == "relu":
            x1 = tl.where(x1 >= 0, x1, 0)
        elif ACT == "sigmoid":
            x1 = tl.sigmoid(x1.to(tl.float32))
        elif ACT == "silu":
            x1 = x1 * tl.sigmoid(x1.to(tl.float32))

        y = x1.to(x2.dtype) * x2
        acc = tl.dot(y, weight, acc=acc)

        x1_block_ptr += BLOCK_D1
        x2_block_ptr += BLOCK_D1
        weight_block_ptr += BLOCK_D1
        array_d1 += BLOCK_D1

    if USE_BIAS:
        bias = tl.load(bias_block_ptr, mask=mask_d2, other=0.0)
        acc += bias

    if USE_RESIDUAL:
        residual = tl.load(residual_block_ptr, mask=mask, other=0.0)
        acc += residual

    tl.store(o_block_ptr, acc.to(o_block_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "num_stages": [2, 3, 4],
        }
    ),
    key=["B", "D"],
)
@triton.jit
def _gate_fn(
    X1,  # B D
    X2,  # B D
    O,  # B D
    B: tl.constexpr,
    D: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    # compute offset
    offset = off_b * D
    # mask
    array_d = tl.arange(0, BLOCK_D)
    mask_d = array_d < D
    mask = mask_d
    # compute block ptr
    x1_block_ptr = X1 + offset + array_d
    x2_block_ptr = X2 + offset + array_d
    o_block_ptr = O + offset + array_d

    x1 = tl.load(x1_block_ptr, mask=mask, other=0.0)
    x2 = tl.load(x2_block_ptr, mask=mask, other=0.0)

    if ACT == "relu":
        x1 = tl.where(x1 >= 0, x1, 0)
    elif ACT == "sigmoid":
        x1 = tl.sigmoid(x1.to(tl.float32))
    elif ACT == "silu":
        x1 = x1 * tl.sigmoid(x1.to(tl.float32))

    o = x1.to(x2.dtype) * x2

    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "num_stages": [2, 3, 4],
        }
    ),
    key=["B", "D"],
)
@triton.jit
def _gate_linear_bwd(
    X1,  # B D
    X2,  # B D
    DY,  # B D
    DX1,  # B D
    DX2,  # B D
    B: tl.constexpr,
    D: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    # compute offset
    offset = off_b * D
    # mask
    array_d = tl.arange(0, BLOCK_D)
    mask_d = array_d < D
    mask = mask_d
    # compute block ptr
    x1_block_ptr = X1 + offset + array_d
    x2_block_ptr = X2 + offset + array_d
    dy_block_ptr = DY + offset + array_d
    dx1_block_ptr = DX1 + offset + array_d
    dx2_block_ptr = DX2 + offset + array_d

    x1 = tl.load(x1_block_ptr, mask=mask, other=0.0)
    x2 = tl.load(x2_block_ptr, mask=mask, other=0.0)
    dy = tl.load(dy_block_ptr, mask=mask, other=0.0)

    if ACT == "relu":
        x1_ = tl.where(x1 >= 0, x1, 0)
    elif ACT == "sigmoid":
        x1_ = tl.sigmoid(x1.to(tl.float32))
    elif ACT == "silu":
        x1_ = x1 * tl.sigmoid(x1.to(tl.float32))
    else:
        x1_ = x1

    dx2 = x1_.to(dy.dtype) * dy
    dx1 = x2 * dy

    if ACT != "none":
        if ACT == "relu":
            dx1_ = tl.where(x1 >= 0, 1, 0)
        elif ACT == "sigmoid":
            sigmoid = tl.sigmoid(x1.to(tl.float32))
            dx1_ = sigmoid * (1 - sigmoid)
        elif ACT == "silu":
            sigmoid = tl.sigmoid(x1.to(tl.float32))
            dx1_ = sigmoid * (1 + x1 * (1 - sigmoid))
        dx1 = dx1 * dx1_.to(dx1.dtype)

    tl.store(dx1_block_ptr, dx1.to(dx1_block_ptr.dtype.element_ty), mask=mask)
    tl.store(dx2_block_ptr, dx2.to(dx2_block_ptr.dtype.element_ty), mask=mask)


class GateLinearTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x1, x2, weight, bias=None, residual=None, act="none"):
        # Prepare shapes and inputs
        x1_ = x1.reshape(-1, x1.shape[-1]).contiguous()
        x2_ = x2.reshape(-1, x2.shape[-1]).contiguous()
        b, d1 = x1_.shape
        d2 = weight.shape[0]
        use_bias = bias is not None
        use_residual = residual is not None
        output_shape = list(x1.shape[:-1]) + [d2]

        # Allocate output
        o = torch.empty(output_shape, device=x1.device, dtype=x1.dtype)

        # Launch kernel
        def grid(meta):
            return (triton.cdiv(b, meta["BLOCK_B"]), triton.cdiv(d2, meta["BLOCK_D2"]))

        _gate_linear_fwd[grid](
            X1=x1_,
            X2=x2_,
            WEIGHT=weight,
            BIAS=bias,
            RESIDUAL=residual,
            O=o,
            USE_BIAS=use_bias,
            USE_RESIDUAL=use_residual,
            B=b,
            D1=d1,
            D2=d2,
            ACT=act,
        )

        # Save for backward
        ctx.save_for_backward(x1, x2, weight, bias, residual)
        ctx.act = act

        return o.reshape(output_shape)

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x1, x2, weight, bias, residual = ctx.saved_tensors
        act = ctx.act

        # Prepare shapes and inputs
        x1_ = x1.reshape(-1, x1.shape[-1]).contiguous()
        x2_ = x2.reshape(-1, x2.shape[-1]).contiguous()
        b, d1 = x1_.shape
        d2 = weight.shape[0]
        use_bias = bias is not None
        use_residual = residual is not None
        output_shape = list(x1.shape[:-1]) + [d2]

        # Allocate output
        dx1 = torch.empty_like(x1_)
        dx2 = torch.empty_like(x2_)
        y = torch.empty_like(x1_)
        dw = torch.empty_like(weight)
        do_ = do.reshape(-1, do.shape[-1]).contiguous()
        if use_bias:
            db = do_.sum(dim=0)
        else:
            db = None

        if use_residual:
            dr = do
        else:
            dr = None

        # b d1
        # implement f(x1) * x2
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x1.element_size()
        BLOCK_D = min(MAX_FUSED_SIZE, triton.next_power_of_2(d1))
        if d1 > BLOCK_D:
            raise RuntimeError("Normalize doesn't support feature dim >= 64KB.")

        def grid(meta):
            return (triton.cdiv(b, meta["BLOCK_B"]),)

        grid = (b,)
        _gate_fn[grid](
            X1=x1_,
            X2=x2_,
            O=y,
            B=b,
            D=d1,
            ACT=act,
            BLOCK_D=BLOCK_D,
        )

        @torch.compile
        def f(do_, y, weight):
            # b d2, b d1 -> d2 d1
            dw = torch.matmul(do_.T, y)
            # b d2, d2 d1 -> b d2
            dy = torch.matmul(do_, weight.to(y.dtype))

            return dw, dy

        dw, dy = f(do_, y, weight)

        _gate_linear_bwd[grid](
            X1=x1_,
            X2=x2_,
            DY=dy,
            DX1=dx1,
            DX2=dx2,
            B=b,
            D=d1,
            ACT=act,
            BLOCK_D=BLOCK_D,
        )

        return dx1.reshape_as(x1), dx2.reshape_as(x2), dw, db, dr, None, None


def gate_linear_triton(
    x1: torch.Tensor,
    x2: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    act: str = "none",
) -> torch.Tensor:
    """
    Apply gate linear using Triton.

    Args:
        x1: Input tensor, ... d1
        x2: Input tensor, ... d1
        weight: Weight tensor, d2 d1
        bias: Bias tensor, d2
        residual: Residual tensor, ... d2
        act: Activation function

    Returns:
        Output tensor, ... d2
    """
    return GateLinearTriton.apply(x1, x2, weight, bias, residual, act)


if __name__ == "__main__":
    b, d1, d2 = 4, 64, 32
    device = "cuda"
    x1 = torch.randn(b, d1).to(device).requires_grad_(True)
    x2 = torch.randn(b, d1).to(device).requires_grad_(True)
    weight = torch.randn(d2, d1).to(device).requires_grad_(True)
    bias = torch.randn(d2).to(device).requires_grad_(True)
    residual = torch.randn(b, d2).to(device).requires_grad_(True)
    act = "relu"
    o = gate_linear_triton(x1, x2, weight, bias, residual, act)
    o.sum().backward()
