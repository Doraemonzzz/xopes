import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs, is_dim_valid, is_op_valid


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
            "BLOCK_B1": [16, 32, 64, 128],
        }
    ),
    key=["B1", "B2"],
)
@triton.jit
def _ewbo_fwd(
    X,  # Shape: (B1, B2)
    Y,  # Shape: (B1)
    O,  # Output tensor
    OP: tl.constexpr,  # Operation type
    INPLACE: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    BLOCK_B1: tl.constexpr,
    BLOCK_B2: tl.constexpr,
):
    off_b1 = tl.program_id(0)

    # compute offset
    array_b1 = tl.arange(0, BLOCK_B1)
    array_b2 = tl.arange(0, BLOCK_B2)
    offset_b1 = off_b1 * BLOCK_B1
    offset_xo = offset_b1 * B2
    offset_y = offset_b1

    # compute block ptr
    x_block_ptr = X + offset_xo + array_b1[:, None] * B2 + array_b2[None, :]
    y_block_ptr = Y + offset_y + array_b1[:, None]
    if INPLACE:
        o_block_ptr = x_block_ptr
    else:
        o_block_ptr = O + offset_xo + array_b1[:, None] * B2 + array_b2[None, :]

    # mask
    mask_b1 = (off_b1 * BLOCK_B1 + array_b1) < B1

    y = tl.load(y_block_ptr, mask=mask_b1[:, None], other=0)
    NUM_BLOCKS_B2 = tl.cdiv(B2, BLOCK_B2)
    for i in range(NUM_BLOCKS_B2):
        # mask
        mask_b2 = array_b2 < B2
        mask = mask_b1[:, None] & mask_b2[None, :]

        # Load data
        x = tl.load(x_block_ptr, mask=mask, other=0)

        # Perform operation
        if OP == "add":  # add
            o = x + y
        elif OP == "mul":  # mul
            o = x * y
        elif OP == "sub":  # sub
            o = x - y
        elif OP == "div":  # div
            o = x / y

        # Store result
        tl.store(o_block_ptr, o.to(X.dtype.element_ty), mask=mask)

        array_b2 += BLOCK_B2
        x_block_ptr += BLOCK_B2
        o_block_ptr += BLOCK_B2


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
            "BLOCK_B1": [16, 32, 64, 128],
        }
    ),
    key=["B1", "B2"],
)
@triton.jit
def _ewbo_bwd(
    X,  # Shape: (B1, B2)
    Y,  # Shape: (B1)
    DX,  # Shape: (B1, B2)
    DY,  # Shape: (B1, )
    DO,  # Shape: (B1, B2)
    OP: tl.constexpr,  # Operation type
    B1: tl.constexpr,
    B2: tl.constexpr,
    BLOCK_B1: tl.constexpr,
    BLOCK_B2: tl.constexpr,
):
    off_b1 = tl.program_id(0)

    # compute offset
    array_b1 = tl.arange(0, BLOCK_B1)
    array_b2 = tl.arange(0, BLOCK_B2)
    offset_b1 = off_b1 * BLOCK_B1
    offset_xo = offset_b1 * B2
    offset_y = offset_b1

    # compute block ptr
    dx_block_ptr = DX + offset_xo + array_b1[:, None] * B2 + array_b2[None, :]
    dy_block_ptr = DY + offset_y + array_b1[:, None]
    do_block_ptr = DO + offset_xo + array_b1[:, None] * B2 + array_b2[None, :]

    # mask
    mask_b1 = (off_b1 * BLOCK_B1 + array_b1) < B1
    NUM_BLOCKS_B2 = tl.cdiv(B2, BLOCK_B2)

    if OP == "mul" or OP == "div":
        y_block_ptr = Y + offset_y + array_b1[:, None]
        y = tl.load(y_block_ptr, mask=mask_b1[:, None], other=0)

    dy = tl.zeros((BLOCK_B1, 1), dtype=tl.float32)
    for i in range(NUM_BLOCKS_B2):
        # mask
        mask_b2 = array_b2 < B2
        mask = mask_b1[:, None] & mask_b2[None, :]

        # Load data
        do = tl.load(do_block_ptr, mask=mask, other=0)

        # Perform operation
        if OP == "add":  # add
            dx = do
            dy += tl.sum(do, axis=1, keep_dims=True)
        elif OP == "sub":  # sub
            dx = do
            dy += -tl.sum(do, axis=1, keep_dims=True)
        else:
            x_block_ptr = X + offset_xo + array_b1[:, None] * B2 + array_b2[None, :]
            x = tl.load(x_block_ptr, mask=mask, other=0)

            if OP == "mul":  # mul
                dx = do * y
                dy += tl.sum(do * x, axis=1, keep_dims=True)
            elif OP == "div":  # div
                dx = do / y
                dy += -tl.sum(do * x / (y * y), axis=1, keep_dims=True)

            x_block_ptr += BLOCK_B2

        # Store result
        if OP == "mul" or OP == "div":
            tl.store(dx_block_ptr, dx.to(DX.dtype.element_ty), mask=mask)
            dx_block_ptr += BLOCK_B2

        array_b2 += BLOCK_B2
        do_block_ptr += BLOCK_B2

    tl.store(dy_block_ptr, dy.to(DY.dtype.element_ty), mask=mask_b1[:, None])


def ewbo_triton_fwd(x, y, op="add", inplace=False):
    b1 = y.numel()
    b2 = x.numel() // b1

    if inplace:
        o = None
    else:
        o = torch.empty_like(x).contiguous()

    def grid(meta):
        return (triton.cdiv(b1, meta["BLOCK_B1"]),)

    MAX_BLOCK_SIZE = 64 * 1024
    BLOCK_B2 = min(triton.next_power_of_2(b2), MAX_BLOCK_SIZE)

    _ewbo_fwd[grid](
        X=x, Y=y, O=o, OP=op, INPLACE=inplace, B1=b1, B2=b2, BLOCK_B2=BLOCK_B2
    )

    if inplace:
        return x
    else:
        return o


class EWBOTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, y, op="add"):
        o = ewbo_triton_fwd(x, y, op)

        if op in ["mul", "div"]:
            ctx.save_for_backward(x, y)
        else:
            ctx.save_for_backward(None, None)

        b1 = y.numel()
        b2 = x.numel() // b1
        ctx.b1 = b1
        ctx.b2 = b2
        ctx.x_shape = x.shape
        ctx.y_shape = y.shape
        ctx.op = op

        return o

    @staticmethod
    def backward(ctx, do):
        x, y = ctx.saved_tensors
        b1 = ctx.b1
        b2 = ctx.b2
        x_shape = ctx.x_shape
        y_shape = ctx.y_shape
        op = ctx.op

        if op in ["mul", "div"]:
            dx = torch.empty(x_shape, dtype=do.dtype, device=do.device)
        else:
            dx = do
        dy = torch.empty(y_shape, dtype=do.dtype, device=do.device)

        def grid(meta):
            return (triton.cdiv(b1, meta["BLOCK_B1"]),)

        MAX_BLOCK_SIZE = 64 * 1024
        BLOCK_B2 = min(triton.next_power_of_2(b2), MAX_BLOCK_SIZE)

        _ewbo_bwd[grid](
            X=x, Y=y, DX=dx, DY=dy, DO=do, OP=op, B1=b1, B2=b2, BLOCK_B2=BLOCK_B2
        )

        return dx, dy, None, None


def ewbo_triton(x: torch.Tensor, y: torch.Tensor, op="add") -> torch.Tensor:
    """
    Element-wise binary operation using Triton.

    Args:
        x: Input tensor, (n1, ... , nk, n(k+1), ... , n(k+m), m >= 0)
        y: Input tensor, (n1, ... , nk)
        op: Binary operation to apply ("add", "mul", "sub", "div")

    Returns:
        Result of the binary operation
    """
    is_op_valid(op)
    is_dim_valid(x.shape, y.shape)
    return EWBOTriton.apply(x, y, op)


@contiguous
def ewbo_triton_fwd_fn(x, y, op="add", inplace=False):
    """
    Element-wise binary operation using Triton.

    Args:
        x: Input tensor of shape (..., N1, ... , Nk, N(k+1), ... , N(k+m), m >= 0)
        y: Input tensor of shape (..., N1, ... , Nk)
        op: Binary operation to apply ("add", "mul", "sub", "div")
        inplace: Whether to perform the operation in place, if inplace is True, the output pointer will be the same as the input x

    Returns:
        Result of the binary operation of shape (..., N1, ... , Nk, N(k+1), ... , N(k+m), m >= 0)
    """
    return ewbo_triton_fwd(x, y, op, inplace)


if __name__ == "__main__":
    # Test code
    b, n = 2, 512
    dtype = torch.float32
    x = torch.randn((b, n), dtype=dtype).cuda()
    y = torch.randn((b,), dtype=dtype).cuda()
    o = ewbo_triton(x, y, "mul")
    print(o.shape)
    o = ewbo_triton_fwd_fn(x, y, "mul")
    print(o.shape)
