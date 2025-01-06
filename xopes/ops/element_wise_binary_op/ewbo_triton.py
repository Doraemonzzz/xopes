import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs, is_dim_valid, is_op_valid


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
        }
    ),
    key=["B1", "B2"],
)
@triton.jit
def _ewbo_fwd(
    X,  # Shape: (B1, B2)
    Y,  # Shape: (B1)
    O,  # Output tensor
    OP: tl.constexpr,  # Operation type (0:add, 1:mul, 2:sub, 3:div)
    INPLACE: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
):
    off_b1 = tl.program_id(0)
    off_b2 = tl.program_id(1)

    # compute offset
    offset_xo = off_b1 * B2 + off_b2
    offset_y = off_b1
    # compute block ptr
    x_block_ptr = X + offset_xo
    y_block_ptr = Y + offset_y
    if INPLACE:
        o_block_ptr = x_block_ptr
    else:
        o_block_ptr = O + offset_xo

    # Load data
    x = tl.load(x_block_ptr)
    y = tl.load(y_block_ptr)

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
    tl.store(o_block_ptr, o.to(O.dtype.element_ty))


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
        }
    ),
    key=["B1", "B2"],
)
@triton.jit
def _ewbo_bwd(
    X,  # Shape: (B1, B2)
    Y,  # Shape: (B1)
    DX,  # Shape: (B1, B2)
    DY,  # Shape: (B1, B2)
    DO,  # Shape: (B1, B2)
    OP: tl.constexpr,  # Operation type (0:add, 1:mul, 2:sub, 3:div)
    B1: tl.constexpr,
    B2: tl.constexpr,
):
    off_b1 = tl.program_id(0)
    off_b2 = tl.program_id(1)

    # compute offset
    offset_xo = off_b1 * B2 + off_b2
    offset_y = off_b1
    # compute block ptr
    x_block_ptr = X + offset_xo
    y_block_ptr = Y + offset_y
    dx_block_ptr = DX + offset_xo
    dy_block_ptr = DY + offset_xo
    do_block_ptr = DO + offset_xo

    # Load data
    do = tl.load(do_block_ptr)

    # Perform operation
    if OP == "add":  # add
        dx = do
        dy = do
    elif OP == "sub":  # sub
        dx = do
        dy = -do
    elif OP == "mul":  # mul
        x = tl.load(x_block_ptr)
        y = tl.load(y_block_ptr)
        dx = do * y
        dy = do * x
    elif OP == "div":  # div
        x = tl.load(x_block_ptr)
        y = tl.load(y_block_ptr)
        dx = do / y
        dy = -do * x / (y * y)

    # Store result
    tl.store(dx_block_ptr, dx.to(DX.dtype.element_ty))
    tl.store(dy_block_ptr, dy.to(DY.dtype.element_ty))


def ewbo_triton_fwd(x, y, op="add", inplace=False):
    b1 = y.numel()
    b2 = x.numel() // b1

    if inplace:
        o = None
    else:
        o = torch.empty_like(x).contiguous()

    grid = (b1, b2)
    _ewbo_fwd[grid](
        X=x,
        Y=y,
        O=o,
        OP=op,
        INPLACE=inplace,
        B1=b1,
        B2=b2,
    )

    return o


class EWBOTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, y, op="add"):
        o = ewbo_triton_fwd(x, y, op)

        ctx.save_for_backward(x, y)
        ctx.x_shape = x.shape
        ctx.y_shape = y.shape
        ctx.op = op

        return o

    @staticmethod
    def backward(ctx, do):
        x, y = ctx.saved_tensors
        ctx.x_shape
        ctx.y_shape
        op = ctx.op
        b1 = y.numel()
        b2 = x.numel() // b1

        dx = torch.empty_like(x)
        dy_shape = list(y.shape)
        if b2 != 1:
            dy_shape += [b2]
        dy = torch.empty(dy_shape, dtype=x.dtype, device=x.device)

        grid = (b1, b2)
        _ewbo_bwd[grid](
            X=x,
            Y=y,
            DX=dx,
            DY=dy,
            DO=do,
            OP=op,
            B1=b1,
            B2=b2,
        )

        return dx, dy.sum(dim=-1) if b2 != 1 else dy, None, None


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
        x: Input tensor, (n1, ... , nk, n(k+1), ... , n(k+m), m >= 0)
        y: Input tensor, (n1, ... , nk)
        op: Binary operation to apply ("add", "mul", "sub", "div")
        inplace: Whether to perform the operation in place, if inplace is True, the output pointer will be the same as the input x
    Returns:
        Result of the binary operation
    """
    return ewbo_triton_fwd_fn(x, y, op, inplace)


if __name__ == "__main__":
    # Test code
    b, n = 2, 512
    dtype = torch.float32
    x = torch.randn((b, n), dtype=dtype).cuda()
    y = torch.randn((b,), dtype=dtype).cuda()
    o = ewbo_triton(x, y, "mul")
    print(o.shape)
