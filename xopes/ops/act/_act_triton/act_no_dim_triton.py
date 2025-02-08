import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs, next_power_of_two


@triton.autotune(
    generate_configs({"num_warps": [1, 2, 4, 8, 16, 32], "num_stages": [2, 4]}),
    key=["N", "D"],
)
@triton.jit
def _act_no_dim_fwd_triton(
    X,
    O,
    N: tl.constexpr,
    D: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_n = tl.program_id(0)
    off_block_d = tl.program_id(1)
    # compute offset
    offset_n = off_n * D
    offset_d = off_block_d * BLOCK
    # mask
    mask_d = (offset_d + tl.arange(0, BLOCK)) < D

    # compute
    x_block_ptr = X + offset_n + offset_d + tl.arange(0, BLOCK)
    o_block_ptr = O + offset_n + offset_d + tl.arange(0, BLOCK)
    x = tl.load(x_block_ptr, mask=mask_d, other=0).to(tl.float32)
    o = x

    if ACT == "relu":
        o = tl.where(x >= 0, x, 0)
    elif ACT == "sigmoid":
        o = tl.sigmoid(x)
    elif ACT == "silu":
        o = x * tl.sigmoid(x)

    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask_d)


@triton.autotune(
    generate_configs({"num_warps": [1, 2, 4, 8, 16, 32], "num_stages": [2, 4]}),
    key=["N", "D"],
)
@triton.jit
def _act_no_dim_bwd_triton(
    X,
    DO,
    DX,
    N: tl.constexpr,
    D: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_n = tl.program_id(0)
    off_block_d = tl.program_id(1)
    # compute offset
    offset_n = off_n * D
    offset_d = off_block_d * BLOCK
    # mask
    mask_d = (offset_d + tl.arange(0, BLOCK)) < D

    # compute
    x_block_ptr = X + offset_n + offset_d + tl.arange(0, BLOCK)
    do_block_ptr = DO + offset_n + offset_d + tl.arange(0, BLOCK)
    dx_block_ptr = DX + offset_n + offset_d + tl.arange(0, BLOCK)
    x = tl.load(x_block_ptr, mask=mask_d, other=0).to(tl.float32)
    do = tl.load(do_block_ptr, mask=mask_d, other=0).to(tl.float32)
    # dx = tl.zeros_like(x)
    dx = do

    if ACT == "relu":
        dx = tl.where(x >= 0, do, 0)
    elif ACT == "sigmoid":
        sigmoid = tl.sigmoid(x)
        dx = do * sigmoid * (1 - sigmoid)
    elif ACT == "silu":
        sigmoid = tl.sigmoid(x)
        dx = do * sigmoid * (1 + x * (1 - sigmoid))

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=mask_d)


class ActNoDimTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, act="none"):
        o = act_no_dim_fwd_triton(
            x=x,
            act=act,
        )

        ctx.save_for_backward(x)
        ctx.act = act

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x = ctx.saved_tensors[0]
        act = ctx.act

        dx = act_no_dim_bwd_triton(
            x=x,
            do=do,
            act=act,
        )

        return dx, None


def act_no_dim_fwd_triton(x, act="none"):
    if act == "none":
        return x

    shape = x.shape
    n = torch.prod(torch.tensor(shape[:-1])).item()
    d = x.shape[-1]
    o = torch.empty_like(x)

    BLOCK = next_power_of_two(d)
    grid = (n,)
    _act_no_dim_fwd_triton[grid](
        X=x,
        O=o,
        N=n,
        D=d,
        ACT=act,
        BLOCK=BLOCK,
    )

    return o


def act_no_dim_bwd_triton(x, do, act="none"):
    if act == "none":
        return do

    shape = x.shape
    n = torch.prod(torch.tensor(shape[:-1])).item()
    d = x.shape[-1]

    dx = torch.empty_like(x)

    BLOCK = next_power_of_two(d)
    grid = (n,)
    _act_no_dim_bwd_triton[grid](
        X=x,
        DO=do,
        DX=dx,
        N=n,
        D=d,
        ACT=act,
        BLOCK=BLOCK,
    )

    return dx


def act_no_dim_triton(x, act="none"):
    """
    Apply activation function on x.

    Args:
        x: Input tensor of shape (..., D)
        act: Activation function, choose from ["none", "relu", "sigmoid", "silu"]

    Returns:
        output: Tensor of shape (..., D)
    """
    return ActNoDimTriton.apply(x, act)


if __name__ == "__main__":
    # unit test
    dtype = torch.bfloat16
    device = torch.cuda.current_device()

    b, n, d = 8, 128, 64
    x = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((b, n, d), dtype=dtype, device=device)
    act = "silu"

    o = act_no_dim_triton(x, act)
    o.backward(do)
