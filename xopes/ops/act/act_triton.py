import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs({"BLOCK": [16, 32, 64, 128], "num_warps": [1, 2, 4, 8, 16, 32]}),
    key=["d"],
)
@triton.jit
def _act_fwd_triton(
    X,
    O,
    n: tl.constexpr,
    d: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_n = tl.program_id(0)
    off_block_d = tl.program_id(1)
    # compute offset
    offset_n = off_n * d
    offset_d = off_block_d * BLOCK
    # mask
    d_mask = (offset_d + tl.arange(0, BLOCK)) < d

    # compute
    x_block_ptr = X + offset_n + offset_d + tl.arange(0, BLOCK)
    o_block_ptr = O + offset_n + offset_d + tl.arange(0, BLOCK)
    x = tl.load(x_block_ptr, mask=d_mask, other=0).to(tl.float32)
    o = tl.zeros_like(x)

    if ACT == "relu":
        o = tl.where(x >= 0, x, 0)
    elif ACT == "sigmoid":
        o = tl.sigmoid(x)
    elif ACT == "silu":
        o = x * tl.sigmoid(x)

    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=d_mask)


@triton.autotune(
    generate_configs({"BLOCK": [16, 32, 64, 128], "num_warps": [2, 4, 8]}),
    key=["d"],
)
@triton.jit
def _act_bwd_triton(
    X,
    DO,
    DX,
    n: tl.constexpr,
    d: tl.constexpr,
    ACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_n = tl.program_id(0)
    off_block_d = tl.program_id(1)
    # compute offset
    offset_n = off_n * d
    offset_d = off_block_d * BLOCK
    # mask
    d_mask = (offset_d + tl.arange(0, BLOCK)) < d

    # compute
    x_block_ptr = X + offset_n + offset_d + tl.arange(0, BLOCK)
    do_block_ptr = DO + offset_n + offset_d + tl.arange(0, BLOCK)
    dx_block_ptr = DX + offset_n + offset_d + tl.arange(0, BLOCK)
    x = tl.load(x_block_ptr, mask=d_mask, other=0).to(tl.float32)
    do = tl.load(do_block_ptr, mask=d_mask, other=0).to(tl.float32)
    dx = tl.zeros_like(x)

    if ACT == "relu":
        dx = tl.where(x >= 0, do, 0)
    elif ACT == "sigmoid":
        sigmoid = tl.sigmoid(x)
        dx = do * sigmoid * (1 - sigmoid)
    elif ACT == "silu":
        sigmoid = tl.sigmoid(x)
        dx = do * sigmoid * (1 + x * (1 - sigmoid))

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=d_mask)


class ActTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, act="none"):
        o = act_fwd_triton(x, act)

        ctx.save_for_backward(x)
        ctx.act = act

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x = ctx.saved_tensors[0]
        act = ctx.act

        dx = act_bwd_triton(x, do, act)

        return dx, None


def act_fwd_triton(x, act="none", dim=None):
    if act == "none":
        return x

    shape = x.shape
    n = torch.prod(torch.tensor(shape[:-1])).item()
    d = x.shape[-1]
    o = torch.empty_like(x)

    def grid(meta):
        return (n, triton.cdiv(d, meta["BLOCK"]))

    _act_fwd_triton[grid](
        x,
        o,
        n,
        d,
        act,
    )

    return o


def act_bwd_triton(x, do, act="none", dim=None):
    if act == "none":
        return do

    shape = x.shape
    n = torch.prod(torch.tensor(shape[:-1])).item()
    d = x.shape[-1]

    dx = torch.empty_like(x)

    def grid(meta):
        return (n, triton.cdiv(d, meta["BLOCK"]))

    _act_bwd_triton[grid](
        x,
        do,
        dx,
        n,
        d,
        act,
    )

    return dx


def act_triton(x, act="none", dim=None):
    return ActTriton.apply(x, act)


if __name__ == "__main__":
    # unit test
    dtype = torch.bfloat16
    device = torch.cuda.current_device()

    b, n, d = 8, 128, 64
    x = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((b, n, d), dtype=dtype, device=device)
    act = "silu"

    o = act_triton(x, act)
    o.backward(do)
