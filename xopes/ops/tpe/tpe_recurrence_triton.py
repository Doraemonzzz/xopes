import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs(
        # {"BLOCK": [16, 32, 64, 128], "num_warps": [2, 4, 8]}
        {"BLOCK": [16], "num_warps": [2]}
    ),
    key=["n", "d"],
)
@triton.jit
def _tpe_recurrence_fwd(
    X,
    B,
    LOG_LAMBDA,
    O,
    b: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_d = tl.program_id(1)
    # compute offset
    offset_x = off_b * n * d + off_d * BLOCK
    offset_b = off_d * BLOCK * e

    x_block_ptr = X + offset_x + tl.arange(0, BLOCK)
    b_block_ptr = B + offset_b + tl.arange(0, e)
    log_lambda_block_ptr = LOG_LAMBDA + tl.arange(0, e)
    o_block_ptr = O + offset_x + tl.arange(0, BLOCK)

    h = tl.zeros([BLOCK, e], dtype=tl.float32)
    b = tl.load(b_block_ptr).to(tl.float32)[None, :]  # (1, e)
    lambda_ = tl.exp(tl.load(log_lambda_block_ptr).to(tl.float32))[None, :]  # (1, e)

    for i in range(n):
        x = tl.load(x_block_ptr).to(tl.float32)[:, None]  # (d, 1)
        h = lambda_ * h + b * x
        o = tl.sum(h, axis=0)

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_type))

        x_block_ptr += BLOCK
        o_block_ptr += BLOCK


@triton.autotune(
    generate_configs(
        # {"BLOCK": [16, 32, 64, 128], "num_warps": [2, 4, 8]}
        {"BLOCK": [16], "num_warps": [2]}
    ),
    key=["n", "d"],
)
@triton.jit
def _tpe_recurrence_bwd(
    X,
    B,
    LOG_LAMBDA,
    DO,
    DX,
    DB,
    DLOG_LAMBDA,
    b: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_d = tl.program_id(1)
    # compute offset
    offset_x = off_b * n * d + off_d * BLOCK
    offset_b = off_d * BLOCK * e

    x_block_ptr = X + offset_x + tl.arange(0, BLOCK)
    b_block_ptr = B + offset_b + tl.arange(0, e)
    log_lambda_block_ptr = LOG_LAMBDA + tl.arange(0, e)
    o_block_ptr = O + offset_x + tl.arange(0, BLOCK)

    h = tl.zeros([BLOCK, e], dtype=tl.float32)
    b = tl.load(b_block_ptr).to(tl.float32)[None, :]  # (1, e)
    lambda_ = tl.exp(tl.load(log_lambda_block_ptr).to(tl.float32))[None, :]  # (1, e)

    for i in range(n):
        x = tl.load(x_block_ptr).to(tl.float32)[:, None]  # (d, 1)
        h = lambda_ * h + b * x
        o = tl.sum(h, axis=0)

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_type))

        x_block_ptr += BLOCK
        o_block_ptr += BLOCK


class TpeRecurrence(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, b, log_lambda):
        b, n, d = x.shape
        e = log_lambda.shape[-1]
        o = torch.empty_like(x)

        def grid(meta):
            return (b, meta["BLOCK"])

        _tpe_recurrence_fwd[grid](x, b, log_lambda, o, b, n, d, e)

        ctx.save_for_backward(x, b, log_lambda_)

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, b, log_lambda = ctx.saved_tensors
        b, h, n, d = x.shape

        dx = torch.empty_like(x)
        db = torch.empty_like(b)
        dlog_lambda = torch.empty_like(log_lambda)

        def grid(meta):
            return (b, meta["BLOCK"])

        _tpe_recurrence_bwd[grid](x, b, log_lambda, do, dx, db, dlog_lambda, b, h, n, d)

        return dx, db, dlog_lambda


def tpe_recurrence_triton(x, b, log_lambda):
    # x: b, h, n, d
    # theta: h, d
    return TpeRecurrence.apply(x, b, log_lambda)
