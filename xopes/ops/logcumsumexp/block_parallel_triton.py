import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs, pack, unpack


@triton.autotune(
    generate_configs({"BLOCK_D": [16, 32, 64, 128], "num_warps": [2, 4, 8]}),
    key=["n", "d"],
)
@triton.jit
def _logcumsumexp_block_parallel_compute(
    X,
    O,
    M,
    b: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    off_d = tl.program_id(2)
    # compute offset
    off = off_b * n * d + off_n * BLOCK_N * d + off_d * BLOCK_D
    off_m = off_b * tl.cdiv(n, BLOCK_N) * d + off_n * d + off_d * BLOCK_D

    m = tl.full([BLOCK_D], float("-inf"), dtype=tl.float32)
    o = tl.full([BLOCK_D], float("-inf"), dtype=tl.float32)
    x_block_ptr = (
        X + off + tl.arange(0, BLOCK_N)[:, None] * d + tl.arange(0, BLOCK_D)[None, :]
    )
    o_block_ptr = (
        O + off + tl.arange(0, BLOCK_N)[:, None] * d + tl.arange(0, BLOCK_D)[None, :]
    )
    m_block_ptr = M + off_m + tl.arange(0, BLOCK_D)

    # get accumulation matrix, using this to compute cumsum
    # | 1 0 0 | | x1 |   | x1           |
    # | 1 1 0 | | x2 | = | x1 + x2      | = cumsum({x1, x2, x3})
    # | 1 1 1 | | x3 |   | x1 + x2 + x3 |
    index = tl.arange(0, BLOCK_N)
    acc_matrix = tl.where(index[:, None] >= index[None, :], 1.0, 0.0)
    feature_mask = off_d * BLOCK_D + tl.arange(0, BLOCK_D) < d

    mask = (off_n * BLOCK_N + tl.arange(0, BLOCK_N) < n)[:, None] and feature_mask[
        None, :
    ]

    # !!!!! important, we don't know which value for padding, this may cause bug in the future
    x = tl.load(
        x_block_ptr,
        mask=mask,
    ).to(tl.float32)

    # get the max value in the block
    m = tl.max(x, axis=0)

    # compute cumsum(exp(x - m)) using matrix production
    x_exp_stable = tl.exp(x - m)
    x_cumsum_exp = tl.dot(acc_matrix, x_exp_stable)

    o = tl.log(x_cumsum_exp)

    tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask)
    tl.store(m_block_ptr, m.to(o_block_ptr.dtype.element_ty), mask=feature_mask)


@triton.autotune(
    generate_configs({"BLOCK_D": [16, 32, 64, 128], "num_warps": [2, 4, 8]}),
    key=["n", "d"],
)
@triton.jit
def _logcumsumexp_block_parallel_reduce(
    X,
    O,
    M,
    b: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_d = tl.program_id(1)
    # compute offset
    off = off_b * n * d + off_d * BLOCK_D
    off_m = off_b * tl.cdiv(n, BLOCK_N) * d + off_d * BLOCK_D

    m = tl.full([BLOCK_D], float("-inf"), dtype=tl.float32)
    o = tl.full([BLOCK_D], float("-inf"), dtype=tl.float32)

    o_block_ptr = (
        O + off + tl.arange(0, BLOCK_N)[:, None] * d + tl.arange(0, BLOCK_D)[None, :]
    )
    m_block_ptr = M + off_m + tl.arange(0, BLOCK_D)

    feature_mask = off_d * BLOCK_D + tl.arange(0, BLOCK_D) < d

    for i in range(tl.cdiv(n, BLOCK_N)):
        mask = (i * BLOCK_N + tl.arange(0, BLOCK_N) < n)[:, None] and feature_mask[
            None, :
        ]

        o_stage1 = tl.load(o_block_ptr, mask=mask).to(tl.float32)
        m_stage1 = tl.load(m_block_ptr, mask=feature_mask).to(tl.float32)

        # get the max value in the block
        # update cummax
        m_ = tl.maximum(m, m_stage1)

        o_ = tl.log(tl.exp(o + m - m_) + tl.exp(o_stage1 + m_stage1 - m_))
        m = m_
        # we whant the get o_[-1], however, triton doesn't support this,
        # since o_ is monotonically increasing on sequence dim,
        # we can use the max to get this
        o = tl.max(o_, 0)
        o_res = o_ + m

        tl.store(o_block_ptr, o_res.to(o_block_ptr.dtype.element_ty), mask=mask)

        o_block_ptr += BLOCK_N * d
        m_block_ptr += d


class LogCumSumExpBlockParallel(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, dim=-2):
        x.dtype
        if dim >= 0:
            dim -= len(x.shape)

        if dim != -2:
            x = x.transpose(-2, dim).contiguous()

        x, ps, is_list = pack(x, "* n d")
        b, n, d = x.shape
        o = torch.empty_like(x)
        BLOCK_N = 128
        m = torch.empty(
            b,
            triton.cdiv(n, BLOCK_N),
            d,
            dtype=x.dtype,
            device=torch.cuda.current_device(),
        )

        # parallel over batch, sequence and feature
        def grid(meta):
            return (b, triton.cdiv(n, BLOCK_N), triton.cdiv(d, meta["BLOCK_D"]))

        _logcumsumexp_block_parallel_compute[grid](x, o, m, b, n, d, BLOCK_N)

        # reduce
        def grid(meta):
            return (b, triton.cdiv(d, meta["BLOCK_D"]))

        _logcumsumexp_block_parallel_reduce[grid](x, o, m, b, n, d, BLOCK_N)

        o = unpack(o, ps, "* n d", is_list)
        if dim != -2:
            o = o.transpose(-2, dim).contiguous()

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        return None


def logcumsumexp_block_parallel_triton(x, dim=-2):
    return LogCumSumExpBlockParallel.apply(x, dim)
