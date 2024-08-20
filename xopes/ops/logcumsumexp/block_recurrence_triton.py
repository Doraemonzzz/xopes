import torch
import triton
import triton.language as tl

# import pdb
from xopes.utils import contiguous, generate_configs, pack, unpack


@triton.autotune(
    generate_configs(
        {"BLOCK_N": [32, 64, 128], "BLOCK_D": [16, 32, 64, 128], "num_warps": [2, 4, 8]}
    ),
    key=["n", "d"],
)
@triton.jit
def _logcumsumexp_block_recurrence_fwd(
    X,
    O,
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

    m = tl.full([BLOCK_D], float("-inf"), dtype=tl.float32)
    o = tl.full([BLOCK_D], float("-inf"), dtype=tl.float32)
    x_block_ptr = (
        X + off + tl.arange(0, BLOCK_N)[:, None] * d + tl.arange(0, BLOCK_D)[None, :]
    )
    o_block_ptr = (
        O + off + tl.arange(0, BLOCK_N)[:, None] * d + tl.arange(0, BLOCK_D)[None, :]
    )

    # get accumulation matrix, using this to compute cumsum
    # | 1 0 0 | | x1 |   | x1           |
    # | 1 1 0 | | x2 | = | x1 + x2      | = cumsum({x1, x2, x3})
    # | 1 1 1 | | x3 |   | x1 + x2 + x3 |
    index = tl.arange(0, BLOCK_N)
    acc_matrix = tl.where(index[:, None] >= index[None, :], 1.0, 0.0)
    feature_mask = (off_d * BLOCK_D + tl.arange(0, BLOCK_D) < d)[None, :]

    for i in range(tl.cdiv(n, BLOCK_N)):
        mask = (i * BLOCK_N + tl.arange(0, BLOCK_N) < n)[:, None] and feature_mask

        x = tl.load(x_block_ptr, mask=mask).to(tl.float32)

        # get the max value in the block
        m_ = tl.max(x, axis=0)
        # update cummax
        m_ = tl.maximum(m, m_)

        # compute cumsum(exp(x - m_)) using matrix production
        x_exp_stable = tl.exp(x - m_)
        x_cumsum_exp = tl.dot(acc_matrix, x_exp_stable)

        o_ = tl.log(tl.exp(o + m - m_) + x_cumsum_exp)
        m = m_
        # we whant the get o_[-1], however, triton doesn't support this,
        # since o_ is monotonically increasing on sequence dim,
        # we can use the max to get this
        o = tl.max(o_, 0)
        o_res = o_ + m

        tl.store(o_block_ptr, o_res.to(o_block_ptr.dtype.element_ty), mask=mask)

        x_block_ptr += BLOCK_N * d
        o_block_ptr += BLOCK_N * d


@triton.autotune(
    # generate_configs(
    #     {"BLOCK_N": [32, 64, 128], "BLOCK_D": [16, 32, 64, 128], "num_warps": [2, 4, 8]}
    # ),
    # key=["n", "d"],
    generate_configs({"BLOCK_N": [32], "BLOCK_D": [16], "num_warps": [2]}),
    key=["n", "d"],
)
@triton.jit
def _logcumsumexp_block_recurrence_bwd(
    X,
    O,
    DX,
    DO,
    b: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
       ______
    0 ｜     ｜
    1 ｜     ｜
       ------
    2 ｜     ｜
       ——————
    3 ｜     ｜|
    4 ｜     ｜
    5 ｜     ｜
       ——————
    6 ｜     ｜
    7 ｜     ｜
       ------
    8 ｜     ｜
       ——————
    Assume the sequence length is 8, block size is 3, there are 3 blocks, and the index is 1, which is at block 0, we assume the index start from 0.
    The algorithm is as follows:
        1. Compute the 2th block (mask the position whose index >= 8 with 0)
        2. Compute the 1th block
        3. Compute the 0th block (mask the position whose index < 1 with 0)
    """
    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    off_d = tl.program_id(2)
    # compute offset
    off_x = off_b * n * d + off_n * d + off_d * BLOCK_D
    # start from the last block
    num_block = tl.cdiv(n, BLOCK_N)
    block_idx = off_n // BLOCK_N
    off_o = off_b * n * d + (num_block - 1) * BLOCK_N * d + off_d * BLOCK_D

    x_block_ptr = X + off_x + tl.arange(0, BLOCK_D)
    o_block_ptr = (
        O + off_o + tl.arange(0, BLOCK_N)[:, None] * d + tl.arange(0, BLOCK_D)[None, :]
    )
    dx_block_ptr = DX + off_x + tl.arange(0, BLOCK_D)
    do_block_ptr = (
        DO + off_o + tl.arange(0, BLOCK_N)[:, None] * d + tl.arange(0, BLOCK_D)[None, :]
    )

    # get rev accumulation matrix, using this to compute revcumsum
    # | 1 1 1 | | x1 |   | x3 + x2 + x1 |
    # | 0 1 1 | | x2 | = | x3 + x2      | = revcumsum({x1, x2, x3})
    # | 0 0 1 | | x3 |   | x3           |
    index = tl.arange(0, BLOCK_N)
    acc_matrix = tl.where(index[:, None] <= index[None, :], 1.0, 0.0)
    feature_mask = off_d * BLOCK_D + tl.arange(0, BLOCK_D) < d
    # feature_mask = tl.arange(0, BLOCK_D) < BLOCK_D

    # sequence mask
    # sequence_mask_front = ((block_idx * BLOCK_N + tl.arange(0, BLOCK_N)) >= off_n)[:, None]
    sequence_mask_front = ((block_idx * BLOCK_N + tl.arange(0, BLOCK_N)) >= off_n)[
        :, None
    ] and ((block_idx * BLOCK_N + tl.arange(0, BLOCK_N)) < n)[:, None]
    array = (num_block - 1) * BLOCK_N + tl.arange(0, BLOCK_N)

    # use this mask to get first row of a matrix
    index_mask = (tl.arange(0, BLOCK_N) == 0)[:, None]

    x = tl.load(x_block_ptr, mask=feature_mask, other=0).to(tl.float32)
    dx = tl.zeros([BLOCK_D], dtype=tl.float32)
    # loop from last block to the first block
    # tl.device_print("aaa", num_block - block_idx)

    # pdb.set_trace()
    # print(acc_matrix)
    m = num_block - block_idx
    for j in range(m):
        sequence_mask_end = (array < n)[:, None]
        # tl.static_print("aaa", feature_mask[None, :], sequence_mask_front, sequence_mask_end)
        # if j == m - 1:
        #     mask = feature_mask[None, :] and sequence_mask_front
        # else:
        #     mask = feature_mask[None, :] and sequence_mask_end
        # mask = feature_mask[None, :] and sequence_mask_end
        mask = (feature_mask[None, :] and sequence_mask_front) and sequence_mask_end

        # tl.device_print("aaa", mask)

        o = tl.load(o_block_ptr, mask=mask, other=0).to(tl.float32)
        do = tl.load(do_block_ptr, mask=mask, other=0).to(tl.float32)

        tmp = do * tl.exp(x - o)
        dx_arr = dx + tl.dot(acc_matrix, tmp)

        # we use this to get the first row of dx_arr,
        # since triton doesn't support index operation
        dx = tl.sum(tl.where(index_mask, dx_arr, 0), axis=0)

        array -= BLOCK_N
        o_block_ptr -= BLOCK_N * d
        do_block_ptr -= BLOCK_N * d

    tl.store(dx_block_ptr, dx.to(dx_block_ptr.dtype.element_ty), mask=feature_mask)


class LogCumSumExpBlockRecurrence(torch.autograd.Function):
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

        # parallel over batch and feature
        def grid(meta):
            return (b, triton.cdiv(d, meta["BLOCK_D"]))

        _logcumsumexp_block_recurrence_fwd[grid](x, o, b, n, d)

        ctx.save_for_backward(x, o)
        ctx.dim = dim

        o = unpack(o, ps, "* n d", is_list)
        if dim != -2:
            o = o.transpose(-2, dim).contiguous()

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        x, o = ctx.saved_tensors
        dim = ctx.dim
        b, n, d = x.shape
        # print(x.shape)

        dx = torch.empty_like(x)

        if dim != -2:
            do = do.transpose(-2, dim).contiguous()

        do, ps, is_list = pack(do, "* n d")

        # parallel over batch, sequence and feature
        def grid(meta):
            return (b, n, triton.cdiv(d, meta["BLOCK_D"]))

        _logcumsumexp_block_recurrence_bwd[grid](x, o, dx, do, b, n, d)

        dx = unpack(dx, ps, "* n d", is_list)
        if dim != -2:
            dx = dx.transpose(-2, dim).contiguous()

        return dx, None


def logcumsumexp_block_recurrence_triton(x, dim=-2):
    return LogCumSumExpBlockRecurrence.apply(x, dim)
