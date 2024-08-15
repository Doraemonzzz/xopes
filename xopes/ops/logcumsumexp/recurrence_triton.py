import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs, pack, unpack


@triton.autotune(
    generate_configs({"BLOCK": [16, 32, 64, 128], "num_warps": [2, 4, 8]}),
    key=["d"],
)
@triton.jit
def _logcumsumexp_recurrence_fwd(
    X,
    O,
    b: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_d = tl.program_id(1)
    # compute offset
    off = off_b * n * d + off_d * BLOCK

    m = tl.full([BLOCK], float("-inf"), dtype=tl.float32)
    o = tl.full([BLOCK], float("-inf"), dtype=tl.float32)
    x_block_ptr = X + off + tl.arange(0, BLOCK)
    o_block_ptr = O + off + tl.arange(0, BLOCK)
    mask = off_d * BLOCK + tl.arange(0, BLOCK) < d

    # m = tl.full([1, BLOCK], float("-inf"), dtype=tl.float32)
    # o = tl.full([1, BLOCK], float("-inf"), dtype=tl.float32)
    # x_block_ptr = tl.make_block_ptr(
    #     base=X+off,
    #     shape=(n, d),
    #     strides=(d, 1),
    #     offsets=(0, 0),
    #     block_shape=(1, BLOCK),
    #     order=(1, 0)
    # )
    # o_block_ptr = tl.make_block_ptr(
    #     base=O+off,
    #     shape=(n, d),
    #     strides=(d, 1),
    #     offsets=(0, 0),
    #     block_shape=(1, BLOCK),
    #     order=(1, 0)
    # )

    for i in range(n):
        # x = tl.load(x_block_ptr, boundary_check=(0, 1)).to(tl.float32)

        x = tl.load(x_block_ptr, mask=mask).to(tl.float32)
        m_ = tl.maximum(x, m)

        o = tl.log(tl.exp(o + m - m_) + tl.exp(x - m_))
        m = m_
        o_res = o + m

        # tl.store(o_block_ptr, o_res.to(o_block_ptr.dtype.element_ty), boundary_check=(0, 1))
        tl.store(o_block_ptr, o_res.to(o_block_ptr.dtype.element_ty), mask=mask)

        x_block_ptr += d
        o_block_ptr += d

        # x_block_ptr = tl.advance(x_block_ptr, (1, 0))
        # o_block_ptr = tl.advance(o_block_ptr, (1, 0))


class LogCumSumExpRecurrence(torch.autograd.Function):
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
            return (b, triton.cdiv(d, meta["BLOCK"]))

        _logcumsumexp_recurrence_fwd[grid](x, o, b, n, d)

        o = unpack(o, ps, "* n d", is_list)
        if dim != -2:
            o = o.transpose(-2, dim).contiguous()

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        return None


def logcumsumexp_recurrence_triton(x, dim=-2):
    return LogCumSumExpRecurrence.apply(x, dim)
