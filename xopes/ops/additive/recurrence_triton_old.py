import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, max_power_of_2_divisor

# HEAD_DIM = 64


def _get_fw_configs():
    return None


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_D': 32}, num_warps=1),
#         triton.Config({'BLOCK_D': 32}, num_warps=2),
#         triton.Config({'BLOCK_D': 32}, num_warps=4),
#         triton.Config({'BLOCK_D': 32}, num_warps=8),
#         triton.Config({'BLOCK_D': 64}, num_warps=1),
#         triton.Config({'BLOCK_D': 64}, num_warps=2),
#         triton.Config({'BLOCK_D': 64}, num_warps=4),
#         triton.Config({'BLOCK_D': 64}, num_warps=8),
#         triton.Config({'BLOCK_D': 128}, num_warps=1),
#         triton.Config({'BLOCK_D': 128}, num_warps=2),
#         triton.Config({'BLOCK_D': 128}, num_warps=4),
#         triton.Config({'BLOCK_D': 128}, num_warps=8),
#     ],
#     key=['d']
# )
@triton.jit
def _additive_recurrence_fwd(
    Q,
    K,
    V,
    G,
    O,
    S0,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NUM_BLOCK_D: tl.constexpr,
    NUM_BLOCK_E: tl.constexpr,
):
    off_bh = tl.program_id(2)
    off_bh % h
    off_bh // h
    off_d, off_e = tl.program_id(0), tl.program_id(1)
    # compute offset
    off_qkg = off_bh * n * d
    off_v = off_bh * n * e
    off_o = (off_d * b * h + off_bh) * n * e
    off_d = off_d * BLOCK_D
    off_e = off_e * BLOCK_E

    # get block ptr
    q_trans_block_ptr = tl.make_block_ptr(
        base=Q + off_qkg,
        shape=(d, n),
        strides=(1, d),
        offsets=(
            off_d,
            0,
        ),
        block_shape=(
            BLOCK_D,
            1,
        ),
        order=(0, 1),
    )
    k_trans_block_ptr = tl.make_block_ptr(
        base=K + off_qkg,
        shape=(d, n),
        strides=(1, d),
        offsets=(off_d, 0),
        block_shape=(BLOCK_D, 1),
        order=(0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
        base=V + off_v,
        shape=(n, e),
        strides=(e, 1),
        offsets=(0, off_e),
        block_shape=(1, BLOCK_E),
        order=(1, 0),
    )
    g_trans_block_ptr = tl.make_block_ptr(
        base=G + off_qkg,
        shape=(d, n),
        strides=(1, d),
        offsets=(
            off_d,
            0,
        ),
        block_shape=(BLOCK_D, 1),
        order=(0, 1),
    )
    o_block_ptr = tl.make_block_ptr(
        base=O + off_o,
        shape=(n, e),
        strides=(e, 1),
        offsets=(0, off_e),
        block_shape=(1, BLOCK_E),
        order=(1, 0),
    )

    s = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)
    denom = tl.zeros([BLOCK_D, 1], dtype=tl.float32)

    for i in range(n):
        # boundary check on feature dim
        q_trans = tl.load(q_trans_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        k_trans = tl.load(k_trans_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        v = tl.load(v_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        g_trans = tl.load(g_trans_block_ptr, boundary_check=(0, 1)).to(tl.float32)
        g_exp_trans = tl.exp(g_trans)

        k_bar_trans = g_exp_trans * k_trans
        # d 1, 1 e -> d e
        s += k_bar_trans.to(v.dtype) * v
        denom += g_exp_trans
        # d 1, d e -> d e
        o = (q_trans / denom) * (s)
        # d e -> 1 e
        o = tl.sum(o, axis=0)[None, :]

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), boundary_check=(0, 1))

        q_trans_block_ptr = tl.advance(q_trans_block_ptr, (0, 1))
        k_trans_block_ptr = tl.advance(k_trans_block_ptr, (0, 1))
        v_block_ptr = tl.advance(v_block_ptr, (1, 0))
        g_trans_block_ptr = tl.advance(g_trans_block_ptr, (0, 1))
        o_block_ptr = tl.advance(o_block_ptr, (1, 0))


class AdditiveRecurrenceFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, g, s=None):
        b, h, n, d = q.shape
        e = v.shape[-1]

        # split over head dim to avoid shared memory not enough
        head_dim = max_power_of_2_divisor(d, e)
        BLOCK_D, BLOCK_E = min(d, head_dim), min(e, head_dim)
        NUM_BLOCK_D, NUM_BLOCK_E = triton.cdiv(d, BLOCK_D), triton.cdiv(e, BLOCK_E)
        o = torch.empty(
            (NUM_BLOCK_D, b, h, n, e), dtype=q.dtype, device=torch.cuda.current_device()
        )

        grid = (
            NUM_BLOCK_D,
            NUM_BLOCK_E,
            b * h,
        )
        # print(torch.mean(o))
        _additive_recurrence_fwd[grid](
            q,
            k,
            v,
            g,
            o,
            s,
            b,
            h,
            n,
            d,
            e,
            BLOCK_D,
            BLOCK_E,
            NUM_BLOCK_D,
            NUM_BLOCK_E,
        )

        o = o.sum(0)

        ctx.save_for_backward(q, k, v, g, s)

        return o


def additive_rule_recurrence_triton(q, k, v, g, s=None, output_final_state=False):
    o = AdditiveRecurrenceFunction.apply(q, k, v, g, s)
    return o
