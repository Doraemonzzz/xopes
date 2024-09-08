import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs(
        {
            "BLOCK_NM": [16, 32, 64, 128],
            "BLOCK_E": [16, 32, 64, 128],
            "num_warps": [2, 4, 8],
        }
    ),
    # generate_configs({"BLOCK_NM": [64, 128], "BLOCK_E": [64, 128], "num_warps": [2, 4, 8]}),
    # generate_configs({"BLOCK_NM": [64, 128], "BLOCK_E": [64, 128],}),
    # generate_configs({"BLOCK_NM": [64], "BLOCK_E": [64],}),
    key=["n", "m", "d", "e"],
)
@triton.jit
def _flao_non_causal_fwd(
    Q,
    K,
    V,
    G,
    O,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    m: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCK_D: tl.constexpr,
    BLOCK_NM: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_bh // h
    off_bh % h
    off_block_d = tl.program_id(1)
    off_block_e = tl.program_id(2)
    # compute offset
    offset_d = off_block_d * BLOCK_D
    offset_e = off_block_e * BLOCK_E
    offset_q = off_bh * n * d + offset_d
    offset_k = off_bh * m * d + offset_d
    offset_v = off_bh * m * e + offset_e
    offset_g = off_bh * n * e + offset_e
    offset_o = off_block_d * b * h * n * e + off_bh * n * e + offset_e
    # mask
    d_mask = (offset_d + tl.arange(0, BLOCK_D)) < d
    e_mask = (offset_e + tl.arange(0, BLOCK_E)) < e

    # compute kv
    k_trans_block_ptr = (
        K
        + offset_k
        + tl.arange(0, BLOCK_NM)[None, :] * d
        + tl.arange(0, BLOCK_D)[:, None]
    )
    v_block_ptr = (
        V
        + offset_v
        + tl.arange(0, BLOCK_NM)[:, None] * e
        + tl.arange(0, BLOCK_E)[None, :]
    )
    array = tl.arange(0, BLOCK_NM)
    NUM_BLOCK_M = tl.cdiv(m, BLOCK_NM)

    kv = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)
    for i in range(0, NUM_BLOCK_M):
        mask = array < m
        k_trans = tl.load(
            k_trans_block_ptr, mask=mask[None, :] & d_mask[:, None], other=0
        ).to(tl.float32)
        v = tl.load(v_block_ptr, mask=mask[:, None] & e_mask[None, :], other=0).to(
            tl.float32
        )
        kv += tl.dot(k_trans, v)

        k_trans_block_ptr += BLOCK_NM * d
        v_block_ptr += BLOCK_NM * e
        array += BLOCK_NM

    # compute qkv
    q_block_ptr = (
        Q
        + offset_q
        + tl.arange(0, BLOCK_NM)[:, None] * d
        + tl.arange(0, BLOCK_D)[None, :]
    )
    g_block_ptr = (
        G
        + offset_g
        + tl.arange(0, BLOCK_NM)[:, None] * e
        + tl.arange(0, BLOCK_E)[None, :]
    )
    o_block_ptr = (
        O
        + offset_o
        + tl.arange(0, BLOCK_NM)[:, None] * e
        + tl.arange(0, BLOCK_E)[None, :]
    )
    array = tl.arange(0, BLOCK_NM)
    NUM_BLOCK_N = tl.cdiv(n, BLOCK_NM)

    for i in range(0, NUM_BLOCK_N):
        mask = (array < n)[:, None]
        q = tl.load(q_block_ptr, mask=mask & d_mask[None, :], other=0).to(tl.float32)
        g = tl.load(g_block_ptr, mask=mask & e_mask[None, :], other=0).to(tl.float32)

        qkv = tl.dot(q, kv)
        o = g * qkv

        tl.store(
            o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask & e_mask[None, :]
        )

        q_block_ptr += BLOCK_NM * d
        g_block_ptr += BLOCK_NM * e
        o_block_ptr += BLOCK_NM * e
        array += BLOCK_NM


class FusedLinearAttentionOutputGate(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, q, k, v, g):
        b, h, n, d = q.shape
        m = k.shape[-2]
        e = v.shape[-1]

        block_d = min(128, triton.next_power_of_2(d))
        # block_d = d
        num_block_d = triton.cdiv(d, block_d)
        # split over q, k head dim to severel chunk of block_d
        o = torch.empty(num_block_d, b, h, n, e, dtype=q.dtype, device=q.device)

        def grid(meta):
            return (b * h, num_block_d, triton.cdiv(e, meta["BLOCK_E"]))

        _flao_non_causal_fwd[grid](
            q,
            k,
            v,
            g,
            o,
            b,
            h,
            n,
            m,
            d,
            e,
            block_d,
            num_block_d,
        )

        ctx.save_for_backward(q, k, v, g)

        o = o.sum(dim=0)

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, v, g = ctx.saved_tensors

        return dq, dk, dv, dg


def flao_non_causal_triton(q, k, v, g):
    return FusedLinearAttentionOutputGate.apply(q, k, v, g)


if __name__ == "__main__":
    # unit test
    import os

    os.environ["XOPES_DEBUG"] = "True"

    dtype = torch.bfloat16
    device = torch.cuda.current_device()

    b, h, n, m, d, e = (6, 8, 512, 256, 128, 64)
    q = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((b, h, m, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, h, m, e), dtype=dtype, device=device).requires_grad_()
    g = torch.randn((b, h, n, e), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((b, h, n, e), dtype=dtype, device=device)

    o = flao_non_causal_triton(q, k, v, g)
    # o.backward(do)
