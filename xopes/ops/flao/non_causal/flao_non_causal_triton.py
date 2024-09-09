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
    key=["n", "m", "d", "e"],
)
@triton.jit
def _flao_non_causal_kv(
    K,
    V,
    KV,
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
    off_block_d = tl.program_id(1)
    off_block_e = tl.program_id(2)
    # compute offset
    offset_d = off_block_d * BLOCK_D
    offset_e = off_block_e * BLOCK_E
    off_bh * n * d + offset_d
    offset_k = off_bh * m * d + offset_d
    offset_v = off_bh * m * e + offset_e
    off_bh * n * e + offset_e
    off_block_d * b * h * n * e + off_bh * n * e + offset_e
    offset_kv = off_bh * d * e + offset_d * e + offset_e
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
    kv_block_ptr = (
        KV
        + offset_kv
        + tl.arange(0, BLOCK_D)[:, None] * e
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

    tl.store(
        kv_block_ptr,
        kv.to(kv_block_ptr.dtype.element_ty),
        mask=d_mask[:, None] & e_mask[None, :],
    )


@triton.autotune(
    generate_configs(
        {
            "BLOCK_NM": [16, 32, 64, 128],
            "BLOCK_E": [16, 32, 64, 128],
            "num_warps": [2, 4, 8],
        }
    ),
    key=["n", "m", "d", "e"],
)
@triton.jit
def _flao_non_causal_fwd(
    Q,
    G,
    KV,
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
    NUM_BLOCK_N = tl.cdiv(n, BLOCK_NM)
    off_bhn = tl.program_id(0)
    off_bh = off_bhn // NUM_BLOCK_N
    off_n = off_bhn % NUM_BLOCK_N
    off_block_d = tl.program_id(1)
    off_block_e = tl.program_id(2)
    # compute offset
    offset_d = off_block_d * BLOCK_D
    offset_e = off_block_e * BLOCK_E
    offset_n = off_n * BLOCK_NM
    offset_q = off_bh * n * d + offset_n * d + offset_d
    offset_g = off_bh * n * e + offset_n * e + offset_e
    offset_o = off_block_d * b * h * n * e + off_bh * n * e + offset_n * e + offset_e
    offset_kv = off_bh * d * e + offset_d * e + offset_e
    # mask
    d_mask = (offset_d + tl.arange(0, BLOCK_D)) < d
    e_mask = (offset_e + tl.arange(0, BLOCK_E)) < e

    array = tl.arange(0, BLOCK_NM)

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
    kv_block_ptr = (
        KV
        + offset_kv
        + tl.arange(0, BLOCK_D)[:, None] * e
        + tl.arange(0, BLOCK_E)[None, :]
    )
    o_block_ptr = (
        O
        + offset_o
        + tl.arange(0, BLOCK_NM)[:, None] * e
        + tl.arange(0, BLOCK_E)[None, :]
    )

    array = offset_n + tl.arange(0, BLOCK_NM)
    NUM_BLOCK_N = tl.cdiv(n, BLOCK_NM)

    mask = (array < n)[:, None]
    q = tl.load(q_block_ptr, mask=mask & d_mask[None, :], other=0).to(tl.float32)
    kv = tl.load(kv_block_ptr, mask=d_mask[:, None] & e_mask[None, :], other=0).to(
        tl.float32
    )
    g = tl.load(g_block_ptr, mask=mask & e_mask[None, :], other=0).to(tl.float32)

    qkv = tl.dot(q, kv)
    o = g * qkv

    tl.store(
        o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask & e_mask[None, :]
    )


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
        kv = torch.empty(b, h, d, e, dtype=torch.float32, device=q.device)

        def grid(meta):
            return (b * h, num_block_d, triton.cdiv(e, meta["BLOCK_E"]))

        # compute kv first
        _flao_non_causal_kv[grid](
            k,
            v,
            kv,
            b,
            h,
            n,
            m,
            d,
            e,
            block_d,
            num_block_d,
        )

        o = torch.empty(num_block_d, b, h, n, e, dtype=q.dtype, device=q.device)

        def grid(meta):
            return (
                b * h * triton.cdiv(n, meta["BLOCK_NM"]),
                num_block_d,
                triton.cdiv(e, meta["BLOCK_E"]),
            )

        _flao_non_causal_fwd[grid](
            q,
            g,
            kv,
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

        o = o.sum(dim=0)

        ctx.save_for_backward(q, k, v, g, kv)

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, k, v, g, kv = ctx.saved_tensors

        qkv = torch.matmul(q, kv.to(q.dtype))

        dg = do * qkv
        dqkv = do * g
        dq = torch.einsum("... n e, ... d e -> ... n d", dqkv, kv.to(q.dtype))
        dkv = torch.einsum("... n d, ... n e -> ... d e", q, dqkv)
        dk = torch.einsum("... n e, ... d e -> ... n d", v, dkv)
        dv = torch.einsum("... n d, ... d e -> ... n e", k, dkv)

        return dq, dk, dv, dg


def flao_non_causal_triton(q, k, v, g):
    return FusedLinearAttentionOutputGate.apply(q, k, v, g)


if __name__ == "__main__":
    # unit test
    dtype = torch.bfloat16
    device = torch.cuda.current_device()

    b, h, n, m, d, e = (6, 8, 512, 256, 128, 64)
    q = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((b, h, m, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, h, m, e), dtype=dtype, device=device).requires_grad_()
    g = torch.randn((b, h, n, e), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((b, h, n, e), dtype=dtype, device=device)

    o = flao_non_causal_triton(q, k, v, g)
    o.backward(do)
