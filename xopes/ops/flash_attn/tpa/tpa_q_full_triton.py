from typing import Optional

import torch
import triton
import triton.language as tl

from xopes.utils import generate_configs


@triton.autotune(
    generate_configs(
        {
            "num_warps": [
                2,
                4,
                8,
            ],
            "BLOCK": [
                128,
            ],
        }
    ),
    key=["B", "H", "D", "E", "M"],
)
@triton.jit
def _tpa_q_full_decode_parallel_bhn(
    Q,  # B N H D
    AK,  # B M H
    AV,  # B M H
    BK,  # B M D
    BV,  # B M E
    O,  # B NUM_BLOCK_M N H E
    LSE,  # B NUM_BLOCK_M H
    CU_SEQLENS,  # L
    SCALE: tl.constexpr,
    SCALE_K: tl.constexpr,
    SCALE_V: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_BLOCK_M: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_m = tl.program_id(1)
    off_h = tl.program_id(2)

    # compute offset
    offset_m = off_m * BLOCK_M
    offset_q = off_b * N * H * D + off_h * D
    offset_ak = off_b * M * H + off_h
    offset_av = off_b * M * H + off_h
    offset_bk = off_b * M * D
    offset_bv = off_b * M * E
    offset_o = off_b * NUM_BLOCK_M * N * H * E + off_m * N * H * E + off_h * E
    offset_lse = off_b * NUM_BLOCK_M * H + off_m * H + off_h

    # compute block ptr and mask
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    array_m = offset_m + tl.arange(0, BLOCK)

    mask_d = array_d < D
    mask_e = array_e < E

    q_block_ptr = Q + offset_q + array_d  # D
    ak_block_ptr = AK + offset_ak + array_m[:, None] * H  # M 1
    av_block_ptr = AV + offset_av + array_m[:, None] * H  # M 1
    bk_block_ptr = BK + offset_bk + array_m[:, None] * D + array_d[None, :]  # M D
    bv_block_ptr = BV + offset_bv + array_m[:, None] * E + array_e[None, :]  # M E

    cnt = offset_m
    NUM_BLOCKS = tl.cdiv(BLOCK_M, BLOCK)

    o = tl.zeros(
        [
            BLOCK_E,
        ],
        dtype=tl.float32,
    )
    m = tl.full([1], -float("inf"), dtype=tl.float32)
    sse = tl.full([1], 0, dtype=tl.float32)
    SCALE * SCALE_K

    q = tl.load(q_block_ptr, mask=mask_d, other=0)

    for i in range(NUM_BLOCKS):
        if cnt < M:
            mask_m = (i * BLOCK + array_m) < M

            ak = tl.load(ak_block_ptr, mask=mask_m[:, None], other=0)
            av = tl.load(av_block_ptr, mask=mask_m[:, None], other=0)
            bk = tl.load(bk_block_ptr, mask=mask_m[:, None] & mask_d[None, :], other=0)
            bv = tl.load(bv_block_ptr, mask=mask_m[:, None] & mask_e[None, :], other=0)

            # M D
            k = ak * bk
            # M E
            v = av * bv

            # M 1
            score = tl.sum(q[None, :] * k, axis=1, keep_dims=True)

            # safe softmax
            # local attention
            # M 1 -> 1
            mi = tl.max(score, axis=0)
            m_ = tl.maximum(m, mi)
            # M 1 -> 1
            sse_local = tl.sum(tl.exp(score - m_), axis=0)
            # M 1, 1 -> M 1
            p = tl.exp(score - m_) / sse_local
            # M 1, M E -> E
            o_ = tl.sum(p * v, axis=0)

            # update
            sse = tl.exp(m - m_) * sse + sse_local
            ratio = sse_local / sse
            o = (1 - ratio) * o + ratio * o_

            ak_block_ptr += BLOCK * H
            av_block_ptr += BLOCK * H
            bk_block_ptr += BLOCK * D
            bv_block_ptr += BLOCK * E
            cnt += BLOCK
            m = m_

    o_block_ptr = O + offset_o + array_e  # E

    tl.store(
        o_block_ptr,
        o.to(o_block_ptr.dtype.element_ty),
        mask=mask_e,
    )

    lse = tl.log(sse) + m
    lse_block_ptr = LSE + offset_lse + tl.arange(0, 1)

    tl.store(
        lse_block_ptr,
        lse.to(lse_block_ptr.dtype.element_ty),
    )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [
                2,
                4,
                8,
            ],
        }
    ),
    key=[
        "B",
        "H",
        "D",
        "E",
    ],
)
@triton.jit
def _tpa_q_full_decode_reduce(
    X,  # B NUM_BLOCK_M N H E
    LSE,  # B NUM_BLOCK_M H
    O,  # B N H E
    CU_SEQLENS,  # L
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_NUM_BLOCK_M: tl.constexpr,
    NUM_BLOCK_M: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)

    # compute offset
    offset_x = off_b * NUM_BLOCK_M * N * H * E + off_h * E
    offset_lse = off_b * NUM_BLOCK_M * H + off_h
    offset_o = off_b * N * H * E + off_h * E

    # compute block ptr and mask
    array_e = tl.arange(0, BLOCK_E)
    array_num_block_m = tl.arange(0, BLOCK_NUM_BLOCK_M)

    mask_e = array_e < E
    mask_num_block_m = array_num_block_m < NUM_BLOCK_M

    x_block_ptr = (
        X + offset_x + array_num_block_m[:, None] * N * H * E + array_e[None, :]
    )  # NUM_BLOCK_M E
    lse_block_ptr = LSE + offset_lse + array_num_block_m  # NUM_BLOCK_M
    o_block_ptr = O + offset_o + array_e  # E

    x = tl.load(x_block_ptr, mask=mask_num_block_m[:, None] & mask_e[None, :], other=0)
    lse = tl.load(lse_block_ptr, mask=mask_num_block_m, other=0)
    m = tl.min(lse)
    p = tl.exp(lse - m)
    p = tl.where(mask_num_block_m, p, 0)
    p = p / tl.sum(p)

    o = tl.sum(x * p[:, None], axis=0)
    tl.store(o_block_ptr, o, mask=mask_e)


def tpa_q_full_decode_parallel_bh_triton(
    q: torch.Tensor,
    ak: torch.Tensor,
    av: torch.Tensor,
    bk: torch.Tensor,
    bv: torch.Tensor,
    scale: Optional[float] = None,
    scale_k: Optional[float] = None,
    scale_v: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Apply Flash Attention for Tensor Product Attention.

    Args:
        q: Query tensor of shape (B, N, H, D)
        ak: Key A tensor of shape (B, M, H)
        av: Value A tensor of shape (B, M, H)
        bk: Key B tensor of shape (B, M, D)
        bv: Value B tensor of shape (B, M, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        Output tensor of shape (B, N, H, E)
    """
    b, n, h, d = q.shape
    assert n == 1, "n must be 1 when using tpa_q_full_decode_parallel_bh_triton"
    m = ak.shape[1]
    e = bv.shape[-1]

    if scale is None:
        scale = d**-0.5
    if scale_k is None:
        scale_k = 1
    if scale_v is None:
        scale_v = 1

    if b <= 16:
        BLOCK_M = 512
    else:
        BLOCK_M = 1024
    NUM_BLOCK_M = triton.cdiv(m, BLOCK_M)

    def grid(meta):
        return (b, NUM_BLOCK_M, h)

    o_ = torch.empty((b, NUM_BLOCK_M, n, h, e), dtype=q.dtype, device=q.device)
    lse = torch.empty((b, NUM_BLOCK_M, h), dtype=q.dtype, device=q.device)

    BLOCK_D = triton.next_power_of_2(d)
    BLOCK_E = triton.next_power_of_2(e)

    _tpa_q_full_decode_parallel_bhn[grid](
        Q=q,
        AK=ak,
        AV=av,
        BK=bk,
        BV=bv,
        O=o_,
        LSE=lse,
        CU_SEQLENS=cu_seqlens,
        SCALE=scale,
        SCALE_K=scale_k,
        SCALE_V=scale_v,
        B=b,
        N=n,
        M=m,
        H=h,
        D=d,
        E=e,
        BLOCK_D=BLOCK_D,
        BLOCK_E=BLOCK_E,
        BLOCK_M=BLOCK_M,
        NUM_BLOCK_M=NUM_BLOCK_M,
    )

    def grid(meta):
        return (b, h)

    o = torch.empty((b, n, h, e), dtype=q.dtype, device=q.device)
    BLOCK_NUM_BLOCK_M = triton.next_power_of_2(NUM_BLOCK_M)

    _tpa_q_full_decode_reduce[grid](
        X=o_,
        LSE=lse,
        O=o,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        E=e,
        BLOCK_E=BLOCK_E,
        BLOCK_M=BLOCK_M,
        BLOCK_NUM_BLOCK_M=BLOCK_NUM_BLOCK_M,
        NUM_BLOCK_M=NUM_BLOCK_M,
    )

    return o


if __name__ == "__main__":
    b = 2
    n = 1
    m = 16
    h = 32
    r = 16
    d = 128
    e = 64
    dtype = torch.bfloat16
    q = torch.randn((b, n, h, d), dtype=dtype).cuda()
    ak = torch.randn((b, m, h), dtype=dtype).cuda()
    av = torch.randn((b, m, h), dtype=dtype).cuda()
    bk = torch.randn((b, m, d), dtype=dtype).cuda()
    bv = torch.randn((b, m, e), dtype=dtype).cuda()
    o = tpa_q_full_decode_parallel_bh_triton(q, ak, av, bk, bv)
    print(o.shape)
