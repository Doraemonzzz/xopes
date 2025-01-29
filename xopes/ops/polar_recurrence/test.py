@triton.jit
def fused_recurrent_bwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: b_dhead
    # NV: number of split in the V dimension. NK: number of split in the K dimension
    q,  # query [B, H, L, K]
    k,  # key [B, H, L, V]
    v,  # value [B, H, L, V]
    a,  # a [B, H, L, K]
    b,  # b [B, H, L, K]
    ha,  # ha [B, H, L, V]
    dht,  # gradient of final state [B, H, K, V]
    dh0,  # gradient of initial state [B, H, K, V]
    do,  # gradient of output [B, H, L, V]
    dq,  # gradient of query [NV, B, H, L, K]
    dk,  # gradient of key [NV, B, H, L, K]
    dv,  # gradient of value [NK, B, H, L, V]
    da,  # gradient of a [NV, B, H, L, K]
    db,  # gradient of b [NV, B, H, L, K]
    dha,  # gradient of ha [NK, B, H, L, V]
    h0,  # initial state [B, H, K, V]
    scale,  # K ** -0.5
    offsets,  # offsets
    B,  # batch_size
    H,  # n_heads
    T,  # seq_len
    K: tl.constexpr,  # K
    V: tl.constexpr,  # V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state h0
    USE_DH0: tl.constexpr,  # whether to use dh0
    USE_DHT: tl.constexpr,  # whether to use dht
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    dk += i_v * B * H * K * T
    db += i_v * B * H * K * T
    dq += i_v * B * H * K * T
    da += i_v * B * H * K * T
    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(
            tl.int64
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    mask_k = tl.arange(0, BK) < K
    mask_v = (tl.arange(0, BV) + i_v * BV) < V

    q += (i_nh * T * K) if HEAD_FIRST else ((bos * H + i_h) * K)
    k += (i_nh * T * K) if HEAD_FIRST else ((bos * H + i_h) * K)
    v += (i_nh * T * V + i_v * BV) if HEAD_FIRST else ((bos * H + i_h) * V + i_v * BV)
    ha += (i_nh * T * V + i_v * BV) if HEAD_FIRST else ((bos * H + i_h) * V + i_v * BV)
    a += (i_nh * T * K) if HEAD_FIRST else ((bos * H + i_h) * K)
    b += (i_nh * T * K) if HEAD_FIRST else ((bos * H + i_h) * K)
    do += (i_nh * T * V + i_v * BV) if HEAD_FIRST else ((bos * H + i_h) * V + i_v * BV)
    dq += (i_nh * T * K) if HEAD_FIRST else ((bos * H + i_h) * K)
    dk += (i_nh * T * K) if HEAD_FIRST else ((bos * H + i_h) * K)
    dv += (i_nh * T * V + i_v * BV) if HEAD_FIRST else ((bos * H + i_h) * V + i_v * BV)
    da += (i_nh * T * K) if HEAD_FIRST else ((bos * H + i_h) * K)
    db += (i_nh * T * K) if HEAD_FIRST else ((bos * H + i_h) * K)
    dha += (i_nh * T * V + i_v * BV) if HEAD_FIRST else ((bos * H + i_h) * V + i_v * BV)

    p_q = q + tl.arange(0, BK) + (T - 1) * K * (1 if HEAD_FIRST else H)
    p_k = k + tl.arange(0, BK) + (T - 1) * K * (1 if HEAD_FIRST else H)
    p_v = v + tl.arange(0, BV) + (T - 1) * V * (1 if HEAD_FIRST else H)
    p_ha = ha + tl.arange(0, BV) + (T - 1) * V * (1 if HEAD_FIRST else H)
    p_a = a + tl.arange(0, BK) + (T - 1) * K * (1 if HEAD_FIRST else H)
    p_b = b + tl.arange(0, BK) + (T - 1) * K * (1 if HEAD_FIRST else H)
    p_do = do + tl.arange(0, BV) + (T - 1) * V * (1 if HEAD_FIRST else H)
    p_dk = dk + tl.arange(0, BK) + (T - 1) * K * (1 if HEAD_FIRST else H)
    p_dv = dv + tl.arange(0, BV) + (T - 1) * V * (1 if HEAD_FIRST else H)
    p_dha = dha + tl.arange(0, BV) + (T - 1) * V * (1 if HEAD_FIRST else H)
    p_db = db + tl.arange(0, BK) + (T - 1) * K * (1 if HEAD_FIRST else H)
    p_da = da + tl.arange(0, BK) + (T - 1) * K * (1 if HEAD_FIRST else H)
    p_dq = dq + tl.arange(0, BK) + (T - 1) * K * (1 if HEAD_FIRST else H)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_DHT:
        p_ht = (
            dht
            + i_nh * K * V
            + (tl.arange(0, BK)[:, None]) * V
            + ((i_v * BV + tl.arange(0, BV))[None, :])
        )
        b_dh += tl.load(p_ht, mask=mask_k[:, None] & mask_v[None, :], other=0).to(
            tl.float32
        )

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=mask_k, other=0).to(tl.float32)
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_ha = tl.load(p_ha, mask=mask_v, other=0).to(tl.float32)

        b_dh += b_q[:, None] * b_do[None, :]
        d_k = tl.sum(b_dh * b_v[None, :], axis=1)
        d_v = tl.sum(b_dh * b_k[:, None], axis=0)
        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_v)

        b_dha = tl.sum(b_dh * b_b[:, None], axis=0)
        tl.store(p_dha, b_dha.to(p_dha.dtype.element_ty), mask=mask_v)
        b_db = tl.sum(b_dh * b_ha[None, :], axis=1)
        tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=mask_k)

        b_dh += b_dha[None, :] * b_a[:, None]
        p_do -= V if HEAD_FIRST else V * H
        p_q -= K if HEAD_FIRST else K * H
        p_k -= K if HEAD_FIRST else K * H
        p_v -= V if HEAD_FIRST else V * H
        p_dk -= K if HEAD_FIRST else K * H
        p_dv -= V if HEAD_FIRST else V * H
        p_b -= K if HEAD_FIRST else K * H
        p_db -= K if HEAD_FIRST else K * H
        p_a -= K if HEAD_FIRST else K * H
        p_dha -= V if HEAD_FIRST else V * H
        p_ha -= V if HEAD_FIRST else V * H

    if USE_DH0:
        p_dh0 = (
            dh0
            + i_nh * K * V
            + (tl.arange(0, BK)[:, None]) * V
            + (i_v * BV + tl.arange(0, BV)[None, :])
        )
        tl.store(
            p_dh0,
            b_dh.to(p_dh0.dtype.element_ty),
            mask=mask_k[:, None] & mask_v[None, :],
        )

    tl.debug_barrier()

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        mask_kv = mask_k[:, None] & mask_v[None, :]
        p_h0 = (
            h0
            + i_nh * K * V
            + (tl.arange(0, BK)[:, None]) * V
            + ((i_v * BV + tl.arange(0, BV))[None, :])
        )
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    p_k = k + tl.arange(0, BK)
    p_v = v + tl.arange(0, BV)
    p_ha = ha + tl.arange(0, BV)
    p_do = do + tl.arange(0, BV)
    p_dha = dha + tl.arange(0, BV)
    p_da = da + tl.arange(0, BK)
    p_dq = dq + tl.arange(0, BK)
    p_b = b + tl.arange(0, BK)

    for i in range(0, T):
        b_dha = tl.load(p_dha, mask=mask_v, other=0).to(tl.float32)
        d_a = tl.sum(b_dha[None, :] * b_h, axis=1)
        tl.store(p_da, d_a.to(p_da.dtype.element_ty), mask=mask_k)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=mask_k, other=0).to(tl.float32)
        b_ha = tl.load(p_ha, mask=mask_v, other=0).to(tl.float32)
        b_h += b_k[:, None] * b_v[None, :] + b_b[:, None] * b_ha[None, :]
        _d_q = b_h * b_do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_k)

        p_k += K if HEAD_FIRST else K * H
        p_do += V if HEAD_FIRST else V * H
        p_v += V if HEAD_FIRST else V * H
        p_da += K if HEAD_FIRST else K * H
        p_dha += V if HEAD_FIRST else V * H
        p_ha += V if HEAD_FIRST else V * H
        p_dq += K if HEAD_FIRST else K * H
        p_b += K if HEAD_FIRST else K * H
