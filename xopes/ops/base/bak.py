@triton.jit
def _additive_recurrence_fwd(
    Q,
    K,
    V,
    O,
    S,
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
    off_qk = off_bh * n * d
    off_v = off_bh * n * e
    off_o = (off_d * b * h + off_bh) * n * e
    off_d = off_d * BLOCK_D
    off_e = off_e * BLOCK_E

    ##### get block ptr
    q_block_ptr = Q + off_qk + off_d + tl.arange(0, BLOCK_D)
    k_block_ptr = K + off_qk + off_d + tl.arange(0, BLOCK_D)
    v_block_ptr = V + off_v + off_e + tl.arange(0, BLOCK_E)
    o_block_ptr = O + off_o + off_e + tl.arange(0, BLOCK_E)

    mask_d = (off_d + tl.arange(0, BLOCK_D)) < d
    mask_e = (off_e + tl.arange(0, BLOCK_E)) < e

    s = tl.zeros(
        [
            BLOCK_D,
            BLOCK_E,
        ],
        dtype=tl.float32,
    )

    for i in range(n):
        # boundary check on feature dim
        q = tl.load(q_block_ptr, mask=mask_d, other=0).to(tl.float32)
        k = tl.load(k_block_ptr, mask=mask_d, other=0).to(tl.float32)
        v = tl.load(v_block_ptr, mask=mask_e, other=0).to(tl.float32)

        # d 1, 1 e -> d e
        tl.static_print("aaa", k[None, :], v[:, None])
        s += k[:, None] * v[None, :]
        # d 1, d e -> d e
        tl.static_print("aaa", q[:, None], s)
        # d e -> e
        o = q[:, None] * s
        o = tl.sum(o, axis=0)
        tl.static_print("bbb", o, o_block_ptr)
        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask_e)

        q_block_ptr += BLOCK_D
        k_block_ptr += BLOCK_D
        v_block_ptr += BLOCK_E
        o_block_ptr += BLOCK_E
