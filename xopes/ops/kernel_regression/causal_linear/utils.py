import triton
import triton.language as tl

from xopes.utils import generate_configs
from xopes.utils.constant import XOPES_DEBUG

if XOPES_DEBUG:
    BLOCK_D_LIST = [64]
    BLOCK_E_LIST = [32]
else:
    BLOCK_D_LIST = [64, 128]
    BLOCK_E_LIST = [64, 128]


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_D": BLOCK_D_LIST,
        }
    ),
    key=[
        "B",
        "MAX_BLOCK_N",
        "H",
        "D",
    ],
)
@triton.jit
def _krcl_parallel_inverse(
    Q,  # B N H D
    K,  # B N H D
    INV,  # B H NUM_BLOCK_N BLOCK_N BLOCK_N
    LOG_DECAY,  # B N H
    ALPHA,  # B N H
    BETA,  # B N H
    CU_SEQLENS,
    USE_Q: tl.constexpr,  # bool
    USE_ALPHA: tl.constexpr,  # bool
    USE_BETA: tl.constexpr,  # bool
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    REVERSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_BLOCK_N: tl.constexpr,
    MAX_BLOCK_N: tl.constexpr,
):
    NUM_BLOCK_D = tl.cdiv(D, BLOCK_D)
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_block_n = tl.program_id(2)

    offset_block_n = off_block_n * BLOCK_N
    offset_qk = off_b * N * H * D + off_h * D
    offset_ld = off_b * N * H + off_h
    offset_inv = (
        off_b * H * NUM_BLOCK_N * BLOCK_N * BLOCK_N
        + off_h * NUM_BLOCK_N * BLOCK_N * BLOCK_N
        + off_block_n * BLOCK_N * BLOCK_N
    )
    offset_block_ld = offset_block_n * H

    a = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
    array = tl.arange(0, BLOCK_N)
    if USE_Q:
        q_block_ptr = tl.make_block_ptr(
            base=Q + offset_qk,
            shape=(N, D),
            strides=(H * D, 1),
            offsets=(offset_block_n, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

    k_block_ptr = tl.make_block_ptr(
        base=K + offset_qk,
        shape=(N, D),
        strides=(H * D, 1),
        offsets=(offset_block_n, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    inv_block_ptr = tl.make_block_ptr(
        base=INV + offset_inv,
        shape=(BLOCK_N, BLOCK_N),
        strides=(BLOCK_N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_N),
        order=(1, 0),
    )

    if USE_ALPHA:
        alpha_block_ptr = tl.make_block_ptr(
            base=ALPHA + offset_ld,
            shape=(N, 1),
            strides=(H, 1),
            offsets=(offset_block_n, 0),
            block_shape=(BLOCK_N, 1),
            order=(1, 0),
        )
        alpha = tl.load(alpha_block_ptr, boundary_check=(0, 1)).to(tl.float32)

    if USE_BETA:
        beta_trans_block_ptr = tl.make_block_ptr(
            base=BETA + offset_ld,
            shape=(N, 1),
            strides=(H, 1),
            offsets=(offset_block_n, 0),
            block_shape=(BLOCK_N, 1),
            order=(1, 0),
        )
        beta = tl.load(beta_trans_block_ptr, boundary_check=(0, 1)).to(tl.float32)

    for i in range(NUM_BLOCK_D):
        k = tl.load(k_block_ptr, boundary_check=(0, 1))
        if USE_Q:
            q = tl.load(q_block_ptr, boundary_check=(0, 1))
        else:
            q = k
        if USE_ALPHA:
            q = q * alpha
        if USE_BETA:
            k = k * beta

        k_trans = tl.trans(k)

        a += tl.dot(q, k_trans)

        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_D))
        if USE_Q:
            q_block_ptr = tl.advance(q_block_ptr, (0, BLOCK_D))

    # add decay
    ld_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array * H
    mask = (offset_block_n + array) < N
    ld = tl.load(ld_sum_block_ptr, mask=mask, other=0.0).to(tl.float32)
    diff = ld[:, None] - ld[None, :]
    if REVERSE:  # triu
        diff = tl.where(array[:, None] < array[None, :], diff, -float("inf"))
    else:  # tril
        diff = tl.where(array[:, None] > array[None, :], diff, -float("inf"))
    a *= tl.exp(diff)

    a_inv = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
    eye = array[:, None] == array[None, :]
    for i in range(BLOCK_N):
        # C = (I + A) ^ -1, Ck = -A * Ck + I
        a_inv = -tl.dot(a, a_inv) + eye

    tl.store(inv_block_ptr, a_inv.to(inv_block_ptr.dtype.element_ty))


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_D": BLOCK_D_LIST,
            "BLOCK_E": BLOCK_E_LIST,
        }
    ),
    key=[
        "B",
        "MAX_BLOCK_N",
        "H",
        "D",
        "E",
    ],
)
@triton.jit
def _krcl_parallel_chunk_loop(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    O,  # B N H E NUM_BLOCK_D
    INV,  # B H NUM_BLOCK_N BLOCK_N BLOCK_N
    LOG_DECAY,  # B N H
    ALPHA,  # B N H
    BETA,  # B N H
    INITIAL_STATE,  # B H D E
    FINAL_STATE,  # B H D E
    USE_Q: tl.constexpr,
    USE_ALPHA: tl.constexpr,
    USE_BETA: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_PAD: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    REVERSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NUM_BLOCK_N: tl.constexpr,
    MAX_BLOCK_N: tl.constexpr,
):
    tl.cdiv(D, BLOCK_D)
    off_bh = tl.program_id(0)
    off_block_e = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H

    # compute offset
    offset_qk = off_b * N * H * D + off_h * D
    offset_vo = off_b * N * H * E + off_h * E
    offset_ld = off_b * N * H + off_h
    offset_inv = (
        off_b * H * NUM_BLOCK_N * BLOCK_N * BLOCK_N
        + off_h * NUM_BLOCK_N * BLOCK_N * BLOCK_N
    )
    offset_state = off_b * H * D * E + off_h * D * E
    offset_block_e = off_block_e * BLOCK_E

    if REVERSE:
        off_block_n = NUM_BLOCK_N - 1
        stride = -1
        offset_ld_sum = 0
    else:
        off_block_n = 0
        stride = 1
        # last block
        # offset of sum of local log decay, when not reverse, the offset is the last position of the block
        offset_ld_sum = BLOCK_N - 1
    offset_block_n = off_block_n * BLOCK_N

    # compute block ptr
    offset_block_d = 0
    if USE_Q:
        q_block_ptr = tl.make_block_ptr(
            base=Q + offset_qk,
            shape=(N, D),
            strides=(H * D, 1),
            offsets=(offset_block_n, offset_block_d),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

    k_block_ptr = tl.make_block_ptr(
        base=K + offset_qk,
        shape=(N, D),
        strides=(H * D, 1),
        offsets=(offset_block_n, offset_block_d),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    if USE_INITIAL_STATE:
        state_block_ptr = tl.make_block_ptr(
            base=INITIAL_STATE + offset_state,
            shape=(D, E),
            strides=(E, 1),
            offsets=(offset_block_d, offset_block_e),
            block_shape=(BLOCK_D, BLOCK_E),
            order=(1, 0),
        )
        state = tl.load(
            state_block_ptr, boundary_check=(0, 1), padding_option="zero"
        ).to(tl.float32)
    else:
        state = tl.zeros((BLOCK_D, BLOCK_E), dtype=tl.float32)

    final_state_block_ptr = tl.make_block_ptr(
        base=FINAL_STATE + offset_state,
        shape=(D, E),
        strides=(E, 1),
        offsets=(offset_block_d, offset_block_e),
        block_shape=(BLOCK_D, BLOCK_E),
        order=(1, 0),
    )

    if D > BLOCK_D:
        offset_block_d = BLOCK_D
        if USE_Q:
            q1_block_ptr = tl.make_block_ptr(
                base=Q + offset_qk,
                shape=(N, D),
                strides=(H * D, 1),
                offsets=(offset_block_n, offset_block_d),
                block_shape=(BLOCK_N, BLOCK_D),
                order=(1, 0),
            )

        k1_block_ptr = tl.make_block_ptr(
            base=K + offset_qk,
            shape=(N, D),
            strides=(H * D, 1),
            offsets=(offset_block_n, offset_block_d),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

        if USE_INITIAL_STATE:
            state1_block_ptr = tl.make_block_ptr(
                base=INITIAL_STATE + offset_state,
                shape=(D, E),
                strides=(E, 1),
                offsets=(offset_block_d, offset_block_e),
                block_shape=(BLOCK_D, BLOCK_E),
                order=(1, 0),
            )
            state1 = tl.load(
                state1_block_ptr, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32)
        else:
            state1 = tl.zeros((BLOCK_D, BLOCK_E), dtype=tl.float32)

        final_state1_block_ptr = tl.make_block_ptr(
            base=FINAL_STATE + offset_state,
            shape=(D, E),
            strides=(E, 1),
            offsets=(offset_block_d, offset_block_e),
            block_shape=(BLOCK_D, BLOCK_E),
            order=(1, 0),
        )

    v_block_ptr = tl.make_block_ptr(
        base=V + offset_vo,
        shape=(N, E),
        strides=(H * E, 1),
        offsets=(offset_block_n, offset_block_e),
        block_shape=(BLOCK_N, BLOCK_E),
        order=(1, 0),
    )

    o_block_ptr = tl.make_block_ptr(
        base=O + offset_vo,
        shape=(N, E),
        strides=(H * E, 1),
        offsets=(offset_block_n, offset_block_e),
        block_shape=(BLOCK_N, BLOCK_E),
        order=(1, 0),
    )

    if USE_ALPHA:
        alpha_block_ptr = tl.make_block_ptr(
            base=ALPHA + offset_ld,
            shape=(N, 1),
            strides=(H, 1),
            offsets=(offset_block_n, 0),
            block_shape=(BLOCK_N, 1),
            order=(1, 0),
        )

    if USE_BETA:
        beta_trans_block_ptr = tl.make_block_ptr(
            base=BETA + offset_ld,
            shape=(N, 1),
            strides=(H, 1),
            offsets=(offset_block_n, 0),
            block_shape=(BLOCK_N, 1),
            order=(1, 0),
        )

    array = offset_block_n + tl.arange(0, BLOCK_N)
    ld_block_ptr = LOG_DECAY + offset_ld

    for j in range(NUM_BLOCK_N):
        inv_block_ptr = tl.make_block_ptr(
            base=INV + offset_inv + (off_block_n + j * stride) * BLOCK_N * BLOCK_N,
            shape=(BLOCK_N, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_N),
            order=(1, 0),
        )

        k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

        if USE_Q:
            q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        else:
            q = k

        if USE_ALPHA:
            alpha = tl.load(
                alpha_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )
            q = q * alpha

        if USE_BETA:
            beta = tl.load(
                beta_trans_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )
            k = k * beta

        inv = tl.load(inv_block_ptr, boundary_check=(0, 1))

        mask = (array < N) & (array >= 0)
        log_decay = tl.load(ld_block_ptr + array * H, mask=mask, other=0.0).to(
            tl.float32
        )
        log_decay_sum = tl.load(ld_block_ptr + offset_ld_sum * H).to(tl.float32)
        log_k_decay = log_decay_sum - log_decay

        q = (q * tl.exp(log_decay[:, None])).to(q.dtype)
        k = (k * tl.exp(log_k_decay[:, None])).to(k.dtype)
        k_trans = tl.trans(k)
        p = tl.dot(q, state)

        if D > BLOCK_D:
            k1 = tl.load(k1_block_ptr, boundary_check=(0, 1), padding_option="zero")
            if USE_Q:
                q1 = tl.load(q1_block_ptr, boundary_check=(0, 1), padding_option="zero")
            else:
                q1 = k1

            if USE_ALPHA:
                q1 = q1 * alpha

            if USE_BETA:
                k1 = k1 * beta

            q1 = (q1 * tl.exp(log_decay[:, None])).to(q1.dtype)
            k1 = (k1 * tl.exp(log_k_decay[:, None])).to(k1.dtype)
            k1_trans = tl.trans(k1)

            p1 = tl.dot(q1, state1)
            p += p1

        o = tl.dot(inv, v - p)

        state *= tl.exp(log_decay_sum)
        state += tl.dot(k_trans, o)

        if D > BLOCK_D:
            state1 *= tl.exp(log_decay_sum)
            state1 += tl.dot(k1_trans, o)

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), boundary_check=(0, 1))

        # update block ptr
        k_block_ptr = tl.advance(k_block_ptr, (BLOCK_N * stride, 0))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N * stride, 0))
        o_block_ptr = tl.advance(o_block_ptr, (BLOCK_N * stride, 0))
        if USE_Q:
            q_block_ptr = tl.advance(q_block_ptr, (BLOCK_N * stride, 0))
        if USE_ALPHA:
            alpha_block_ptr = tl.advance(alpha_block_ptr, (BLOCK_N * stride, 0))
        if USE_BETA:
            beta_trans_block_ptr = tl.advance(
                beta_trans_block_ptr, (BLOCK_N * stride, 0)
            )
        array += stride * BLOCK_N
        offset_ld_sum += stride * BLOCK_N
        offset_block_n += stride * BLOCK_N
        if REVERSE:
            offset_ld_sum = max(0, offset_ld_sum)
        else:
            offset_ld_sum = min(offset_ld_sum, N - 1)

        if D > BLOCK_D:
            k1_block_ptr = tl.advance(k1_block_ptr, (BLOCK_N * stride, 0))
            if USE_Q:
                q1_block_ptr = tl.advance(q1_block_ptr, (BLOCK_N * stride, 0))

    tl.store(
        final_state_block_ptr,
        state.to(final_state_block_ptr.dtype.element_ty),
        boundary_check=(0, 1),
    )

    if D > BLOCK_D:
        tl.store(
            final_state1_block_ptr,
            state1.to(final_state1_block_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )
