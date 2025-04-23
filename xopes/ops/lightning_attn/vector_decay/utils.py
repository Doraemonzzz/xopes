import triton
import triton.language as tl

from xopes.utils import generate_configs

BLOCK_C = 16
BLOCK_C_LIST = [BLOCK_C]


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": BLOCK_C_LIST,
            "BLOCK_D": [64, 128],
            "BLOCK_E": [64, 128],
        }
    ),
    key=[
        "B",
        "N",
        "H",
        "D",
        "E",
        "USE_CU_SEQLENS",
    ],
)
@triton.jit
def _lavd_parallel_state_parallel(
    K,  # B N H D
    V,  # B N H E
    STATES,  # B H L D E
    LOG_DECAY_K,  # B N H D
    LOG_DECAY_V,  # B N H E
    LOG_DECAY_K_CUMSUM,  # B N H D
    LOG_DECAY_V_CUMSUM,  # B N H E
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_DECAY_K: tl.constexpr,
    USE_DECAY_V: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_PAD: tl.constexpr,
    REVERSE: tl.constexpr,
    SHARE_K: tl.constexpr,
    SHARE_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCK_C = tl.cdiv(BLOCK_N, BLOCK_C)

    off_bhn = tl.program_id(0)
    off_bh = off_bhn // NUM_BLOCK_N
    off_b = off_bh // H
    off_h = off_bh % H
    off_block_n = off_bhn % NUM_BLOCK_N
    off_block_d = tl.program_id(1)
    off_block_e = tl.program_id(2)

    offset_qk = off_b * N * H * D + off_h * D
    offset_vo = off_b * N * H * E + off_h * E
    offset_block_n = off_block_n * BLOCK_N
    offset_block_qk = offset_block_n * H * D
    offset_block_vo = offset_block_n * H * E
    offset_block_d = off_block_d * BLOCK_D
    offset_block_e = off_block_e * BLOCK_E
    offset_state = off_bh * (NUM_BLOCK_N + 1) * D * E

    # compute block ptr and mask
    array_c = tl.arange(0, BLOCK_C)
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)

    # for reverse, local loop start from the last block
    if REVERSE:
        stride = -1
        # BLOCK_N - 1, ... , BLOCK_N - BLOCK_C
        array_c = BLOCK_N - 1 - array_c
        # offset of sum of local log decay, when reverse, the offset is 0
        offset_ldk_sum = 0
        offset_ldv_sum = 0
    else:
        stride = 1
        array_c = array_c
        # last block
        # offset of sum of local log decay, when not reverse, the offset is the last position of the block
        if off_block_n == NUM_BLOCK_N - 1:
            if USE_PAD:
                offset_ldk_sum = (N % BLOCK_N - 1) * H * D
                offset_ldv_sum = (N % BLOCK_N - 1) * H * E
            else:
                offset_ldk_sum = (BLOCK_N - 1) * H * D
                offset_ldv_sum = (BLOCK_N - 1) * H * E
        else:
            offset_ldk_sum = (BLOCK_N - 1) * H * D
            offset_ldv_sum = (BLOCK_N - 1) * H * E

    offset_block_state = off_block_n * D * E

    if SHARE_K:
        k_trans_block_ptr = (
            LOG_DECAY_K
            + offset_qk
            + offset_block_qk
            + array_c[None, :] * H * D
            + (array_d + offset_block_d)[:, None]
        )
    else:
        k_trans_block_ptr = (
            K
            + offset_qk
            + offset_block_qk
            + array_c[None, :] * H * D
            + (array_d + offset_block_d)[:, None]
        )

    if SHARE_V:
        v_block_ptr = (
            LOG_DECAY_V
            + offset_vo
            + offset_block_vo
            + array_c[:, None] * H * E
            + (array_e + offset_block_e)[None, :]
        )
    else:
        v_block_ptr = (
            V
            + offset_vo
            + offset_block_vo
            + array_c[:, None] * H * E
            + (array_e + offset_block_e)[None, :]
        )

    state_block_ptr = (
        STATES
        + offset_state
        + offset_block_state
        + (offset_block_d + array_d[:, None]) * E
        + (offset_block_e + array_e[None, :])
    )

    if USE_DECAY_K:
        ldk_trans_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + array_c[None, :] * H * D
            + (array_d + offset_block_d)[:, None]
        )
        ldk_trans_sum_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + offset_ldk_sum
            + (array_d + offset_block_d)[:, None]
        )
        log_decay_k_trans_sum = tl.load(ldk_trans_sum_block_ptr).to(tl.float32)

    if USE_DECAY_V:
        ldv_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + array_c[:, None] * H * E
            + (array_e + offset_block_e)[None, :]
        )
        ldv_sum_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + offset_ldv_sum
            + (array_e + offset_block_e)[None, :]
        )
        log_decay_v_sum = tl.load(ldv_sum_block_ptr).to(tl.float32)

    mask_d = (array_d + offset_block_d) < D
    mask_e = (array_e + offset_block_e) < E

    state = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)

    cnt = offset_block_n
    for i in range(NUM_BLOCK_C):
        array = offset_block_n + array_c
        mask_c = array < N

        if cnt < N:
            mask_k_trans = mask_c[None, :] & mask_d[:, None]
            mask_v = mask_c[:, None] & mask_e[None, :]

            k_trans = tl.load(k_trans_block_ptr, mask=mask_k_trans, other=0.0)
            if SHARE_K:
                k_trans = 1 - tl.exp(k_trans.to(tl.float32)).to(k_trans.dtype)

            if USE_DECAY_K:
                log_decay_k_trans = tl.load(
                    ldk_trans_block_ptr, mask=mask_k_trans, other=0.0
                ).to(tl.float32)
                log_k_trans_decay = log_decay_k_trans_sum - log_decay_k_trans
                k_trans_decay = tl.exp(log_k_trans_decay)
                k_trans = (k_trans * k_trans_decay).to(k_trans.dtype)

            v = tl.load(v_block_ptr, mask=mask_v, other=0.0)
            if SHARE_V:
                v = 1 - tl.exp(v.to(tl.float32)).to(v.dtype)

            if USE_DECAY_V:
                log_decay_v = tl.load(ldv_block_ptr, mask=mask_v, other=0.0).to(
                    tl.float32
                )
                log_v_decay = log_decay_v_sum - log_decay_v
                v_decay = tl.exp(log_v_decay)
                v = (v * v_decay).to(v.dtype)

            # for local state, since the local decay has been applied, we don't need to apply block_decay
            state += tl.dot(k_trans, v)

        k_trans_block_ptr += BLOCK_C * H * D * stride
        v_block_ptr += BLOCK_C * H * E * stride
        if USE_DECAY_K:
            ldk_trans_block_ptr += BLOCK_C * H * D * stride
        if USE_DECAY_V:
            ldv_block_ptr += BLOCK_C * H * E * stride
        array_c += BLOCK_C * stride
        cnt += BLOCK_C * stride

    tl.store(
        state_block_ptr,
        state.to(state_block_ptr.dtype.element_ty),
        mask=mask_d[:, None] & mask_e[None, :],
    )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_D": [64, 128],
            "BLOCK_E": [64, 128],
        }
    ),
    key=[
        "B",
        "N",
        "H",
        "D",
        "E",
        "USE_CU_SEQLENS",
    ],
)
@triton.jit
def _lavd_parallel_state_reduce(
    STATE,  # B H D E
    STATES,  # B H L D E
    LOG_DECAY_K,  # B N H D
    LOG_DECAY_V,  # B N H E
    LOG_DECAY_K_CUMSUM,  # B N H D
    LOG_DECAY_V_CUMSUM,  # B N H E
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_DECAY_K: tl.constexpr,
    USE_DECAY_V: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_PAD: tl.constexpr,
    REVERSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)

    off_bh = tl.program_id(0)
    off_b = off_bh // H
    off_h = off_bh % H
    off_block_d = tl.program_id(1)
    off_block_e = tl.program_id(2)

    # compute offset
    offset_states = off_bh * (NUM_BLOCK_N + 1) * D * E
    offset_state = off_b * H * D * E + off_h * D * E
    offset_block_d = off_block_d * BLOCK_D
    offset_block_e = off_block_e * BLOCK_E
    offset_ldk = off_b * N * H * D + off_h * D
    offset_ldv = off_b * N * H * E + off_h * E

    # compute array for block ptr
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)

    if REVERSE:
        stride = -1
        states_start = (NUM_BLOCK_N - 1) * D * E
        off_block_n = NUM_BLOCK_N - 1
    else:
        stride = 1
        states_start = 0
        off_block_n = 0

    offset_block_n = off_block_n * BLOCK_N

    states_block_ptr = (
        STATES
        + offset_states
        + states_start
        + (offset_block_d + array_d[:, None]) * E
        + (offset_block_e + array_e[None, :])
    )

    final_states_block_ptr = (
        STATES
        + offset_states
        + NUM_BLOCK_N * D * E
        + (offset_block_d + array_d[:, None]) * E
        + (offset_block_e + array_e[None, :])
    )

    mask_d = (array_d + offset_block_d) < D
    mask_e = (array_e + offset_block_e) < E
    mask = mask_d[:, None] & mask_e[None, :]

    if REVERSE:
        if USE_DECAY_K:
            # !!! important
            first_decay_block_ptr = (
                LOG_DECAY_K + offset_ldk + (array_d + offset_block_d)[:, None]
            )
            ck = tl.exp(tl.load(first_decay_block_ptr).to(tl.float32))

        if USE_DECAY_V:
            # !!! important
            first_decay_block_ptr = (
                LOG_DECAY_V + offset_ldv + (array_e + offset_block_e)[None, :]
            )
            cv = tl.exp(tl.load(first_decay_block_ptr).to(tl.float32))

    # compute
    if USE_INITIAL_STATE:
        state_block_ptr = (
            STATE
            + offset_state
            + (offset_block_d + array_d[:, None]) * E
            + (offset_block_e + array_e[None, :])
        )
        # !!! important
        state = tl.load(state_block_ptr, mask=mask, other=0.0).to(tl.float32)
    else:
        state = tl.zeros((BLOCK_D, BLOCK_E), dtype=tl.float32)

    for i in range(NUM_BLOCK_N):
        current_state = tl.load(states_block_ptr, mask=mask, other=0.0)

        tl.store(
            states_block_ptr, state.to(states_block_ptr.dtype.element_ty), mask=mask
        )

        offset_block_ldk = offset_block_n * H * D
        offset_block_ldv = offset_block_n * H * E

        if USE_DECAY_K:
            if REVERSE:
                # offset of sum of local log decay, when reverse, the offset is 0
                offset_ldk_sum = 0
            else:
                # last block
                # offset of sum of local log decay, when not reverse, the offset is the last position of the block
                if off_block_n == NUM_BLOCK_N - 1:
                    if USE_PAD:
                        offset_ldk_sum = (N % BLOCK_N - 1) * H * D
                    else:
                        offset_ldk_sum = (BLOCK_N - 1) * H * D
                else:
                    offset_ldk_sum = (BLOCK_N - 1) * H * D

            ldk_trans_sum_block_ptr = (
                LOG_DECAY_K_CUMSUM
                + offset_ldk
                + offset_block_ldk
                + offset_ldk_sum
                + (array_d + offset_block_d)[:, None]
            )
            log_decay_k_trans_sum = tl.load(ldk_trans_sum_block_ptr).to(tl.float32)
            state *= tl.exp(log_decay_k_trans_sum)

        if USE_DECAY_V:
            if REVERSE:
                # offset of sum of local log decay, when reverse, the offset is 0
                offset_ldv_sum = 0
            else:
                # last block
                # offset of sum of local log decay, when not reverse, the offset is the last position of the block
                if off_block_n == NUM_BLOCK_N - 1:
                    if USE_PAD:
                        offset_ldv_sum = (N % BLOCK_N - 1) * H * E
                    else:
                        offset_ldv_sum = (BLOCK_N - 1) * H * E
                else:
                    offset_ldv_sum = (BLOCK_N - 1) * H * E

            ldv_sum_block_ptr = (
                LOG_DECAY_V_CUMSUM
                + offset_ldv
                + offset_block_ldv
                + offset_ldv_sum
                + (array_e + offset_block_e)[None, :]
            )
            log_decay_v_sum = tl.load(ldv_sum_block_ptr).to(tl.float32)
            state *= tl.exp(log_decay_v_sum)

        state += current_state

        states_block_ptr += D * E * stride
        offset_block_n += BLOCK_N * stride
        off_block_n += stride  # !!! important

    # !!! important
    if REVERSE:
        if USE_DECAY_K:
            state *= ck
        if USE_DECAY_V:
            state *= cv

    tl.store(
        final_states_block_ptr,
        state.to(final_states_block_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": BLOCK_C_LIST,
            "BLOCK_D": [64, 128],
            "BLOCK_E": [64, 128],
        }
    ),
    key=[
        "B",
        "N",
        "H",
        "D",
        "E",
        "USE_CU_SEQLENS",
    ],
)
@triton.jit
def _lavd_parallel_state_parallel_reduce(
    K,  # B N H D
    V,  # B N H E
    STATE,  # B H D E
    STATES,  # B H L D E
    LOG_DECAY_K,  # B N H D
    LOG_DECAY_V,  # B N H E
    LOG_DECAY_K_CUMSUM,  # B N H D
    LOG_DECAY_V_CUMSUM,  # B N H E
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_DECAY_K: tl.constexpr,
    USE_DECAY_V: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_PAD: tl.constexpr,
    REVERSE: tl.constexpr,
    SHARE_K: tl.constexpr,
    SHARE_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCK_C = tl.cdiv(BLOCK_N, BLOCK_C)

    off_bh = tl.program_id(0)
    off_b = off_bh // H
    off_h = off_bh % H
    off_block_d = tl.program_id(1)
    off_block_e = tl.program_id(2)

    # compute offset
    offset_states = off_bh * (NUM_BLOCK_N + 1) * D * E
    offset_state = off_b * H * D * E + off_h * D * E
    offset_block_d = off_block_d * BLOCK_D
    offset_block_e = off_block_e * BLOCK_E
    offset_ldk = off_b * N * H * D + off_h * D
    offset_ldv = off_b * N * H * E + off_h * E

    offset_qk = off_b * N * H * D + off_h * D
    offset_vo = off_b * N * H * E + off_h * E
    if REVERSE:
        off_block_n = NUM_BLOCK_N - 1
    else:
        off_block_n = 0
    offset_block_n = off_block_n * BLOCK_N

    # compute array for block ptr
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)

    if REVERSE:
        stride = -1
        states_start = (NUM_BLOCK_N - 1) * D * E
        # last chunk's first element
        offset_ldk_sum = (NUM_BLOCK_N - 1) * BLOCK_N * H * D
        offset_ldv_sum = (NUM_BLOCK_N - 1) * BLOCK_N * H * E
    else:
        stride = 1
        states_start = 0
        # first chunk's last element
        offset_ldk_sum = (BLOCK_N - 1) * H * D
        offset_ldv_sum = (BLOCK_N - 1) * H * E

    states_block_ptr = (
        STATES
        + offset_states
        + states_start
        + (offset_block_d + array_d[:, None]) * E
        + (offset_block_e + array_e[None, :])
    )

    final_states_block_ptr = (
        STATES
        + offset_states
        + NUM_BLOCK_N * D * E
        + (offset_block_d + array_d[:, None]) * E
        + (offset_block_e + array_e[None, :])
    )

    mask_d = (array_d + offset_block_d) < D
    mask_e = (array_e + offset_block_e) < E
    mask = mask_d[:, None] & mask_e[None, :]

    if REVERSE:
        if USE_DECAY_K:
            # !!! important
            first_decay_block_ptr = (
                LOG_DECAY_K + offset_ldk + (array_d + offset_block_d)[:, None]
            )
            ck = tl.exp(tl.load(first_decay_block_ptr).to(tl.float32))

        if USE_DECAY_V:
            # !!! important
            first_decay_block_ptr = (
                LOG_DECAY_V + offset_ldv + (array_e + offset_block_e)[None, :]
            )
            cv = tl.exp(tl.load(first_decay_block_ptr).to(tl.float32))

    # compute
    if USE_INITIAL_STATE:
        state_block_ptr = (
            STATE
            + offset_state
            + (offset_block_d + array_d[:, None]) * E
            + (offset_block_e + array_e[None, :])
        )
        # !!! important
        state = tl.load(state_block_ptr, mask=mask, other=0.0).to(tl.float32)
    else:
        state = tl.zeros((BLOCK_D, BLOCK_E), dtype=tl.float32)

    for i in range(NUM_BLOCK_N):
        tl.store(
            states_block_ptr, state.to(states_block_ptr.dtype.element_ty), mask=mask
        )

        cnt = offset_block_n
        offset_block_qk = offset_block_n * H * D
        offset_block_vo = offset_block_n * H * E
        offset_block_ldk = offset_block_qk
        offset_block_ldv = offset_block_vo
        if REVERSE:
            array_c = BLOCK_N - 1 - tl.arange(0, BLOCK_C)
        else:
            array_c = tl.arange(0, BLOCK_C)

        ##### update global state
        if USE_DECAY_K:
            if REVERSE:
                # offset of sum of local log decay, when reverse, the offset is 0
                offset_ldk_sum = 0
            else:
                # last block
                # offset of sum of local log decay, when not reverse, the offset is the last position of the block
                if off_block_n == NUM_BLOCK_N - 1:
                    if USE_PAD:
                        offset_ldk_sum = (N % BLOCK_N - 1) * H * D
                    else:
                        offset_ldk_sum = (BLOCK_N - 1) * H * D
                else:
                    offset_ldk_sum = (BLOCK_N - 1) * H * D

            ldk_trans_sum_block_ptr = (
                LOG_DECAY_K_CUMSUM
                + offset_ldk
                + offset_block_ldk
                + offset_ldk_sum
                + (array_d + offset_block_d)[:, None]
            )
            log_decay_k_trans_sum = tl.load(ldk_trans_sum_block_ptr).to(tl.float32)
            state *= tl.exp(log_decay_k_trans_sum)

        if USE_DECAY_V:
            if REVERSE:
                # offset of sum of local log decay, when reverse, the offset is 0
                offset_ldv_sum = 0
            else:
                # last block
                # offset of sum of local log decay, when not reverse, the offset is the last position of the block
                if off_block_n == NUM_BLOCK_N - 1:
                    if USE_PAD:
                        offset_ldv_sum = (N % BLOCK_N - 1) * H * E
                    else:
                        offset_ldv_sum = (BLOCK_N - 1) * H * E
                else:
                    offset_ldv_sum = (BLOCK_N - 1) * H * E

            ldv_sum_block_ptr = (
                LOG_DECAY_V_CUMSUM
                + offset_ldv
                + offset_block_ldv
                + offset_ldv_sum
                + (array_e + offset_block_e)[None, :]
            )
            log_decay_v_sum = tl.load(ldv_sum_block_ptr).to(tl.float32)
            state *= tl.exp(log_decay_v_sum)

        ##### compute local state
        if SHARE_K:
            k_trans_block_ptr = (
                LOG_DECAY_K
                + offset_qk
                + offset_block_qk
                + array_c[None, :] * H * D
                + (array_d + offset_block_d)[:, None]
            )
        else:
            k_trans_block_ptr = (
                K
                + offset_qk
                + offset_block_qk
                + array_c[None, :] * H * D
                + (array_d + offset_block_d)[:, None]
            )

        if SHARE_V:
            v_block_ptr = (
                LOG_DECAY_V
                + offset_vo
                + offset_block_vo
                + array_c[:, None] * H * E
                + (array_e + offset_block_e)[None, :]
            )
        else:
            v_block_ptr = (
                V
                + offset_vo
                + offset_block_vo
                + array_c[:, None] * H * E
                + (array_e + offset_block_e)[None, :]
            )

        if USE_DECAY_K:
            ldk_trans_block_ptr = (
                LOG_DECAY_K_CUMSUM
                + offset_qk
                + offset_block_qk
                + array_c[None, :] * H * D
                + (array_d + offset_block_d)[:, None]
            )
            ldk_trans_sum_block_ptr = (
                LOG_DECAY_K_CUMSUM
                + offset_qk
                + offset_block_qk
                + offset_ldk_sum
                + (array_d + offset_block_d)[:, None]
            )
            log_decay_k_trans_sum = tl.load(ldk_trans_sum_block_ptr).to(tl.float32)

        if USE_DECAY_V:
            ldv_block_ptr = (
                LOG_DECAY_V_CUMSUM
                + offset_vo
                + offset_block_vo
                + array_c[:, None] * H * E
                + (array_e + offset_block_e)[None, :]
            )
            ldv_sum_block_ptr = (
                LOG_DECAY_V_CUMSUM
                + offset_vo
                + offset_block_vo
                + offset_ldv_sum
                + (array_e + offset_block_e)[None, :]
            )
            log_decay_v_sum = tl.load(ldv_sum_block_ptr).to(tl.float32)

        for j in range(NUM_BLOCK_C):
            array = offset_block_n + array_c
            mask_c = array < N

            if cnt < N:
                mask_k_trans = mask_c[None, :] & mask_d[:, None]
                mask_v = mask_c[:, None] & mask_e[None, :]

                k_trans = tl.load(k_trans_block_ptr, mask=mask_k_trans, other=0.0)
                if SHARE_K:
                    k_trans = 1 - tl.exp(k_trans.to(tl.float32)).to(k_trans.dtype)

                if USE_DECAY_K:
                    log_decay_k_trans = tl.load(
                        ldk_trans_block_ptr, mask=mask_k_trans, other=0.0
                    ).to(tl.float32)
                    log_k_trans_decay = log_decay_k_trans_sum - log_decay_k_trans
                    k_trans_decay = tl.exp(log_k_trans_decay)
                    k_trans = (k_trans * k_trans_decay).to(k_trans.dtype)

                v = tl.load(v_block_ptr, mask=mask_v, other=0.0)
                if SHARE_V:
                    v = 1 - tl.exp(v.to(tl.float32)).to(v.dtype)

                if USE_DECAY_V:
                    log_decay_v = tl.load(ldv_block_ptr, mask=mask_v, other=0.0).to(
                        tl.float32
                    )
                    log_v_decay = log_decay_v_sum - log_decay_v
                    v_decay = tl.exp(log_v_decay)
                    v = (v * v_decay).to(v.dtype)

                # for local state, since the local decay has been applied, we don't need to apply block_decay
                state += tl.dot(k_trans, v)

            k_trans_block_ptr += BLOCK_C * H * D * stride
            v_block_ptr += BLOCK_C * H * E * stride
            if USE_DECAY_K:
                ldk_trans_block_ptr += BLOCK_C * H * D * stride
            if USE_DECAY_V:
                ldv_block_ptr += BLOCK_C * H * E * stride
            array_c += BLOCK_C * stride
            cnt += BLOCK_C * stride

        states_block_ptr += D * E * stride
        offset_block_n += BLOCK_N * stride
        off_block_n += stride  # !!! important

    # !!! important
    if REVERSE:
        if USE_DECAY_K:
            state *= ck
        if USE_DECAY_V:
            state *= cv

    tl.store(
        final_states_block_ptr,
        state.to(final_states_block_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": BLOCK_C_LIST,
            "BLOCK_D": [64, 128],
            "BLOCK_E": [64, 128],
        }
    ),
    key=["B", "N", "H", "D", "E", "USE_CU_SEQLENS", "USE_DECAY_K", "USE_DECAY_V"],
)
@triton.jit
def _lavd_parallel_intra(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    O,  # B N H E
    LOG_DECAY_K,  # B N H D
    LOG_DECAY_V,  # B N H E
    LOG_DECAY_K_CUMSUM,  # B N H D
    LOG_DECAY_V_CUMSUM,  # B N H E
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_DECAY_K: tl.constexpr,
    USE_DECAY_V: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_PAD: tl.constexpr,
    REVERSE: tl.constexpr,
    SHARE_K: tl.constexpr,
    SHARE_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCK_D = tl.cdiv(D, BLOCK_D)
    tl.cdiv(BLOCK_N, BLOCK_C)

    off_bhn = tl.program_id(0)
    off_bh = off_bhn // NUM_BLOCK_N
    off_b = off_bh // H
    off_h = off_bh % H
    off_block_n = off_bhn % NUM_BLOCK_N
    off_block_c = tl.program_id(1)
    off_block_e = tl.program_id(2)

    # compute offset
    offset_qk = off_b * N * H * D + off_h * D
    offset_vo = off_b * N * H * E + off_h * E
    offset_block_n = off_block_n * BLOCK_N
    offset_block_qk = offset_block_n * H * D
    offset_block_vo = offset_block_n * H * E
    offset_block_c = off_block_c * BLOCK_C
    offset_block_e = off_block_e * BLOCK_E

    array_c = tl.arange(0, BLOCK_C)
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    array_n = tl.arange(0, BLOCK_N)

    # compute mask
    array_kv = array_n
    mask_c = (offset_block_n + offset_block_c + array_c) < N
    mask_n = (offset_block_n + array_n) < N
    mask_e = (offset_block_e + array_e) < E

    # compute block ptr
    q_block_ptr = (
        Q
        + offset_qk
        + offset_block_qk
        + (offset_block_c + array_c[:, None]) * H * D
        + array_d[None, :]
    )

    M = offset_block_c
    if REVERSE:
        M = M + BLOCK_C

        offset_ld = offset_block_c + BLOCK_C
        offset_ldk_sum = offset_ld * H * D
        offset_ldv_sum = offset_ld * H * E

        mask_kv = (array_kv >= M) & mask_n
        ld_sum_mask = (offset_block_n + offset_ld) < N
    else:
        # if off_block_c = 0, no sub intra is needed
        if off_block_n == NUM_BLOCK_N - 1:
            if USE_PAD:
                if offset_block_n + offset_block_c < N:
                    offset_ld = offset_block_c - 1
                else:
                    offset_ld = N % BLOCK_N - 1
            else:
                offset_ld = offset_block_c - 1
        else:
            offset_ld = offset_block_c - 1

        offset_ldk_sum = offset_ld * H * D
        offset_ldv_sum = offset_ld * H * E

        mask_kv = (array_kv < M) & mask_n
        ld_sum_mask = offset_ld >= 0

    if USE_DECAY_K:
        ldk_trans_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + array_kv[None, :] * H * D
            + array_d[:, None]
        )
        ldk_trans_sum_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + offset_ldk_sum
            + array_d[:, None]
        )

        ldk_sub_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + (offset_block_c + array_c[:, None]) * H * D
            + array_d[None, :]
        )

        if REVERSE:
            offset_ldk_start = offset_block_c + BLOCK_C
            mask_ldk_start = (offset_ldk_start < BLOCK_N) & (
                (offset_block_n + offset_ldk_start) < N
            )
        else:
            offset_ldk_start = offset_block_c - 1
            mask_ldk_start = (offset_ldk_start >= 0) & (
                (offset_block_n + offset_ldk_start) < N
            )

        ldk_start_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + offset_ldk_start * H * D
            + array_d[None, :]
        )

    o_block_ptr = (
        O
        + offset_vo
        + offset_block_vo
        + (offset_block_c + array_c[:, None]) * H * E
        + (offset_block_e + array_e[None, :])
    )

    if USE_DECAY_V:
        ldv_sub_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + (offset_block_c + array_c[:, None]) * H * E
            + (offset_block_e + array_e[None, :])
        )

        if REVERSE:
            offset_ldv_start = offset_block_c + BLOCK_C
            mask_ldv_start = (offset_ldv_start < BLOCK_N) & (
                (offset_block_n + offset_ldv_start) < N
            )
        else:
            offset_ldv_start = offset_block_c - 1
            mask_ldv_start = (offset_ldv_start >= 0) & (
                (offset_block_n + offset_ldv_start) < N
            )

        ldv_start_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + offset_ldv_start * H * E
            + (offset_block_e + array_e[None, :])
        )

        log_decay_v_sub = tl.load(
            ldv_sub_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)
        log_decay_v_start = tl.load(
            ldv_start_block_ptr, mask=mask_ldv_start & mask_e[None, :], other=0.0
        ).to(tl.float32)
        tl.exp(log_decay_v_sub - log_decay_v_start)

    o = tl.zeros([BLOCK_C, BLOCK_E], dtype=tl.float32)

    if SHARE_K:
        k_start = LOG_DECAY_K
    else:
        k_start = K

    if SHARE_V:
        v_start = LOG_DECAY_V
    else:
        v_start = V

    v_block_ptr = (
        v_start
        + offset_vo
        + offset_block_vo
        + array_kv[:, None] * H * E
        + (offset_block_e + array_e)[None, :]
    )

    v = tl.load(v_block_ptr, mask=mask_kv[:, None] & mask_e[None, :], other=0.0)

    if SHARE_V:
        v = 1 - tl.exp(v.to(tl.float32)).to(v.dtype)

    if USE_DECAY_V:
        ldv_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + array_kv[:, None] * H * E
            + (offset_block_e + array_e)[None, :]
        )
        ldv_sum_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + offset_ldv_sum
            + (offset_block_e + array_e)[None, :]
        )
        log_decay_v_sum = tl.load(
            ldv_sum_block_ptr, mask=ld_sum_mask & mask_e[None, :]
        ).to(tl.float32)

        log_decay_v = tl.load(
            ldv_block_ptr, mask=mask_kv[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)
        log_decay_v = log_decay_v_sum - log_decay_v
        v_decay = tl.exp(log_decay_v)
        v = (v * v_decay).to(v.dtype)

    for i in range(NUM_BLOCK_D):
        mask_d = (i * BLOCK_D + array_d) < D

        ##### start sub intra part, use loop to compute
        if REVERSE:  # if reverse, start from the last element of the chunk
            array_kv_elem = offset_block_c + BLOCK_C - 1 + tl.arange(0, 1)
            stride_elem = -1
        else:
            array_kv_elem = offset_block_c + tl.arange(0, 1)
            stride_elem = 1

        q_trans_elem_block_ptr = (
            Q
            + offset_qk
            + offset_block_qk
            + array_kv_elem[None, :] * H * D
            + (i * BLOCK_D + array_d)[:, None]
        )

        k_trans_elem_block_ptr = (
            k_start
            + offset_qk
            + offset_block_qk
            + array_kv_elem[None, :] * H * D
            + (i * BLOCK_D + array_d)[:, None]
        )

        v_elem_block_ptr = (
            v_start
            + offset_vo
            + offset_block_vo
            + array_kv_elem[:, None] * H * E
            + (offset_block_e + array_e)[None, :]
        )

        if USE_DECAY_K:
            ldk_trans_elem_block_ptr = (
                LOG_DECAY_K
                + offset_qk
                + offset_block_qk
                + array_kv_elem[None, :] * H * D
                + (i * BLOCK_D + array_d)[:, None]
            )
            log_decay_k_trans_elem = tl.zeros(
                [BLOCK_D, 1], dtype=tl.float32
            )  # for reverse use

        if USE_DECAY_V:
            ldv_elem_block_ptr = (
                LOG_DECAY_V
                + offset_vo
                + offset_block_vo
                + array_kv_elem[:, None] * H * E
                + (offset_block_e + array_e)[None, :]
            )
            log_decay_v_elem = tl.zeros(
                [1, BLOCK_E], dtype=tl.float32
            )  # for reverse use

        state_sub_intra = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)
        index_array = tl.arange(0, BLOCK_C)
        for j in range(BLOCK_C):
            mask_elem_c = (offset_block_n + array_kv_elem) < N

            if USE_DECAY_K:
                if not REVERSE:
                    log_decay_k_trans_elem = tl.load(
                        ldk_trans_elem_block_ptr,
                        mask=mask_elem_c[None, :] & mask_d[:, None],
                        other=0.0,
                    ).to(tl.float32)
                state_sub_intra *= tl.exp(log_decay_k_trans_elem)

            if USE_DECAY_V:
                if not REVERSE:
                    log_decay_v_elem = tl.load(
                        ldv_elem_block_ptr,
                        mask=mask_elem_c[:, None] & mask_e[None, :],
                        other=0.0,
                    ).to(tl.float32)
                state_sub_intra *= tl.exp(log_decay_v_elem)

            k_trans_elem = tl.load(
                k_trans_elem_block_ptr,
                mask=mask_elem_c[None, :] & mask_d[:, None],
                other=0.0,
            )
            if SHARE_K:
                k_trans_elem = 1 - tl.exp(k_trans_elem.to(tl.float32)).to(
                    k_trans_elem.dtype
                )

            v_elem = tl.load(
                v_elem_block_ptr, mask=mask_elem_c[:, None] & mask_e[None, :], other=0.0
            )
            if SHARE_V:
                v_elem = 1 - tl.exp(v_elem.to(tl.float32)).to(v_elem.dtype)

            state_sub_intra += k_trans_elem * v_elem

            q_trans_elem = tl.load(
                q_trans_elem_block_ptr,
                mask=mask_elem_c[None, :] & mask_d[:, None],
                other=0.0,
            )
            # BLOCK_D 1, BLOCK_D BLOCK_E -> 1 BLOCK_E
            o_sub_intra = tl.sum(q_trans_elem * state_sub_intra, axis=0, keep_dims=True)
            if REVERSE:
                mask_array = index_array == (BLOCK_C - 1 - j)
            else:
                mask_array = index_array == j
            # BLOCK_C, 1
            o_intra = tl.where(mask_array[:, None], o_sub_intra, 0.0)
            o += o_intra

            q_trans_elem_block_ptr += stride_elem * H * D
            k_trans_elem_block_ptr += stride_elem * H * D
            v_elem_block_ptr += stride_elem * H * E
            array_kv_elem += stride_elem

            if USE_DECAY_K:
                if REVERSE:
                    log_decay_k_trans_elem = tl.load(
                        ldk_trans_elem_block_ptr,
                        mask=mask_elem_c[None, :] & mask_d[:, None],
                        other=0.0,
                    ).to(tl.float32)
                ldk_trans_elem_block_ptr += stride_elem * H * D

            if USE_DECAY_V:
                if REVERSE:
                    log_decay_v_elem = tl.load(
                        ldv_elem_block_ptr,
                        mask=mask_elem_c[:, None] & mask_e[None, :],
                        other=0.0,
                    ).to(tl.float32)
                ldv_elem_block_ptr += stride_elem * H * E
        ##### end sub intra part

        ##### start sub inter part
        q = tl.load(q_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)

        k_trans_block_ptr = (
            k_start
            + offset_qk
            + offset_block_qk
            + array_kv[None, :] * H * D
            + (i * BLOCK_D + array_d)[:, None]
        )

        k_trans = tl.load(
            k_trans_block_ptr, mask=mask_kv[None, :] & mask_d[:, None], other=0.0
        )

        if SHARE_K:
            k_trans = 1 - tl.exp(k_trans.to(tl.float32)).to(k_trans.dtype)

        if USE_DECAY_K:
            log_decay_k_trans_sum = tl.load(
                ldk_trans_sum_block_ptr, mask=ld_sum_mask & mask_d[:, None]
            ).to(tl.float32)

            log_decay_k_trans = tl.load(
                ldk_trans_block_ptr, mask=mask_kv[None, :] & mask_d[:, None], other=0.0
            ).to(tl.float32)
            log_decay_k_trans = log_decay_k_trans_sum - log_decay_k_trans
            k_trans_decay = tl.exp(log_decay_k_trans)
            k_trans = (k_trans * k_trans_decay).to(k_trans.dtype)

            # sub inter decay
            log_decay_k_sub = tl.load(
                ldk_sub_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)
            log_decay_k_start = tl.load(
                ldk_start_block_ptr, mask=mask_ldk_start & mask_d[None, :], other=0.0
            ).to(tl.float32)
            k_decay_sub = tl.exp(log_decay_k_sub - log_decay_k_start)
            q = (q * k_decay_sub).to(q.dtype)

        state = tl.dot(k_trans, v).to(q.dtype)
        o_inter = tl.dot(q, state)

        # sub inter decay
        if USE_DECAY_V:
            o_inter = (o_inter * v_decay_sub).to(o_inter.dtype)

        o += o_inter
        ##### end sub inter part

        # !!! important
        tl.debug_barrier()

        q_block_ptr += BLOCK_D
        if USE_DECAY_K:
            ldk_trans_sum_block_ptr += BLOCK_D
            ldk_trans_block_ptr += BLOCK_D
            ldk_sub_block_ptr += BLOCK_D
            ldk_start_block_ptr += BLOCK_D

    tl.store(
        o_block_ptr,
        o.to(o_block_ptr.dtype.element_ty),
        mask=mask_c[:, None] & mask_e[None, :],
    )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": BLOCK_C_LIST,
            "BLOCK_D": [64, 128],
            "BLOCK_E": [64, 128],
        }
    ),
    key=[
        "B",
        "N",
        "H",
        "D",
        "E",
        "USE_CU_SEQLENS",
    ],
)
@triton.jit
def _lavd_parallel_inter(
    Q,  # B N H D
    O,  # B N H E
    STATES,  # B H L D E if not trans_states, B H L E D if trans_states
    LOG_DECAY_K_CUMSUM,  # B N H D
    LOG_DECAY_V_CUMSUM,  # B N H E
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_DECAY_K: tl.constexpr,
    USE_DECAY_V: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    REVERSE: tl.constexpr,
    TRANS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCK_D = tl.cdiv(D, BLOCK_D)

    off_bhn = tl.program_id(0)
    off_bh = off_bhn // NUM_BLOCK_N
    off_b = off_bh // H
    off_h = off_bh % H
    off_block_n = off_bhn % NUM_BLOCK_N
    off_block_c = tl.program_id(1)
    off_block_e = tl.program_id(2)

    # compute offset
    offset_qk = off_b * N * H * D + off_h * D
    offset_vo = off_b * N * H * E + off_h * E
    offset_ldk = offset_qk
    offset_ldv = offset_vo
    offset_block_n = off_block_n * BLOCK_N
    offset_block_qk = offset_block_n * H * D
    offset_block_vo = offset_block_n * H * E
    offset_block_ldk = offset_block_qk
    offset_block_ldv = offset_block_vo
    offset_block_c = off_block_c * BLOCK_C
    offset_block_e = off_block_e * BLOCK_E

    offset_state = off_bh * (NUM_BLOCK_N + 1) * D * E
    offset_block_state = off_block_n * D * E

    # compute block ptr and mask
    array_e = tl.arange(0, BLOCK_E)
    array_d = tl.arange(0, BLOCK_D)
    array_c = tl.arange(0, BLOCK_C)
    q_block_ptr = (
        Q
        + offset_qk
        + offset_block_qk
        + (offset_block_c + array_c[:, None]) * H * D
        + array_d[None, :]
    )

    o_block_ptr = (
        O
        + offset_vo
        + offset_block_vo
        + (offset_block_c + array_c[:, None]) * H * E
        + (offset_block_e + array_e[None, :])
    )
    if TRANS:
        # if trans_states, the states are stored in the shape of B H L E D, the shape we need to load is D E
        state_block_ptr = (
            STATES
            + offset_state
            + offset_block_state
            + array_d[:, None]
            + (offset_block_e + array_e)[None, :] * D
        )
    else:
        state_block_ptr = (
            STATES
            + offset_state
            + offset_block_state
            + array_d[:, None] * E
            + (offset_block_e + array_e)[None, :]
        )

    mask_e = (offset_block_e + array_e) < E
    mask_c = (offset_block_n + offset_block_c + array_c) < N
    mask_v = mask_c[:, None] & mask_e[None, :]

    if USE_DECAY_K:
        ldk_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_ldk
            + offset_block_ldk
            + (offset_block_c + array_c[:, None]) * H * D
            + array_d[None, :]
        )

    o = tl.load(o_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0).to(
        tl.float32
    )
    for i in range(NUM_BLOCK_D):
        mask_d = (array_d + i * BLOCK_D) < D
        mask_q = mask_c[:, None] & mask_d[None, :]
        mask_de = mask_d[:, None] & mask_e[None, :]

        q = tl.load(q_block_ptr, mask=mask_q, other=0.0)

        if USE_DECAY_K:
            ldk = tl.load(ldk_block_ptr, mask=mask_q, other=0.0).to(tl.float32)
            q = (q * tl.exp(ldk)).to(q.dtype)

        state = tl.load(state_block_ptr, mask=mask_de, other=0.0).to(q.dtype)

        o_ = tl.dot(q, state)
        o += o_

        q_block_ptr += BLOCK_D

        if USE_DECAY_K:
            ldk_block_ptr += BLOCK_D

        if TRANS:
            state_block_ptr += BLOCK_D
        else:
            state_block_ptr += BLOCK_D * E

    if USE_DECAY_V:
        ldv_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_ldv
            + offset_block_ldv
            + (offset_block_c + array_c[:, None]) * H * E
            + (offset_block_e + array_e[None, :])
        )
        ldv = tl.load(
            ldv_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)
        o = (o * tl.exp(ldv)).to(o.dtype)

    tl.store(
        o_block_ptr,
        o.to(o_block_ptr.dtype.element_ty),
        mask=mask_c[:, None] & mask_e[None, :],
    )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": BLOCK_C_LIST,
            "BLOCK_D": [64, 128],
            "BLOCK_E": [64, 128],
        }
    ),
    key=["B", "N", "H", "D", "E", "USE_CU_SEQLENS", "USE_DECAY_K", "USE_DECAY_V"],
)
@triton.jit
def _lavd_parallel_intra_inter(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    O,  # B N H E
    STATES,  # B H L D E if not trans_states, B H L E D if trans_states
    LOG_DECAY_K,  # B N H D
    LOG_DECAY_V,  # B N H E
    LOG_DECAY_K_CUMSUM,  # B N H D
    LOG_DECAY_V_CUMSUM,  # B N H E
    X,  # B N H E
    DLOG_DECAY,  # B N H E
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_DECAY_K: tl.constexpr,
    USE_DECAY_V: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_PAD: tl.constexpr,
    COMPUTE_DLD: tl.constexpr,
    REVERSE: tl.constexpr,
    TRANS: tl.constexpr,
    SHARE_Q: tl.constexpr,
    SHARE_K: tl.constexpr,
    SHARE_V: tl.constexpr,
    SHARE_X: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCK_D = tl.cdiv(D, BLOCK_D)
    tl.cdiv(BLOCK_N, BLOCK_C)

    off_bhn = tl.program_id(0)
    off_bh = off_bhn // NUM_BLOCK_N
    off_b = off_bh // H
    off_h = off_bh % H
    off_block_n = off_bhn % NUM_BLOCK_N
    off_block_c = tl.program_id(1)
    off_block_e = tl.program_id(2)

    # compute offset
    offset_qk = off_b * N * H * D + off_h * D
    offset_vo = off_b * N * H * E + off_h * E
    offset_block_n = off_block_n * BLOCK_N
    offset_block_qk = offset_block_n * H * D
    offset_block_vo = offset_block_n * H * E
    offset_block_c = off_block_c * BLOCK_C
    offset_block_e = off_block_e * BLOCK_E

    offset_state = off_bh * (NUM_BLOCK_N + 1) * D * E
    offset_block_state = off_block_n * D * E

    array_c = tl.arange(0, BLOCK_C)
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    array_n = tl.arange(0, BLOCK_N)

    # compute mask
    array_kv = array_n
    mask_c = (offset_block_n + offset_block_c + array_c) < N
    mask_n = (offset_block_n + array_n) < N
    mask_e = (offset_block_e + array_e) < E

    if SHARE_Q:
        q_start = LOG_DECAY_K
    else:
        q_start = Q

    if SHARE_K:
        k_start = LOG_DECAY_K
    else:
        k_start = K

    if SHARE_V:
        v_start = LOG_DECAY_V
    else:
        v_start = V

    # compute block ptr
    q_block_ptr = (
        q_start
        + offset_qk
        + offset_block_qk
        + (offset_block_c + array_c[:, None]) * H * D
        + array_d[None, :]
    )

    M = offset_block_c
    if REVERSE:
        M = M + BLOCK_C

        offset_ld = offset_block_c + BLOCK_C
        offset_ldk_sum = offset_ld * H * D
        offset_ldv_sum = offset_ld * H * E

        mask_kv = (array_kv >= M) & mask_n
        ld_sum_mask = (offset_block_n + offset_ld) < N
    else:
        # if off_block_c = 0, no sub intra is needed
        if off_block_n == NUM_BLOCK_N - 1:
            if USE_PAD:
                if offset_block_n + offset_block_c < N:
                    offset_ld = offset_block_c - 1
                else:
                    offset_ld = N % BLOCK_N - 1
            else:
                offset_ld = offset_block_c - 1
        else:
            offset_ld = offset_block_c - 1

        offset_ldk_sum = offset_ld * H * D
        offset_ldv_sum = offset_ld * H * E

        mask_kv = (array_kv < M) & mask_n
        ld_sum_mask = offset_ld >= 0

    if USE_DECAY_K:
        ldk_trans_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + array_kv[None, :] * H * D
            + array_d[:, None]
        )
        ldk_trans_sum_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + offset_ldk_sum
            + array_d[:, None]
        )

        ldk_sub_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + (offset_block_c + array_c[:, None]) * H * D
            + array_d[None, :]
        )

        if REVERSE:
            offset_ldk_start = offset_block_c + BLOCK_C
            mask_ldk_start = (offset_ldk_start < BLOCK_N) & (
                (offset_block_n + offset_ldk_start) < N
            )
        else:
            offset_ldk_start = offset_block_c - 1
            mask_ldk_start = (offset_ldk_start >= 0) & (
                (offset_block_n + offset_ldk_start) < N
            )

        ldk_start_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + offset_ldk_start * H * D
            + array_d[None, :]
        )

        ##### for inter
        ldk_inter_block_ptr = ldk_sub_block_ptr

    o_block_ptr = (
        O
        + offset_vo
        + offset_block_vo
        + (offset_block_c + array_c[:, None]) * H * E
        + (offset_block_e + array_e[None, :])
    )

    if USE_DECAY_V:
        ldv_sub_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + (offset_block_c + array_c[:, None]) * H * E
            + (offset_block_e + array_e[None, :])
        )

        if REVERSE:
            offset_ldv_start = offset_block_c + BLOCK_C
            mask_ldv_start = (offset_ldv_start < BLOCK_N) & (
                (offset_block_n + offset_ldv_start) < N
            )
        else:
            offset_ldv_start = offset_block_c - 1
            mask_ldv_start = (offset_ldv_start >= 0) & (
                (offset_block_n + offset_ldv_start) < N
            )

        ldv_start_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + offset_ldv_start * H * E
            + (offset_block_e + array_e[None, :])
        )

        log_decay_v_sub = tl.load(
            ldv_sub_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)
        log_decay_v_start = tl.load(
            ldv_start_block_ptr, mask=mask_ldv_start & mask_e[None, :], other=0.0
        ).to(tl.float32)
        v_decay_sub = tl.exp(log_decay_v_sub - log_decay_v_start)

        ##### for inter
        ldv_inter_block_ptr = ldv_sub_block_ptr

        ldv_inter = tl.load(
            ldv_inter_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)

    o = tl.zeros([BLOCK_C, BLOCK_E], dtype=tl.float32)

    v_block_ptr = (
        v_start
        + offset_vo
        + offset_block_vo
        + array_kv[:, None] * H * E
        + (offset_block_e + array_e)[None, :]
    )

    v = tl.load(v_block_ptr, mask=mask_kv[:, None] & mask_e[None, :], other=0.0)

    if SHARE_V:
        v = 1 - tl.exp(v.to(tl.float32)).to(v.dtype)

    if USE_DECAY_V:
        ldv_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + array_kv[:, None] * H * E
            + (offset_block_e + array_e)[None, :]
        )
        ldv_sum_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + offset_ldv_sum
            + (offset_block_e + array_e)[None, :]
        )
        log_decay_v_sum = tl.load(
            ldv_sum_block_ptr, mask=ld_sum_mask & mask_e[None, :]
        ).to(tl.float32)

        log_decay_v = tl.load(
            ldv_block_ptr, mask=mask_kv[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)
        log_decay_v = log_decay_v_sum - log_decay_v
        v_decay = tl.exp(log_decay_v)
        v = (v * v_decay).to(v.dtype)

    if TRANS:
        # if trans_states, the states are stored in the shape of B H L E D, the shape we need to load is D E
        state_block_ptr = (
            STATES
            + offset_state
            + offset_block_state
            + array_d[:, None]
            + (offset_block_e + array_e)[None, :] * D
        )
    else:
        state_block_ptr = (
            STATES
            + offset_state
            + offset_block_state
            + array_d[:, None] * E
            + (offset_block_e + array_e)[None, :]
        )

    for i in range(NUM_BLOCK_D):
        mask_d = (i * BLOCK_D + array_d) < D
        mask_de = mask_d[:, None] & mask_e[None, :]

        ##### start sub intra part, use loop to compute
        if REVERSE:  # if reverse, start from the last element of the chunk
            array_kv_elem = offset_block_c + BLOCK_C - 1 + tl.arange(0, 1)
            stride_elem = -1
        else:
            array_kv_elem = offset_block_c + tl.arange(0, 1)
            stride_elem = 1

        q_trans_elem_block_ptr = (
            q_start
            + offset_qk
            + offset_block_qk
            + array_kv_elem[None, :] * H * D
            + (i * BLOCK_D + array_d)[:, None]
        )

        k_trans_elem_block_ptr = (
            k_start
            + offset_qk
            + offset_block_qk
            + array_kv_elem[None, :] * H * D
            + (i * BLOCK_D + array_d)[:, None]
        )

        v_elem_block_ptr = (
            v_start
            + offset_vo
            + offset_block_vo
            + array_kv_elem[:, None] * H * E
            + (offset_block_e + array_e)[None, :]
        )

        if USE_DECAY_K:
            ldk_trans_elem_block_ptr = (
                LOG_DECAY_K
                + offset_qk
                + offset_block_qk
                + array_kv_elem[None, :] * H * D
                + (i * BLOCK_D + array_d)[:, None]
            )
            log_decay_k_trans_elem = tl.zeros(
                [BLOCK_D, 1], dtype=tl.float32
            )  # for reverse use

        if USE_DECAY_V:
            ldv_elem_block_ptr = (
                LOG_DECAY_V
                + offset_vo
                + offset_block_vo
                + array_kv_elem[:, None] * H * E
                + (offset_block_e + array_e)[None, :]
            )
            log_decay_v_elem = tl.zeros(
                [1, BLOCK_E], dtype=tl.float32
            )  # for reverse use

        state_sub_intra = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)
        index_array = tl.arange(0, BLOCK_C)
        for j in range(BLOCK_C):
            mask_elem_c = (offset_block_n + array_kv_elem) < N

            if USE_DECAY_K:
                if not REVERSE:
                    log_decay_k_trans_elem = tl.load(
                        ldk_trans_elem_block_ptr,
                        mask=mask_elem_c[None, :] & mask_d[:, None],
                        other=0.0,
                    ).to(tl.float32)
                state_sub_intra *= tl.exp(log_decay_k_trans_elem)

            if USE_DECAY_V:
                if not REVERSE:
                    log_decay_v_elem = tl.load(
                        ldv_elem_block_ptr,
                        mask=mask_elem_c[:, None] & mask_e[None, :],
                        other=0.0,
                    ).to(tl.float32)
                state_sub_intra *= tl.exp(log_decay_v_elem)

            k_trans_elem = tl.load(
                k_trans_elem_block_ptr,
                mask=mask_elem_c[None, :] & mask_d[:, None],
                other=0.0,
            )
            if SHARE_K:
                k_trans_elem = 1 - tl.exp(k_trans_elem.to(tl.float32)).to(
                    k_trans_elem.dtype
                )

            v_elem = tl.load(
                v_elem_block_ptr, mask=mask_elem_c[:, None] & mask_e[None, :], other=0.0
            )
            if SHARE_V:
                v_elem = 1 - tl.exp(v_elem.to(tl.float32)).to(v_elem.dtype)

            state_sub_intra += k_trans_elem * v_elem

            q_trans_elem = tl.load(
                q_trans_elem_block_ptr,
                mask=mask_elem_c[None, :] & mask_d[:, None],
                other=0.0,
            )
            if SHARE_Q:
                q_trans_elem = 1 - tl.exp(q_trans_elem.to(tl.float32)).to(
                    q_trans_elem.dtype
                )
            # BLOCK_D 1, BLOCK_D BLOCK_E -> 1 BLOCK_E
            o_sub_intra = tl.sum(q_trans_elem * state_sub_intra, axis=0, keep_dims=True)
            if REVERSE:
                mask_array = index_array == (BLOCK_C - 1 - j)
            else:
                mask_array = index_array == j
            BLOCK_C, 1
            o_intra = tl.where(mask_array[:, None], o_sub_intra, 0.0)
            o += o_intra  # too slow, update this later

            q_trans_elem_block_ptr += stride_elem * H * D
            k_trans_elem_block_ptr += stride_elem * H * D
            v_elem_block_ptr += stride_elem * H * E
            array_kv_elem += stride_elem

            if USE_DECAY_K:
                if REVERSE:
                    log_decay_k_trans_elem = tl.load(
                        ldk_trans_elem_block_ptr,
                        mask=mask_elem_c[None, :] & mask_d[:, None],
                        other=0.0,
                    ).to(tl.float32)
                ldk_trans_elem_block_ptr += stride_elem * H * D

            if USE_DECAY_V:
                if REVERSE:
                    log_decay_v_elem = tl.load(
                        ldv_elem_block_ptr,
                        mask=mask_elem_c[:, None] & mask_e[None, :],
                        other=0.0,
                    ).to(tl.float32)
                ldv_elem_block_ptr += stride_elem * H * E
        ##### end sub intra part

        ##### start sub inter part
        q = tl.load(q_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)

        if SHARE_Q:
            q = 1 - tl.exp(q.to(tl.float32)).to(q.dtype)

        k_trans_block_ptr = (
            k_start
            + offset_qk
            + offset_block_qk
            + array_kv[None, :] * H * D
            + (i * BLOCK_D + array_d)[:, None]
        )

        k_trans = tl.load(
            k_trans_block_ptr, mask=mask_kv[None, :] & mask_d[:, None], other=0.0
        )

        if SHARE_K:
            k_trans = 1 - tl.exp(k_trans.to(tl.float32)).to(k_trans.dtype)

        if USE_DECAY_K:
            log_decay_k_trans_sum = tl.load(
                ldk_trans_sum_block_ptr, mask=ld_sum_mask & mask_d[:, None]
            ).to(tl.float32)

            log_decay_k_trans = tl.load(
                ldk_trans_block_ptr,
                mask=mask_kv[None, :] & mask_d[:, None],
                other=0.0,
            ).to(tl.float32)
            log_decay_k_trans = log_decay_k_trans_sum - log_decay_k_trans
            k_trans_decay = tl.exp(log_decay_k_trans)
            k_trans = (k_trans * k_trans_decay).to(k_trans.dtype)

            # sub inter decay
            log_decay_k_sub = tl.load(
                ldk_sub_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)
            log_decay_k_start = tl.load(
                ldk_start_block_ptr,
                mask=mask_ldk_start & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            k_decay_sub = tl.exp(log_decay_k_sub - log_decay_k_start)
            state = tl.dot(k_trans, v).to(q.dtype)
            o_inter = tl.dot((q * k_decay_sub).to(q.dtype), state)
        else:
            state = tl.dot(k_trans, v).to(q.dtype)
            o_inter = tl.dot(q, state)

        # sub inter decay
        if USE_DECAY_V:
            o_inter = (o_inter * v_decay_sub).to(o_inter.dtype)

        o += o_inter
        ##### end sub inter part

        ##### start inter part
        if USE_DECAY_K:
            ldk_inter = tl.load(
                ldk_inter_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)
            dk_inter = tl.exp(ldk_inter)
            q = (q * dk_inter).to(q.dtype)

        state_ = tl.load(state_block_ptr, mask=mask_de, other=0.0).to(q.dtype)

        o_inter_ = tl.dot(q, state_)

        if USE_DECAY_V:
            o_inter_ = (o_inter_ * tl.exp(ldv_inter)).to(o_inter_.dtype)

        o += o_inter_
        ##### end inter part

        q_block_ptr += BLOCK_D
        if USE_DECAY_K:
            ldk_trans_block_ptr += BLOCK_D
            ldk_trans_sum_block_ptr += BLOCK_D
            ldk_inter_block_ptr += BLOCK_D
            ldk_sub_block_ptr += BLOCK_D
            ldk_start_block_ptr += BLOCK_D

        if TRANS:
            state_block_ptr += BLOCK_D
        else:
            state_block_ptr += BLOCK_D * E

    tl.store(
        o_block_ptr,
        o.to(o_block_ptr.dtype.element_ty),
        mask=mask_c[:, None] & mask_e[None, :],
    )

    if COMPUTE_DLD:
        if SHARE_X:
            x_start = LOG_DECAY_V
        else:
            x_start = X

        x_block_ptr = (
            x_start
            + offset_vo
            + offset_block_vo
            + (offset_block_c + array_c[:, None]) * H * E
            + (offset_block_e + array_e[None, :])
        )
        x = tl.load(x_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0)
        if SHARE_X:
            x = 1 - tl.exp(x.to(tl.float32)).to(x.dtype)
        # N E
        dld = x * o

        offset_dld = off_b * N * H * E + off_h * E
        offset_block_dld = offset_block_n * H * E
        dld_block_ptr = (
            DLOG_DECAY
            + offset_dld
            + offset_block_dld
            + (offset_block_c + array_c[:, None]) * H * E
            + (offset_block_e + array_e[None, :])
        )
        tl.store(
            dld_block_ptr,
            dld.to(dld_block_ptr.dtype.element_ty),
            mask=mask_c[:, None] & mask_e[None, :],
        )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": BLOCK_C_LIST,
            "BLOCK_D": [64, 128],
            "BLOCK_E": [64, 128],
        }
    ),
    key=["B", "N", "H", "D", "E", "USE_CU_SEQLENS", "USE_DECAY_K", "USE_DECAY_V"],
)
@triton.jit
def _lavd_parallel_intra_inter_no_loop(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    O,  # B N H E
    A,  # B H NUM_ATTN_MATRIX BLOCK_C BLOCK_C
    STATES,  # B H L D E if not trans_states, B H L E D if trans_states
    LOG_DECAY_K,  # B N H D
    LOG_DECAY_V,  # B N H E
    LOG_DECAY_K_CUMSUM,  # B N H D
    LOG_DECAY_V_CUMSUM,  # B N H E
    X,  # B N H E
    DLOG_DECAY,  # B N H E
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_DECAY_K: tl.constexpr,
    USE_DECAY_V: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_PAD: tl.constexpr,
    COMPUTE_DLD: tl.constexpr,
    REVERSE: tl.constexpr,
    TRANS_STATE: tl.constexpr,
    TRANS_A: tl.constexpr,
    SHARE_Q: tl.constexpr,
    SHARE_K: tl.constexpr,
    SHARE_V: tl.constexpr,
    SHARE_X: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCK_D = tl.cdiv(D, BLOCK_D)
    tl.cdiv(BLOCK_N, BLOCK_C)

    off_bhn = tl.program_id(0)
    off_bh = off_bhn // NUM_BLOCK_N
    off_b = off_bh // H
    off_h = off_bh % H
    off_block_n = off_bhn % NUM_BLOCK_N
    off_block_c = tl.program_id(1)
    off_block_e = tl.program_id(2)

    # compute offset
    offset_qk = off_b * N * H * D + off_h * D
    offset_vo = off_b * N * H * E + off_h * E
    offset_block_n = off_block_n * BLOCK_N
    offset_block_qk = offset_block_n * H * D
    offset_block_vo = offset_block_n * H * E
    offset_block_c = off_block_c * BLOCK_C
    offset_block_e = off_block_e * BLOCK_E

    offset_state = off_bh * (NUM_BLOCK_N + 1) * D * E
    offset_block_state = off_block_n * D * E

    array_c = tl.arange(0, BLOCK_C)
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    array_n = tl.arange(0, BLOCK_N)

    # compute mask
    array_kv = array_n
    mask_c = (offset_block_n + offset_block_c + array_c) < N
    mask_n = (offset_block_n + array_n) < N
    mask_e = (offset_block_e + array_e) < E

    if SHARE_Q:
        q_start = LOG_DECAY_K
    else:
        q_start = Q

    if SHARE_K:
        k_start = LOG_DECAY_K
    else:
        k_start = K

    if SHARE_V:
        v_start = LOG_DECAY_V
    else:
        v_start = V

    # compute block ptr
    q_block_ptr = (
        q_start
        + offset_qk
        + offset_block_qk
        + (offset_block_c + array_c[:, None]) * H * D
        + array_d[None, :]
    )

    M = offset_block_c
    if REVERSE:
        M = M + BLOCK_C

        offset_ld = offset_block_c + BLOCK_C
        offset_ldk_sum = offset_ld * H * D
        offset_ldv_sum = offset_ld * H * E

        mask_kv = (array_kv >= M) & mask_n
        ld_sum_mask = (offset_block_n + offset_ld) < N
    else:
        # if off_block_c = 0, no sub intra is needed
        if off_block_n == NUM_BLOCK_N - 1:
            if USE_PAD:
                if offset_block_n + offset_block_c < N:
                    offset_ld = offset_block_c - 1
                else:
                    offset_ld = N % BLOCK_N - 1
            else:
                offset_ld = offset_block_c - 1
        else:
            offset_ld = offset_block_c - 1

        offset_ldk_sum = offset_ld * H * D
        offset_ldv_sum = offset_ld * H * E

        mask_kv = (array_kv < M) & mask_n
        ld_sum_mask = offset_ld >= 0

    if USE_DECAY_K:
        ldk_trans_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + array_kv[None, :] * H * D
            + array_d[:, None]
        )
        ldk_trans_sum_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + offset_ldk_sum
            + array_d[:, None]
        )

        ldk_sub_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + (offset_block_c + array_c[:, None]) * H * D
            + array_d[None, :]
        )

        if REVERSE:
            offset_ldk_start = offset_block_c + BLOCK_C
            mask_ldk_start = (offset_ldk_start < BLOCK_N) & (
                (offset_block_n + offset_ldk_start) < N
            )
        else:
            offset_ldk_start = offset_block_c - 1
            mask_ldk_start = (offset_ldk_start >= 0) & (
                (offset_block_n + offset_ldk_start) < N
            )

        ldk_start_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + offset_ldk_start * H * D
            + array_d[None, :]
        )

        ##### for inter
        ldk_inter_block_ptr = ldk_sub_block_ptr

    o_block_ptr = (
        O
        + offset_vo
        + offset_block_vo
        + (offset_block_c + array_c[:, None]) * H * E
        + (offset_block_e + array_e[None, :])
    )

    if USE_DECAY_V:
        ldv_sub_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + (offset_block_c + array_c[:, None]) * H * E
            + (offset_block_e + array_e[None, :])
        )

        if REVERSE:
            offset_ldv_start = offset_block_c + BLOCK_C
            mask_ldv_start = (offset_ldv_start < BLOCK_N) & (
                (offset_block_n + offset_ldv_start) < N
            )
        else:
            offset_ldv_start = offset_block_c - 1
            mask_ldv_start = (offset_ldv_start >= 0) & (
                (offset_block_n + offset_ldv_start) < N
            )

        ldv_start_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + offset_ldv_start * H * E
            + (offset_block_e + array_e[None, :])
        )

        log_decay_v_sub = tl.load(
            ldv_sub_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)
        log_decay_v_start = tl.load(
            ldv_start_block_ptr, mask=mask_ldv_start & mask_e[None, :], other=0.0
        ).to(tl.float32)
        v_decay_sub = tl.exp(log_decay_v_sub - log_decay_v_start)

        ##### for inter
        ldv_inter_block_ptr = ldv_sub_block_ptr

        ldv_inter = tl.load(
            ldv_inter_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)

    o = tl.zeros([BLOCK_C, BLOCK_E], dtype=tl.float32)

    v_block_ptr = (
        v_start
        + offset_vo
        + offset_block_vo
        + array_kv[:, None] * H * E
        + (offset_block_e + array_e)[None, :]
    )

    v = tl.load(v_block_ptr, mask=mask_kv[:, None] & mask_e[None, :], other=0.0)

    if SHARE_V:
        v = 1 - tl.exp(v.to(tl.float32)).to(v.dtype)

    if USE_DECAY_V:
        ldv_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + array_kv[:, None] * H * E
            + (offset_block_e + array_e)[None, :]
        )
        ldv_sum_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + offset_ldv_sum
            + (offset_block_e + array_e)[None, :]
        )
        log_decay_v_sum = tl.load(
            ldv_sum_block_ptr, mask=ld_sum_mask & mask_e[None, :]
        ).to(tl.float32)

        log_decay_v = tl.load(
            ldv_block_ptr, mask=mask_kv[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)
        log_decay_v = log_decay_v_sum - log_decay_v
        v_decay = tl.exp(log_decay_v)
        v = (v * v_decay).to(v.dtype)

    if TRANS_STATE:
        # if trans_states, the states are stored in the shape of B H L E D, the shape we need to load is D E
        state_block_ptr = (
            STATES
            + offset_state
            + offset_block_state
            + array_d[:, None]
            + (offset_block_e + array_e)[None, :] * D
        )
    else:
        state_block_ptr = (
            STATES
            + offset_state
            + offset_block_state
            + array_d[:, None] * E
            + (offset_block_e + array_e)[None, :]
        )

    ##### sub intra part
    k_sub_intra_block_ptr = (
        k_start
        + offset_qk
        + offset_block_qk
        + (offset_block_c + array_c[:, None]) * H * D
        + array_d[None, :]
    )

    v_sub_intra_block_ptr = (
        v_start
        + offset_vo
        + offset_block_vo
        + (offset_block_c + array_c[:, None]) * H * E
        + (offset_block_e + array_e)[None, :]
    )

    if USE_DECAY_V:
        ldv_sub_intra_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + (offset_block_c + array_c[:, None]) * H * E
            + (offset_block_e + array_e)[None, :]
        )

    if REVERSE:
        mask_a = (array_c[:, None] <= array_c[None, :])[:, :, None]
    else:
        mask_a = (array_c[:, None] >= array_c[None, :])[:, :, None]

    for i in range(NUM_BLOCK_D):
        mask_d = (i * BLOCK_D + array_d) < D
        mask_de = mask_d[:, None] & mask_e[None, :]

        q = tl.load(q_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)

        if SHARE_Q:
            q = 1 - tl.exp(q.to(tl.float32)).to(q.dtype)

        ##### start sub inter part
        k_trans_block_ptr = (
            k_start
            + offset_qk
            + offset_block_qk
            + array_kv[None, :] * H * D
            + (i * BLOCK_D + array_d)[:, None]
        )

        k_trans = tl.load(
            k_trans_block_ptr, mask=mask_kv[None, :] & mask_d[:, None], other=0.0
        )

        if SHARE_K:
            k_trans = 1 - tl.exp(k_trans.to(tl.float32)).to(k_trans.dtype)

        if USE_DECAY_K:
            log_decay_k_trans_sum = tl.load(
                ldk_trans_sum_block_ptr, mask=ld_sum_mask & mask_d[:, None]
            ).to(tl.float32)

            log_decay_k_trans = tl.load(
                ldk_trans_block_ptr,
                mask=mask_kv[None, :] & mask_d[:, None],
                other=0.0,
            ).to(tl.float32)
            log_decay_k_trans = log_decay_k_trans_sum - log_decay_k_trans
            k_trans_decay = tl.exp(log_decay_k_trans)
            k_trans = (k_trans * k_trans_decay).to(k_trans.dtype)

            # sub inter decay
            log_decay_k_sub = tl.load(
                ldk_sub_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)
            log_decay_k_start = tl.load(
                ldk_start_block_ptr,
                mask=mask_ldk_start & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            k_decay_sub = tl.exp(log_decay_k_sub - log_decay_k_start)
            state = tl.dot(k_trans, v).to(q.dtype)
            o_inter = tl.dot((q * k_decay_sub).to(q.dtype), state)
        else:
            state = tl.dot(k_trans, v).to(q.dtype)
            o_inter = tl.dot(q, state)

        # sub inter decay
        if USE_DECAY_V:
            o_inter = (o_inter * v_decay_sub).to(o_inter.dtype)

        o += o_inter
        ##### end sub inter part

        ##### start inter part
        if USE_DECAY_K:
            ldk_inter = tl.load(
                ldk_inter_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)
            dk_inter = tl.exp(ldk_inter)
            q = (q * dk_inter).to(q.dtype)

        state_ = tl.load(state_block_ptr, mask=mask_de, other=0.0).to(q.dtype)

        o_inter_ = tl.dot(q, state_)

        if USE_DECAY_V:
            o_inter_ = (o_inter_ * tl.exp(ldv_inter)).to(o_inter_.dtype)

        o += o_inter_
        ##### end inter part

        q_block_ptr += BLOCK_D
        if USE_DECAY_K:
            ldk_trans_block_ptr += BLOCK_D
            ldk_trans_sum_block_ptr += BLOCK_D
            ldk_inter_block_ptr += BLOCK_D
            ldk_sub_block_ptr += BLOCK_D
            ldk_start_block_ptr += BLOCK_D

        if TRANS_STATE:
            state_block_ptr += BLOCK_D
        else:
            state_block_ptr += BLOCK_D * E

    ##### start sub intra part
    NUM_ATTN_MATRIX = tl.cdiv(N, BLOCK_C)
    NUM_BLOCK_C = tl.cdiv(BLOCK_N, BLOCK_C)
    off_c = off_block_n * NUM_BLOCK_C + off_block_c
    offset_a = (
        off_b * H * NUM_ATTN_MATRIX * BLOCK_C * BLOCK_C
        + off_h * NUM_ATTN_MATRIX * BLOCK_C * BLOCK_C
        + off_c * BLOCK_C * BLOCK_C
    )
    mask_a_start = off_c < NUM_ATTN_MATRIX
    if TRANS_A:
        a_block_ptr = A + offset_a + array_c[None, :] * BLOCK_C + array_c[:, None]
    else:
        a_block_ptr = A + offset_a + array_c[:, None] * BLOCK_C + array_c[None, :]

    a = tl.load(a_block_ptr, mask=mask_a_start).to(tl.float32)

    v_sub_intra_block_ptr = (
        v_start
        + offset_vo
        + offset_block_vo
        + (offset_block_c + array_c[:, None]) * H * E
        + (offset_block_e + array_e)[None, :]
    )

    if USE_DECAY_V:
        ldv_sub_intra_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + (offset_block_c + array_c[:, None]) * H * E
            + (offset_block_e + array_e)[None, :]
        )

    v_sub_intra = tl.load(
        v_sub_intra_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0
    )
    if SHARE_V:
        v_sub_intra = 1 - tl.exp(v_sub_intra.to(tl.float32)).to(v_sub_intra.dtype)

    if not USE_DECAY_V:
        o += tl.dot(a.to(v_sub_intra.dtype), v_sub_intra)
    else:
        ld_vo_sub_intra = tl.load(
            ldv_sub_intra_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)
        ld_vo_diff_sub_intra = ld_vo_sub_intra[:, None, :] - ld_vo_sub_intra[None, :, :]
        decay_vo_diff_sub_intra = tl.exp(ld_vo_diff_sub_intra)
        a_ = a[:, :, None] * decay_vo_diff_sub_intra
        a_ = tl.where(mask_a, a_, 0.0)
        o += tl.sum(a_ * v_sub_intra[None, :, :], axis=1)
    #### end sub intra part

    tl.store(
        o_block_ptr,
        o.to(o_block_ptr.dtype.element_ty),
        mask=mask_c[:, None] & mask_e[None, :],
    )

    if COMPUTE_DLD:
        if SHARE_X:
            x_start = LOG_DECAY_V
        else:
            x_start = X

        x_block_ptr = (
            x_start
            + offset_vo
            + offset_block_vo
            + (offset_block_c + array_c[:, None]) * H * E
            + (offset_block_e + array_e[None, :])
        )
        x = tl.load(x_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0)
        if SHARE_X:
            x = 1 - tl.exp(x.to(tl.float32)).to(x.dtype)
        # N E
        dld = x * o

        offset_dld = off_b * N * H * E + off_h * E
        offset_block_dld = offset_block_n * H * E
        dld_block_ptr = (
            DLOG_DECAY
            + offset_dld
            + offset_block_dld
            + (offset_block_c + array_c[:, None]) * H * E
            + (offset_block_e + array_e[None, :])
        )
        tl.store(
            dld_block_ptr,
            dld.to(dld_block_ptr.dtype.element_ty),
            mask=mask_c[:, None] & mask_e[None, :],
        )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": BLOCK_C_LIST,
            "BLOCK_D": [64, 128],
            "BLOCK_E": [64, 128],
        }
    ),
    key=["B", "N", "H", "D", "E", "USE_CU_SEQLENS", "USE_DECAY_K", "USE_DECAY_V"],
)
@triton.jit
def _lavd_parallel_sub_intra(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    O,  # B N H E
    LOG_DECAY_K,  # B N H D
    LOG_DECAY_V,  # B N H E
    LOG_DECAY_K_CUMSUM,  # B N H D
    LOG_DECAY_V_CUMSUM,  # B N H E
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_DECAY_K: tl.constexpr,
    USE_DECAY_V: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_PAD: tl.constexpr,
    REVERSE: tl.constexpr,
    SHARE_K: tl.constexpr,
    SHARE_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    tl.cdiv(N, BLOCK_N)
    NUM_BLOCK_D = tl.cdiv(D, BLOCK_D)

    off_bh = tl.program_id(0)
    off_b = off_bh // H
    off_h = off_bh % H
    off_c = tl.program_id(1)
    off_block_e = tl.program_id(2)

    # compute offset
    offset_qk = off_b * N * H * D + off_h * D
    offset_vo = off_b * N * H * E + off_h * E
    offset_block_c = off_c * BLOCK_C
    offset_block_e = off_block_e * BLOCK_E

    array_c = tl.arange(0, BLOCK_C)
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)

    # compute mask
    mask_c = (offset_block_c + array_c) < N
    mask_e = (offset_block_e + array_e) < E

    # compute block ptr
    q_block_ptr = (
        Q + offset_qk + (offset_block_c + array_c[:, None]) * H * D + array_d[None, :]
    )

    if SHARE_K:
        k_start = LOG_DECAY_K
    else:
        k_start = K

    k_block_ptr = (
        k_start
        + offset_qk
        + (offset_block_c + array_c[:, None]) * H * D
        + array_d[None, :]
    )

    if USE_DECAY_K:
        ld_qk_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + (offset_block_c + array_c[:, None]) * H * D
            + array_d[None, :]
        )

    if SHARE_V:
        v_start = LOG_DECAY_V
    else:
        v_start = V

    if USE_DECAY_V:
        ldv_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + (offset_block_c + array_c[:, None]) * H * E
            + (offset_block_e + array_e)[None, :]
        )

    v_block_ptr = (
        v_start
        + offset_vo
        + (offset_block_c + array_c[:, None]) * H * E
        + (offset_block_e + array_e)[None, :]
    )

    o_block_ptr = (
        O
        + offset_vo
        + (offset_block_c + array_c[:, None]) * H * E
        + (offset_block_e + array_e)[None, :]
    )

    if REVERSE:
        mask_a = (array_c[:, None] <= array_c[None, :])[:, :, None]
    else:
        mask_a = (array_c[:, None] >= array_c[None, :])[:, :, None]

    # attention matrix
    a = tl.zeros([BLOCK_C, BLOCK_C], dtype=tl.float32)
    for i in range(NUM_BLOCK_D):
        mask_d = (i * BLOCK_D + array_d) < D

        q = tl.load(q_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)
        k = tl.load(k_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)

        if SHARE_K:
            k = 1 - tl.exp(k.to(tl.float32)).to(k.dtype)

        # BLOCK_C BLOCK_D, BLOCK_C BLOCK_D -> BLOCK_C BLOCK_C BLOCK_D
        score = q[:, None, :] * k[None, :, :]

        if USE_DECAY_K:
            ld_qk = tl.load(
                ld_qk_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)
            ld_qk_diff = ld_qk[:, None, :] - ld_qk[None, :, :]
            decay_qk_diff = tl.exp(ld_qk_diff)
            score *= decay_qk_diff

        score = tl.where(mask_a, score, 0.0)
        a += tl.sum(score, axis=-1)

        q_block_ptr += BLOCK_D
        k_block_ptr += BLOCK_D
        if USE_DECAY_K:
            ld_qk_block_ptr += BLOCK_D

    v = tl.load(v_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0).to(
        tl.float32
    )
    if SHARE_V:
        v = 1 - tl.exp(v.to(tl.float32)).to(v.dtype)

    if not USE_DECAY_V:
        o = tl.dot(a, v)
    else:
        ld_vo = tl.load(
            ldv_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)
        ld_vo_diff = ld_vo[:, None, :] - ld_vo[None, :, :]
        decay_vo_diff = tl.exp(ld_vo_diff)
        a_ = a[:, :, None] * decay_vo_diff
        a_ = tl.where(mask_a, a_, 0.0)
        o = tl.sum(a_ * v[None, :, :], axis=1)

    tl.store(
        o_block_ptr,
        o.to(o_block_ptr.dtype.element_ty),
        mask=mask_c[:, None] & mask_e[None, :],
    )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_D": [64, 128],
            "BLOCK_E": [64, 128],
        }
    ),
    key=["B", "N", "H", "D", "E", "USE_CU_SEQLENS", "USE_DECAY_K", "USE_DECAY_V"],
)
@triton.jit
def _lavd_parallel_sub_intra_attn(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    A,  # B H NUM_ATTN_MATRIX BLOCK_C BLOCK_C
    LOG_DECAY_K,  # B N H D
    LOG_DECAY_V,  # B N H E
    LOG_DECAY_K_CUMSUM,  # B N H D
    LOG_DECAY_V_CUMSUM,  # B N H E
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_DECAY_K: tl.constexpr,
    USE_DECAY_V: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_PAD: tl.constexpr,
    REVERSE: tl.constexpr,
    SHARE_Q: tl.constexpr,
    SHARE_K: tl.constexpr,
    SHARE_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    NUM_ATTN_MATRIX = tl.cdiv(N, BLOCK_C)
    NUM_BLOCK_D = tl.cdiv(D, BLOCK_D)
    tl.cdiv(E, BLOCK_E)

    off_bh = tl.program_id(0)
    off_b = off_bh // H
    off_h = off_bh % H
    off_c = tl.program_id(1)

    # compute offset
    offset_qk = off_b * N * H * D + off_h * D
    off_b * N * H * E + off_h * E
    offset_block_c = off_c * BLOCK_C
    offset_a = (
        off_b * H * NUM_ATTN_MATRIX * BLOCK_C * BLOCK_C
        + off_h * NUM_ATTN_MATRIX * BLOCK_C * BLOCK_C
        + off_c * BLOCK_C * BLOCK_C
    )

    array_c = tl.arange(0, BLOCK_C)
    array_d = tl.arange(0, BLOCK_D)
    tl.arange(0, BLOCK_E)
    # compute mask
    mask_c = (offset_block_c + array_c) < N

    # compute block ptr
    if SHARE_Q:
        q_start = LOG_DECAY_K
    else:
        q_start = Q

    if SHARE_K:
        k_start = LOG_DECAY_K
    else:
        k_start = K

    for i in range(BLOCK_C):
        k_block_ptr = (
            k_start
            + offset_qk
            + (offset_block_c + array_c[:, None]) * H * D
            + array_d[None, :]
        )

        q_block_ptr = (
            q_start + offset_qk + (offset_block_c + i) * H * D + array_d[None, :]
        )
        if USE_DECAY_K:
            ld_q_block_ptr = (
                LOG_DECAY_K_CUMSUM
                + offset_qk
                + (offset_block_c + i) * H * D
                + array_d[None, :]
            )

            ld_k_block_ptr = (
                LOG_DECAY_K_CUMSUM
                + offset_qk
                + (offset_block_c + array_c[:, None]) * H * D
                + array_d[None, :]
            )

        mask_q = (offset_block_c + i) < N

        a = tl.zeros([BLOCK_C], dtype=tl.float32)

        if REVERSE:
            mask_a = i <= array_c[:, None]
        else:
            mask_a = i >= array_c[:, None]

        for j in range(NUM_BLOCK_D):
            mask_d = (j * BLOCK_D + array_d) < D

            q = tl.load(q_block_ptr, mask=mask_q & mask_d[None, :], other=0.0)
            if SHARE_Q:
                q = 1 - tl.exp(q.to(tl.float32)).to(q.dtype)

            k = tl.load(k_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)
            if SHARE_K:
                k = 1 - tl.exp(k.to(tl.float32)).to(k.dtype)

            # 1 BLOCK_D, BLOCK_C BLOCK_D -> BLOCK_C BLOCK_D
            score = q * k

            if USE_DECAY_K:
                ld_q = tl.load(
                    ld_q_block_ptr, mask=mask_q & mask_d[None, :], other=0.0
                ).to(tl.float32)
                ld_k = tl.load(
                    ld_k_block_ptr,
                    mask=mask_q & mask_c[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(
                    tl.float32
                )  # add mask_q is important !!!
                ld_qk_diff = ld_q - ld_k
                decay_qk_diff = tl.exp(ld_qk_diff)
                score *= decay_qk_diff

            score = tl.where(mask_a, score, 0.0)
            a += tl.sum(score, axis=-1)

            q_block_ptr += BLOCK_D
            k_block_ptr += BLOCK_D
            if USE_DECAY_K:
                ld_q_block_ptr += BLOCK_D
                ld_k_block_ptr += BLOCK_D

        a_block_ptr = A + offset_a + i * BLOCK_C + array_c

        tl.store(
            a_block_ptr,
            a.to(a_block_ptr.dtype.element_ty),
        )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_D": [64, 128],
            "BLOCK_E": [64, 128],
        }
    ),
    key=["B", "N", "H", "D", "E", "USE_CU_SEQLENS", "USE_DECAY_K", "USE_DECAY_V"],
)
@triton.jit
def _lavd_parallel_sub_intra_o(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    O,  # B N H E
    A,  # B H NUM_ATTN_MATRIX BLOCK_C BLOCK_C
    LOG_DECAY_K,  # B N H D
    LOG_DECAY_V,  # B N H E
    LOG_DECAY_K_CUMSUM,  # B N H D
    LOG_DECAY_V_CUMSUM,  # B N H E
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_DECAY_K: tl.constexpr,
    USE_DECAY_V: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_PAD: tl.constexpr,
    REVERSE: tl.constexpr,
    SHARE_K: tl.constexpr,
    SHARE_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    NUM_ATTN_MATRIX = tl.cdiv(N, BLOCK_C)
    tl.cdiv(D, BLOCK_D)

    off_bh = tl.program_id(0)
    off_b = off_bh // H
    off_h = off_bh % H
    off_c = tl.program_id(1)
    off_block_e = tl.program_id(2)

    # compute offset
    off_b * N * H * D + off_h * D
    offset_vo = off_b * N * H * E + off_h * E
    offset_block_c = off_c * BLOCK_C
    offset_block_e = off_block_e * BLOCK_E
    offset_a = (
        off_b * H * NUM_ATTN_MATRIX * BLOCK_C * BLOCK_C
        + off_h * NUM_ATTN_MATRIX * BLOCK_C * BLOCK_C
        + off_c * BLOCK_C * BLOCK_C
    )

    array_c = tl.arange(0, BLOCK_C)
    tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)

    # compute mask
    mask_c = (offset_block_c + array_c) < N
    mask_e = (offset_block_e + array_e) < E

    # compute block ptr
    if SHARE_V:
        v_start = LOG_DECAY_V
    else:
        v_start = V

    if USE_DECAY_V:
        ldv_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + (offset_block_c + array_c[:, None]) * H * E
            + (offset_block_e + array_e)[None, :]
        )

    v_block_ptr = (
        v_start
        + offset_vo
        + (offset_block_c + array_c[:, None]) * H * E
        + (offset_block_e + array_e)[None, :]
    )

    o_block_ptr = (
        O
        + offset_vo
        + (offset_block_c + array_c[:, None]) * H * E
        + (offset_block_e + array_e)[None, :]
    )

    a_block_ptr = A + offset_a + array_c[:, None] * BLOCK_C + array_c[None, :]

    a = tl.load(a_block_ptr).to(tl.float32)

    if REVERSE:
        mask_a = (array_c[:, None] <= array_c[None, :])[:, :, None]
    else:
        mask_a = (array_c[:, None] >= array_c[None, :])[:, :, None]

    v = tl.load(v_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0).to(
        tl.float32
    )
    if SHARE_V:
        v = 1 - tl.exp(v.to(tl.float32)).to(v.dtype)

    o = tl.dot(a, v)
    if not USE_DECAY_V:
        o = tl.dot(a, v)
    else:
        ld_vo = tl.load(
            ldv_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)
        ld_vo_diff = ld_vo[:, None, :] - ld_vo[None, :, :]
        decay_vo_diff = tl.exp(ld_vo_diff)
        a_ = a[:, :, None] * decay_vo_diff
        a_ = tl.where(mask_a, a_, 0.0)
        o = tl.sum(a_ * v[None, :, :], axis=1)

    tl.store(
        o_block_ptr,
        o.to(o_block_ptr.dtype.element_ty),
        mask=mask_c[:, None] & mask_e[None, :],
    )
