import triton
import triton.language as tl

from xopes.utils import generate_configs


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": [16, 128],
            "BLOCK_D": [128],
            "BLOCK_E": [128],
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
                k_trans = 1 - tl.exp(k_trans)

            if USE_DECAY_K:
                log_decay_k_trans = tl.load(
                    ldk_trans_block_ptr, mask=mask_k_trans, other=0.0
                ).to(tl.float32)
                log_k_trans_decay = log_decay_k_trans_sum - log_decay_k_trans
                k_trans_decay = tl.exp(log_k_trans_decay)
                k_trans = (k_trans * k_trans_decay).to(k_trans.dtype)

            v = tl.load(v_block_ptr, mask=mask_v, other=0.0)
            if SHARE_V:
                v = 1 - tl.exp(v)

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
            "BLOCK_D": [128],
            "BLOCK_E": [128],
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
            "BLOCK_C": [16, 128],
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
                    k_trans = 1 - tl.exp(k_trans)

                if USE_DECAY_K:
                    log_decay_k_trans = tl.load(
                        ldk_trans_block_ptr, mask=mask_k_trans, other=0.0
                    ).to(tl.float32)
                    log_k_trans_decay = log_decay_k_trans_sum - log_decay_k_trans
                    k_trans_decay = tl.exp(log_k_trans_decay)
                    k_trans = (k_trans * k_trans_decay).to(k_trans.dtype)

                v = tl.load(v_block_ptr, mask=mask_v, other=0.0)
                if SHARE_V:
                    v = 1 - tl.exp(v)

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


_lavd_parallel_inter = None
_lavd_parallel_intra = None
_lavd_parallel_intra_inter = None
