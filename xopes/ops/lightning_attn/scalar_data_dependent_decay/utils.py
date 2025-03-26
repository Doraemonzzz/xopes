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
def _lasd3_parallel_state_parallel(
    K,  # B N H D
    V,  # B N H E
    STATES,  # B H L D E
    LOG_DECAY,  # B N H
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_PAD: tl.constexpr,
    REVERSE: tl.constexpr,
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
    offset_ld = off_b * N * H + off_h
    offset_block_n = off_block_n * BLOCK_N
    offset_block_qk = offset_block_n * H * D
    offset_block_vo = offset_block_n * H * E
    offset_block_ld = offset_block_n * H
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
        offset_ld_sum = 0
    else:
        stride = 1
        array_c = array_c
        # last block
        # offset of sum of local log decay, when not reverse, the offset is the last position of the block
        if off_block_n == NUM_BLOCK_N - 1:
            if USE_PAD:
                offset_ld_sum = (N % BLOCK_N - 1) * H
            else:
                offset_ld_sum = (BLOCK_N - 1) * H
        else:
            offset_ld_sum = (BLOCK_N - 1) * H

    offset_block_state = off_block_n * D * E

    k_trans_block_ptr = (
        K
        + offset_qk
        + offset_block_qk
        + array_c[None, :] * H * D
        + (array_d + offset_block_d)[:, None]
    )
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
    ld_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array_c * H
    ld_sum_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + offset_ld_sum
    log_decay_sum = tl.load(ld_sum_block_ptr).to(tl.float32)

    mask_d = (array_d + offset_block_d) < D
    mask_e = (array_e + offset_block_e) < E

    state = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)

    cnt = offset_block_n
    for i in range(NUM_BLOCK_C):
        array = offset_block_n + array_c
        mask_c = array < N
        log_decay = tl.load(ld_block_ptr, mask=mask_c, other=0.0).to(tl.float32)
        log_k_decay = log_decay_sum - log_decay

        if cnt < N:
            if USE_PAD and (off_block_n == NUM_BLOCK_N - 1):
                if i == NUM_BLOCK_C - 1:
                    M = N % BLOCK_C
                    if REVERSE:
                        array = tl.arange(0, BLOCK_C)
                    else:
                        array = M - 1 - tl.arange(0, BLOCK_C)
                    log_k_decay = tl.where(
                        array >= 0, log_k_decay, 0
                    )  # !!! important, otherwise the decay will be nan, nan * 0 != 0

            k_trans = tl.load(
                k_trans_block_ptr, mask=mask_c[None, :] & mask_d[:, None], other=0.0
            )
            v = tl.load(v_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0)

            k_decay = tl.exp(log_k_decay)
            k_trans = (k_trans * k_decay[None, :]).to(k_trans.dtype)
            # for local state, since the local decay has been applied, we don't need to apply block_decay
            state += tl.dot(k_trans, v)

        k_trans_block_ptr += BLOCK_C * H * D * stride
        v_block_ptr += BLOCK_C * H * E * stride
        ld_block_ptr += BLOCK_C * H * stride
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
def _lasd3_parallel_state_reduce(
    STATE,  # B H D E
    STATES,  # B H L D E
    LOG_DECAY,  # B N H
    LOG_DECAY_CUMSUM,  # B N H
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
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
    offset_ld = off_b * N * H + off_h

    # compute array for block ptr
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)

    if REVERSE:
        stride = -1
        states_start = (NUM_BLOCK_N - 1) * D * E
        # last chunk's first element
        offset_ld_sum = (NUM_BLOCK_N - 1) * BLOCK_N * H
    else:
        stride = 1
        states_start = 0
        # first chunk's last element
        offset_ld_sum = (BLOCK_N - 1) * H

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
    ld_block_ptr = LOG_DECAY_CUMSUM + offset_ld + offset_ld_sum

    mask_d = (array_d + offset_block_d) < D
    mask_e = (array_e + offset_block_e) < E
    mask = mask_d[:, None] & mask_e[None, :]

    c = 1.0
    if REVERSE:
        # last block's first element
        last_block_decay_block_ptr = (
            LOG_DECAY_CUMSUM + offset_ld + (NUM_BLOCK_N - 1) * BLOCK_N * H
        )  # (N - C) * H

        # !!! important
        first_decay_block_ptr = LOG_DECAY + offset_ld
        c = tl.exp(tl.load(first_decay_block_ptr).to(tl.float32))
    else:
        # first block's last element
        last_block_decay_block_ptr = (
            LOG_DECAY_CUMSUM + offset_ld + (N - 1) * H
        )  # (BLOCK_N - 1) * H

    # !!! important
    last_block_decay = tl.exp(tl.load(last_block_decay_block_ptr).to(tl.float32))

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
        block_decay = tl.exp(tl.load(ld_block_ptr).to(tl.float32))

        tl.store(
            states_block_ptr, state.to(states_block_ptr.dtype.element_ty), mask=mask
        )

        if REVERSE:
            if i == 0:
                state = last_block_decay * state + current_state
            else:
                state = block_decay * state + current_state
        else:
            if i == NUM_BLOCK_N - 1:
                state = last_block_decay * state + current_state
            else:
                state = block_decay * state + current_state

        states_block_ptr += D * E * stride
        ld_block_ptr += BLOCK_N * H * stride

    # !!! important
    state *= c

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
def _lasd3_parallel_state_parallel_reduce(
    K,  # B N H D
    V,  # B N H E
    STATE,  # B H D E
    STATES,  # B H L D E
    LOG_DECAY,  # B N H
    LOG_DECAY_CUMSUM,  # B N H
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_PAD: tl.constexpr,
    REVERSE: tl.constexpr,
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
    offset_ld = off_b * N * H + off_h

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
        offset_ld_sum = (NUM_BLOCK_N - 1) * BLOCK_N * H
    else:
        stride = 1
        states_start = 0
        # first chunk's last element
        offset_ld_sum = (BLOCK_N - 1) * H

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
    ld_cumsum_block_ptr = LOG_DECAY_CUMSUM + offset_ld + offset_ld_sum

    mask_d = (array_d + offset_block_d) < D
    mask_e = (array_e + offset_block_e) < E
    mask = mask_d[:, None] & mask_e[None, :]

    c = 1.0
    if REVERSE:
        # last block's first element
        last_block_decay_block_ptr = (
            LOG_DECAY_CUMSUM + offset_ld + (NUM_BLOCK_N - 1) * BLOCK_N * H
        )  # (N - C) * H

        # !!! important
        first_decay_block_ptr = LOG_DECAY + offset_ld
        c = tl.exp(tl.load(first_decay_block_ptr).to(tl.float32))
    else:
        # first block's last element
        last_block_decay_block_ptr = (
            LOG_DECAY_CUMSUM + offset_ld + (N - 1) * H
        )  # (BLOCK_N - 1) * H

    # !!! important
    last_block_decay = tl.exp(tl.load(last_block_decay_block_ptr).to(tl.float32))

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

        ##### compute local state
        cnt = offset_block_n
        offset_block_qk = offset_block_n * H * D
        offset_block_vo = offset_block_n * H * E
        offset_block_ld = offset_block_n * H
        array_c = tl.arange(0, BLOCK_C)

        # for reverse, local loop start from the last block
        if REVERSE:
            # BLOCK_N - 1, ... , BLOCK_N - BLOCK_C
            array_c = BLOCK_N - 1 - array_c
            # offset of sum of local log decay, when reverse, the offset is 0
            offset_ld_sum = 0
        else:
            array_c = array_c
            # last block
            # offset of sum of local log decay, when not reverse, the offset is the last position of the block
            if off_block_n == NUM_BLOCK_N - 1:
                if USE_PAD:
                    offset_ld_sum = (N % BLOCK_N - 1) * H
                else:
                    offset_ld_sum = (BLOCK_N - 1) * H
            else:
                offset_ld_sum = (BLOCK_N - 1) * H

        k_trans_block_ptr = (
            K
            + offset_qk
            + offset_block_qk
            + array_c[None, :] * H * D
            + (array_d + offset_block_d)[:, None]
        )
        v_block_ptr = (
            V
            + offset_vo
            + offset_block_vo
            + array_c[:, None] * H * E
            + (array_e + offset_block_e)[None, :]
        )
        ld_block_ptr = LOG_DECAY_CUMSUM + offset_ld + offset_block_ld + array_c * H
        ld_sum_block_ptr = (
            LOG_DECAY_CUMSUM + offset_ld + offset_block_ld + offset_ld_sum
        )
        log_decay_sum = tl.load(ld_sum_block_ptr).to(tl.float32)

        current_state = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)

        for j in range(NUM_BLOCK_C):
            array = offset_block_n + array_c
            mask_c = array < N
            log_decay = tl.load(ld_block_ptr, mask=mask_c, other=0.0).to(tl.float32)
            log_k_decay = log_decay_sum - log_decay

            if cnt < N:
                if USE_PAD and (off_block_n == NUM_BLOCK_N - 1):
                    if j == NUM_BLOCK_C - 1:
                        M = N % BLOCK_C
                        if REVERSE:
                            array = tl.arange(0, BLOCK_C)
                        else:
                            array = M - 1 - tl.arange(0, BLOCK_C)
                        log_k_decay = tl.where(
                            array >= 0, log_k_decay, 0
                        )  # !!! important, otherwise the decay will be nan, nan * 0 != 0

                k_trans = tl.load(
                    k_trans_block_ptr, mask=mask_c[None, :] & mask_d[:, None], other=0.0
                )
                v = tl.load(
                    v_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0
                )

                k_decay = tl.exp(log_k_decay)
                k_trans = (k_trans * k_decay[None, :]).to(k_trans.dtype)
                # for local state, since the local decay has been applied, we don't need to apply block_decay
                current_state += tl.dot(k_trans, v)

            k_trans_block_ptr += BLOCK_C * H * D * stride
            v_block_ptr += BLOCK_C * H * E * stride
            ld_block_ptr += BLOCK_C * H * stride
            array_c += BLOCK_C * stride
            cnt += BLOCK_C * stride

        ##### update global state
        block_decay = tl.exp(tl.load(ld_cumsum_block_ptr).to(tl.float32))
        if REVERSE:
            if i == 0:
                state = last_block_decay * state + current_state
            else:
                state = block_decay * state + current_state
        else:
            if i == NUM_BLOCK_N - 1:
                state = last_block_decay * state + current_state
            else:
                state = block_decay * state + current_state

        states_block_ptr += D * E * stride
        ld_cumsum_block_ptr += BLOCK_N * H * stride
        offset_block_n += BLOCK_N * stride
        off_block_n += stride  # !!! important

    # !!! important
    state *= c

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
            "BLOCK_D": [128],
            "BLOCK_E": [128],
        }
    ),
    key=["B", "N", "H", "D", "E", "USE_CU_SEQLENS", "USE_LOG_DECAY"],
)
@triton.jit
def _lasd3_parallel_intra(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    O,  # B N H E
    LOG_DECAY,  # B N H
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    REVERSE: tl.constexpr,
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
    offset_ld = off_b * N * H + off_h
    offset_block_n = off_block_n * BLOCK_N
    offset_block_qk = offset_block_n * H * D
    offset_block_vo = offset_block_n * H * E
    offset_block_ld = offset_block_n * H
    offset_block_c = off_block_c * BLOCK_C
    offset_block_e = off_block_e * BLOCK_E

    array_c = tl.arange(0, BLOCK_C)
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)

    # compute block ptr and mask
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
    mask_c = (offset_block_n + offset_block_c + array_c) < N
    mask_e = (offset_block_e + array_e) < E

    # compute mask
    array_c = tl.arange(0, BLOCK_C)
    array_q = offset_block_c + array_c

    o = tl.zeros([BLOCK_C, BLOCK_E], dtype=tl.float32)

    if REVERSE:
        stride = -1
        NUM_BLOCK_C = tl.cdiv(BLOCK_N, BLOCK_C)
        NUM_LOOP = NUM_BLOCK_C - off_block_c
    else:
        stride = 1
        NUM_LOOP = off_block_c + 1

    ldq_block_ptr = (
        LOG_DECAY + offset_ld + offset_block_ld + (offset_block_c + array_c) * H
    )
    ldq = tl.load(ldq_block_ptr, mask=mask_c, other=0.0).to(tl.float32)

    for i in range(NUM_BLOCK_D):
        mask_d = (i * BLOCK_D + array_d) < D
        q = tl.load(q_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)

        if REVERSE:
            array_kv = BLOCK_N - BLOCK_C + array_c
        else:
            array_kv = array_c

        k_trans_block_ptr = (
            K
            + offset_qk
            + offset_block_qk
            + array_kv[None, :] * H * D
            + (i * BLOCK_D + array_d)[:, None]
        )
        v_block_ptr = (
            V
            + offset_vo
            + offset_block_vo
            + array_kv[:, None] * H * E
            + (offset_block_e + array_e)[None, :]
        )
        ldk_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array_kv * H

        for j in range(NUM_LOOP):
            mask_kv = (offset_block_n + array_kv) < N

            k_trans = tl.load(
                k_trans_block_ptr, mask=mask_kv[None, :] & mask_d[:, None], other=0.0
            )
            v = tl.load(v_block_ptr, mask=mask_kv[:, None] & mask_e[None, :], other=0.0)
            ldk = tl.load(ldk_block_ptr, mask=mask_kv, other=0).to(tl.float32)

            score = tl.dot(q, k_trans)
            diff = (array_q[:, None] - array_kv[None, :]) * stride  # !!! important
            log_decay = (ldq[:, None] - ldk[None, :]) * stride
            # decay = tl.exp(tl.where(diff >= 0, log_decay, float("-inf")))
            decay = tl.exp(tl.where(log_decay <= 0, log_decay, float("-inf")))
            score *= decay
            o += tl.dot(score.to(v.dtype), v)

            k_trans_block_ptr += BLOCK_C * H * D * stride
            v_block_ptr += BLOCK_C * H * E * stride
            ldk_block_ptr += BLOCK_C * H * stride
            array_kv += BLOCK_C * stride

        # !!! important
        tl.debug_barrier()

        q_block_ptr += BLOCK_D

    tl.store(
        o_block_ptr,
        o.to(o_block_ptr.dtype.element_ty),
        mask=mask_c[:, None] & mask_e[None, :],
    )


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
def _lasd3_parallel_inter(
    Q,  # B N H D
    O,  # B N H E
    STATES,  # B H L D E if not trans_states, B H L E D if trans_states
    LOG_DECAY,  # B N H
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
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
    offset_ld = off_b * N * H + off_h
    offset_block_n = off_block_n * BLOCK_N
    offset_block_qk = offset_block_n * H * D
    offset_block_vo = offset_block_n * H * E
    offset_block_ld = offset_block_n * H
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
        + (offset_block_c + array_c)[:, None] * H * E
        + (offset_block_e + array_e)[None, :]
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

    ldq_block_ptr = (
        LOG_DECAY + offset_ld + offset_block_ld + (offset_block_c + array_c) * H
    )
    ldq = tl.load(ldq_block_ptr, mask=mask_c, other=0.0).to(tl.float32)

    o = tl.load(o_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0).to(
        tl.float32
    )
    for i in range(NUM_BLOCK_D):
        mask_d = (array_d + i * BLOCK_D) < D
        q = tl.load(q_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)
        state = tl.load(
            state_block_ptr, mask=mask_d[:, None] & mask_e[None, :], other=0.0
        ).to(q.dtype)
        q_decay = tl.exp(ldq[:, None])

        o_ = tl.dot(q, state)
        o_ *= q_decay
        o += o_

        q_block_ptr += BLOCK_D
        if TRANS:
            state_block_ptr += BLOCK_D
        else:
            state_block_ptr += BLOCK_D * E

    tl.store(
        o_block_ptr,
        o.to(o_block_ptr.dtype.element_ty),
        mask=mask_c[:, None] & mask_e[None, :],
    )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": [16, 32, 64, 128],
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
def _lasd3_parallel_intra_inter(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    O,  # B N H E
    STATES,  # B H L D E if not trans_states, B H L E D if trans_states
    LOG_DECAY,  # B N H
    LOG_DECAY_REVERSE,  # B N H
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
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
    offset_ld = off_b * N * H + off_h
    offset_block_n = off_block_n * BLOCK_N
    offset_block_qk = offset_block_n * H * D
    offset_block_vo = offset_block_n * H * E
    offset_block_ld = offset_block_n * H
    offset_block_c = off_block_c * BLOCK_C
    offset_block_e = off_block_e * BLOCK_E

    offset_state = off_bh * (NUM_BLOCK_N + 1) * D * E
    offset_block_state = off_block_n * D * E

    # compute block ptr and mask
    array_e = tl.arange(0, BLOCK_E)
    array_d = tl.arange(0, BLOCK_D)
    array_c = tl.arange(0, BLOCK_C)
    array_q = offset_block_c + array_c

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
        + (offset_block_c + array_c)[:, None] * H * E
        + (offset_block_e + array_e)[None, :]
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

    o = tl.zeros([BLOCK_C, BLOCK_E], dtype=tl.float32)

    if REVERSE:
        stride = -1
        NUM_BLOCK_C = tl.cdiv(BLOCK_N, BLOCK_C)
        NUM_LOOP = NUM_BLOCK_C - off_block_c
    else:
        stride = 1
        NUM_LOOP = off_block_c + 1

    ldq_block_ptr = (
        LOG_DECAY + offset_ld + offset_block_ld + (offset_block_c + array_c) * H
    )
    ldq = tl.load(ldq_block_ptr, mask=mask_c, other=0.0).to(tl.float32)

    if REVERSE:
        ldq_inter_block_ptr = (
            LOG_DECAY_REVERSE
            + offset_ld
            + offset_block_ld
            + (offset_block_c + array_c) * H
        )
        ldq_inter = tl.load(ldq_inter_block_ptr, mask=mask_c, other=0.0).to(tl.float32)
    else:
        ldq_inter = ldq

    for i in range(NUM_BLOCK_D):
        mask_d = (array_d + i * BLOCK_D) < D
        q = tl.load(q_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)

        ##### intra start #####
        if REVERSE:
            array_kv = BLOCK_N - BLOCK_C + array_c
        else:
            array_kv = array_c

        k_trans_block_ptr = (
            K
            + offset_qk
            + offset_block_qk
            + array_kv[None, :] * H * D
            + (i * BLOCK_D + array_d)[:, None]
        )
        v_block_ptr = (
            V
            + offset_vo
            + offset_block_vo
            + array_kv[:, None] * H * E
            + (offset_block_e + array_e)[None, :]
        )
        ldk_block_ptr = LOG_DECAY + offset_ld + offset_block_ld + array_kv * H

        for j in range(NUM_LOOP):
            mask_kv = (offset_block_n + array_kv) < N

            k_trans = tl.load(
                k_trans_block_ptr, mask=mask_kv[None, :] & mask_d[:, None], other=0.0
            )
            v = tl.load(v_block_ptr, mask=mask_kv[:, None] & mask_e[None, :], other=0.0)
            ldk = tl.load(ldk_block_ptr, mask=mask_kv, other=0).to(tl.float32)

            score = tl.dot(q, k_trans)
            diff = (array_q[:, None] - array_kv[None, :]) * stride  # !!! important
            log_decay = (ldq[:, None] - ldk[None, :]) * stride
            # decay = tl.exp(tl.where(diff >= 0, log_decay, float("-inf")))
            decay = tl.exp(tl.where(log_decay <= 0, log_decay, float("-inf")))
            score *= decay
            o += tl.dot(score.to(v.dtype), v)

            k_trans_block_ptr += BLOCK_C * H * D * stride
            v_block_ptr += BLOCK_C * H * E * stride
            ldk_block_ptr += BLOCK_C * H * stride
            array_kv += BLOCK_C * stride
        ##### intra end #####

        ##### inter start #####
        state = tl.load(
            state_block_ptr, mask=mask_d[:, None] & mask_e[None, :], other=0.0
        ).to(q.dtype)
        q_decay = tl.exp(ldq_inter[:, None])

        o_ = tl.dot(q, state)
        o_ *= q_decay
        o += o_
        ##### inter end #####

        q_block_ptr += BLOCK_D
        if TRANS:
            state_block_ptr += BLOCK_D
        else:
            state_block_ptr += BLOCK_D * E

    tl.store(
        o_block_ptr,
        o.to(o_block_ptr.dtype.element_ty),
        mask=mask_c[:, None] & mask_e[None, :],
    )
