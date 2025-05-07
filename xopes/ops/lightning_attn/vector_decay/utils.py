import triton
import triton.language as tl

from xopes.utils import generate_configs
from xopes.utils.constant import XOPES_DEBUG

BLOCK_C = 16
BLOCK_C_LIST = [BLOCK_C]
if XOPES_DEBUG:
    BLOCK_D_LIST = [16]
    BLOCK_E_LIST = [16]
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
            "BLOCK_C": [64, 128],
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
        "USE_CU_SEQLENS",
        "USE_DECAY_K",
        "USE_DECAY_V",
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
    MAX_BLOCK_N: tl.constexpr,
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
        "USE_CU_SEQLENS",
        "USE_DECAY_K",
        "USE_DECAY_V",
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
    MAX_BLOCK_N: tl.constexpr,
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


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": [64, 128],
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
        "USE_CU_SEQLENS",
        "USE_DECAY_K",
        "USE_DECAY_V",
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
    MAX_BLOCK_N: tl.constexpr,
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


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": BLOCK_C_LIST,
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
        "USE_CU_SEQLENS",
        "USE_DECAY_K",
        "USE_DECAY_V",
    ],
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


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": BLOCK_C_LIST,
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
        "USE_CU_SEQLENS",
        "USE_DECAY_K",
        "USE_DECAY_V",
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
    MAX_BLOCK_N: tl.constexpr,
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


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": BLOCK_C_LIST,
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
        "USE_CU_SEQLENS",
        "USE_DECAY_K",
        "USE_DECAY_V",
    ],
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
    MAX_BLOCK_N: tl.constexpr,
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


# no loop slower version, only for reference
@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": BLOCK_C_LIST,
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
        "USE_CU_SEQLENS",
        "USE_DECAY_K",
        "USE_DECAY_V",
    ],
)
@triton.jit
def _lavd_parallel_intra_inter_no_loop_v1(
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
    BLOCK_C_: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    MAX_BLOCK_N: tl.constexpr,
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
            q_ = (q * k_decay_sub).to(q.dtype)
            score = tl.dot(q_, k_trans).to(q.dtype)
            o_inter = tl.dot(score, v)
        else:
            score = tl.dot(q, k_trans).to(q.dtype)
            o_inter = tl.dot(score, v)

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


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": [BLOCK_C * 2],
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
        "USE_CU_SEQLENS",
        "USE_DECAY_K",
        "USE_DECAY_V",
    ],
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
    BLOCK_C_: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    MAX_BLOCK_N: tl.constexpr,
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

    offset_block_c1 = offset_block_c
    offset_block_c2 = offset_block_c + BLOCK_C_

    offset_state = off_bh * (NUM_BLOCK_N + 1) * D * E
    offset_block_state = off_block_n * D * E

    # array_c = tl.arange(0, BLOCK_C)
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    array_n = tl.arange(0, BLOCK_N)
    array_c_ = tl.arange(0, BLOCK_C_)

    # compute mask
    array_kv = array_n
    mask_n = (offset_block_n + array_n) < N
    mask_e = (offset_block_e + array_e) < E
    mask_c1 = (offset_block_n + offset_block_c1 + array_c_) < N
    mask_c2 = (offset_block_n + offset_block_c2 + array_c_) < N

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
    q1_block_ptr = (
        q_start
        + offset_qk
        + offset_block_qk
        + (offset_block_c1 + array_c_[:, None]) * H * D
        + array_d[None, :]
    )

    q2_block_ptr = (
        q_start
        + offset_qk
        + offset_block_qk
        + (offset_block_c2 + array_c_[:, None]) * H * D
        + array_d[None, :]
    )

    M = offset_block_c
    if REVERSE:
        offset_ld1 = offset_block_c1 + BLOCK_C_
        offset_ld2 = offset_block_c2 + BLOCK_C_
        offset_ldk1_sum = offset_ld1 * H * D
        offset_ldk2_sum = offset_ld2 * H * D
        offset_ldv1_sum = offset_ld1 * H * E
        offset_ldv2_sum = offset_ld2 * H * E

        mask_kv1 = (array_kv >= M) & mask_n
        mask_kv2 = (array_kv >= (M + BLOCK_C_)) & mask_n
        mask_kv = mask_kv2
        ld1_sum_mask = (offset_ld1 < BLOCK_N) & ((offset_block_n + offset_ld1) < N)
        ld2_sum_mask = (offset_ld2 < BLOCK_N) & ((offset_block_n + offset_ld2) < N)
    else:
        # if off_block_c = 0, no sub intra is needed
        if off_block_n == NUM_BLOCK_N - 1:
            if USE_PAD:
                if offset_block_n + offset_block_c1 < N:
                    offset_ld1 = offset_block_c1 - 1
                else:
                    offset_ld1 = N % BLOCK_N - 1

                if offset_block_n + offset_block_c2 < N:
                    offset_ld2 = offset_block_c2 - 1
                else:
                    offset_ld2 = N % BLOCK_N - 1
            else:
                offset_ld1 = offset_block_c1 - 1
                offset_ld2 = offset_block_c2 - 1
        else:
            offset_ld1 = offset_block_c1 - 1
            offset_ld2 = offset_block_c2 - 1

        offset_ldk1_sum = offset_ld1 * H * D
        offset_ldk2_sum = offset_ld2 * H * D
        offset_ldv1_sum = offset_ld1 * H * E
        offset_ldv2_sum = offset_ld2 * H * E

        (array_kv < M) & mask_n
        mask_kv2 = (array_kv < (M + BLOCK_C_)) & mask_n
        mask_kv = mask_kv2
        ld1_sum_mask = (offset_ld1 >= 0) & ((offset_block_n + offset_ld1) < N)
        ld2_sum_mask = (offset_ld2 >= 0) & ((offset_block_n + offset_ld2) < N)

    if USE_DECAY_K:
        ldk_trans_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + array_kv[None, :] * H * D
            + array_d[:, None]
        )
        ldk1_trans_sum_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + offset_ldk1_sum
            + array_d[:, None]
        )
        ldk2_trans_sum_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + offset_ldk2_sum
            + array_d[:, None]
        )

        ldq1_sub_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + (offset_block_c1 + array_c_[:, None]) * H * D
            + array_d[None, :]
        )

        ldq2_sub_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + (offset_block_c2 + array_c_[:, None]) * H * D
            + array_d[None, :]
        )

        ##### debug
        if REVERSE:
            offset_ldq1_start = offset_block_c1 + BLOCK_C_
            mask_ldq1_start = (offset_ldq1_start < BLOCK_N) & (
                (offset_block_n + offset_ldq1_start) < N
            )
            offset_ldq2_start = offset_block_c2 + BLOCK_C_
            mask_ldq2_start = (offset_ldq2_start < BLOCK_N) & (
                (offset_block_n + offset_ldq2_start) < N
            )
        else:
            offset_ldq1_start = offset_block_c1 - 1
            mask_ldq1_start = (offset_ldq1_start >= 0) & (
                (offset_block_n + offset_ldq1_start) < N
            )
            offset_ldq2_start = offset_block_c2 - 1
            mask_ldq2_start = (offset_ldq2_start >= 0) & (
                (offset_block_n + offset_ldq2_start) < N
            )

        ldq1_start_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + offset_ldq1_start * H * D
            + array_d[None, :]
        )

        ldq2_start_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + offset_ldq2_start * H * D
            + array_d[None, :]
        )

        if REVERSE:
            offset_ldq1_start = offset_ld1
            mask_ldq1_start = (offset_ldq1_start < BLOCK_N) & (
                (offset_block_n + offset_ldq1_start) < N
            )
            offset_ldq2_start = offset_ld2
            mask_ldq2_start = (offset_ldq2_start < BLOCK_N) & (
                (offset_block_n + offset_ldq2_start) < N
            )
        else:
            offset_ldq1_start = offset_ld1
            mask_ldq1_start = (offset_ldq1_start >= 0) & (
                (offset_block_n + offset_ldq1_start) < N
            )
            offset_ldq2_start = offset_ld2
            mask_ldq2_start = (offset_ldq2_start >= 0) & (
                (offset_block_n + offset_ldq2_start) < N
            )

        ldq1_start_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + offset_ldq1_start * H * D
            + array_d[None, :]
        )

        ldq2_start_block_ptr = (
            LOG_DECAY_K_CUMSUM
            + offset_qk
            + offset_block_qk
            + offset_ldq2_start * H * D
            + array_d[None, :]
        )

    o1_block_ptr = (
        O
        + offset_vo
        + offset_block_vo
        + (offset_block_c1 + array_c_[:, None]) * H * E
        + (offset_block_e + array_e[None, :])
    )

    o2_block_ptr = (
        O
        + offset_vo
        + offset_block_vo
        + (offset_block_c2 + array_c_[:, None]) * H * E
        + (offset_block_e + array_e[None, :])
    )

    if USE_DECAY_V:
        ldo1_sub_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + (offset_block_c1 + array_c_[:, None]) * H * E
            + (offset_block_e + array_e[None, :])
        )

        ldo2_sub_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + (offset_block_c2 + array_c_[:, None]) * H * E
            + (offset_block_e + array_e[None, :])
        )

        if REVERSE:
            offset_ldo1_start = offset_block_c1 + BLOCK_C_
            mask_ldo1_start = (offset_ldo1_start < BLOCK_N) & (
                (offset_block_n + offset_ldo1_start) < N
            )
            offset_ldo2_start = offset_block_c2 + BLOCK_C_
            mask_ldo2_start = (offset_ldo2_start < BLOCK_N) & (
                (offset_block_n + offset_ldo2_start) < N
            )
        else:
            offset_ldo1_start = offset_block_c1 - 1
            mask_ldo1_start = (offset_ldo1_start >= 0) & (
                (offset_block_n + offset_ldo1_start) < N
            )
            offset_ldo2_start = offset_block_c2 - 1
            mask_ldo2_start = (offset_ldo2_start >= 0) & (
                (offset_block_n + offset_ldo2_start) < N
            )

        ldo1_start_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + offset_ldo1_start * H * E
            + (offset_block_e + array_e[None, :])
        )

        ldo2_start_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + offset_ldo2_start * H * E
            + (offset_block_e + array_e[None, :])
        )

        log_decay_o1_sub = tl.load(
            ldo1_sub_block_ptr, mask=mask_c1[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)
        log_decay_o1_start = tl.load(
            ldo1_start_block_ptr, mask=mask_ldo1_start & mask_e[None, :], other=0.0
        ).to(tl.float32)
        o1_decay_sub = tl.exp(log_decay_o1_sub - log_decay_o1_start)

        log_decay_o2_sub = tl.load(
            ldo2_sub_block_ptr, mask=mask_c2[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)
        log_decay_o2_start = tl.load(
            ldo2_start_block_ptr, mask=mask_ldo2_start & mask_e[None, :], other=0.0
        ).to(tl.float32)
        o2_decay_sub = tl.exp(log_decay_o2_sub - log_decay_o2_start)

        ##### for inter
        o1_decay_inter = tl.exp(log_decay_o1_sub)
        o2_decay_inter = tl.exp(log_decay_o2_sub)

    o1 = tl.zeros([BLOCK_C_, BLOCK_E], dtype=tl.float32)
    o2 = tl.zeros([BLOCK_C_, BLOCK_E], dtype=tl.float32)

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
        log_decay_v = tl.load(
            ldv_block_ptr, mask=mask_kv[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)

        ldv1_sum_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + offset_ldv1_sum
            + (offset_block_e + array_e)[None, :]
        )
        log_decay_v1_sum = tl.load(
            ldv1_sum_block_ptr, mask=ld1_sum_mask & mask_e[None, :]
        ).to(tl.float32)
        log_decay_v1 = log_decay_v1_sum - log_decay_v
        # !!! important
        log_decay_v1 = tl.where(log_decay_v1 < 0, log_decay_v1, 0)
        v1_decay = tl.exp(log_decay_v1)
        v1 = (v * v1_decay).to(v.dtype)

        ldv2_sum_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + offset_ldv2_sum
            + (offset_block_e + array_e)[None, :]
        )
        log_decay_v2_sum = tl.load(
            ldv2_sum_block_ptr, mask=ld2_sum_mask & mask_e[None, :]
        ).to(tl.float32)
        log_decay_v2 = log_decay_v2_sum - log_decay_v
        # !!! important
        log_decay_v2 = tl.where(log_decay_v2 < 0, log_decay_v2, 0)
        v2_decay = tl.exp(log_decay_v2)
        v2 = (v * v2_decay).to(v.dtype)

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

    array1_sub_inter_row = (offset_block_c1 + array_c_) // BLOCK_C_
    array2_sub_inter_row = (offset_block_c2 + array_c_) // BLOCK_C_
    array_sub_inter_col = array_n // BLOCK_C_
    if REVERSE:
        mask1_sub_inter = array1_sub_inter_row[:, None] < array_sub_inter_col[None, :]
        mask2_sub_inter = array2_sub_inter_row[:, None] < array_sub_inter_col[None, :]
    else:
        mask1_sub_inter = array1_sub_inter_row[:, None] > array_sub_inter_col[None, :]
        mask2_sub_inter = array2_sub_inter_row[:, None] > array_sub_inter_col[None, :]

    for i in range(NUM_BLOCK_D):
        mask_d = (i * BLOCK_D + array_d) < D
        mask_de = mask_d[:, None] & mask_e[None, :]

        q1 = tl.load(q1_block_ptr, mask=mask_c1[:, None] & mask_d[None, :], other=0.0)
        q2 = tl.load(q2_block_ptr, mask=mask_c2[:, None] & mask_d[None, :], other=0.0)

        if SHARE_Q:
            q1 = 1 - tl.exp(q1.to(tl.float32)).to(q1.dtype)
            q2 = 1 - tl.exp(q2.to(tl.float32)).to(q2.dtype)

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
            log_decay_k1_trans_sum = tl.load(
                ldk1_trans_sum_block_ptr, mask=ld1_sum_mask & mask_d[:, None]
            ).to(tl.float32)
            log_decay_k2_trans_sum = tl.load(
                ldk2_trans_sum_block_ptr, mask=ld2_sum_mask & mask_d[:, None]
            ).to(tl.float32)

            log_decay_k_trans = tl.load(
                ldk_trans_block_ptr,
                mask=mask_kv[None, :] & mask_d[:, None],
                other=0.0,
            ).to(tl.float32)
            log_decay_k1_trans = log_decay_k1_trans_sum - log_decay_k_trans
            log_decay_k2_trans = log_decay_k2_trans_sum - log_decay_k_trans
            # !!! important
            k1_trans_decay = tl.exp(log_decay_k1_trans)
            k2_trans_decay = tl.exp(log_decay_k2_trans)
            k1_trans = (k_trans * k1_trans_decay).to(k_trans.dtype)
            k2_trans = (k_trans * k2_trans_decay).to(k_trans.dtype)

            # sub inter decay
            log_decay_q1_start = tl.load(
                ldq1_start_block_ptr,
                mask=mask_ldq1_start & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            log_decay_q2_start = tl.load(
                ldq2_start_block_ptr,
                mask=mask_ldq2_start & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)

            log_decay_q1_sub = tl.load(
                ldq1_sub_block_ptr, mask=mask_c1[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)
            q1_decay_sub = tl.exp(log_decay_q1_sub - log_decay_q1_start)
            q1_ = (q1 * q1_decay_sub).to(q1.dtype)
            score1 = tl.dot(q1_, k1_trans).to(q1.dtype)
            score1 = tl.where(mask1_sub_inter, score1, 0.0)
            if USE_DECAY_V:
                o1_inter = tl.dot(score1, v1)
            else:
                o1_inter = tl.dot(score1, v)

            log_decay_q2_sub = tl.load(
                ldq2_sub_block_ptr, mask=mask_c2[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)
            q2_decay_sub = tl.exp(log_decay_q2_sub - log_decay_q2_start)
            q2_ = (q2 * q2_decay_sub).to(q2.dtype)
            score2 = tl.dot(q2_, k2_trans).to(q2.dtype)
            score2 = tl.where(mask2_sub_inter, score2, 0.0)
            if USE_DECAY_V:
                o2_inter = tl.dot(score2, v2)
            else:
                o2_inter = tl.dot(score2, v)
        else:
            score1 = tl.dot(q1, k_trans).to(q1.dtype)
            score1 = tl.where(mask1_sub_inter, score1, 0.0)
            if USE_DECAY_V:
                o1_inter = tl.dot(score1, v1)
            else:
                o1_inter = tl.dot(score1, v)

            score2 = tl.dot(q2, k_trans).to(q2.dtype)
            score2 = tl.where(mask2_sub_inter, score2, 0.0)
            if USE_DECAY_V:
                o2_inter = tl.dot(score2, v2)
            else:
                o2_inter = tl.dot(score2, v)

        # sub inter decay
        if USE_DECAY_V:
            o1_inter = (o1_inter * o1_decay_sub).to(o1_inter.dtype)
            o2_inter = (o2_inter * o2_decay_sub).to(o2_inter.dtype)

        o1 += o1_inter
        o2 += o2_inter
        ##### end sub inter part

        ##### start inter part
        if USE_DECAY_K:
            q1_decay_inter = tl.exp(log_decay_q1_sub)
            q1 = (q1 * q1_decay_inter).to(q1.dtype)

            q2_decay_inter = tl.exp(log_decay_q2_sub)
            q2 = (q2 * q2_decay_inter).to(q2.dtype)

        state_ = tl.load(state_block_ptr, mask=mask_de, other=0.0).to(q1.dtype)

        o1_inter_ = tl.dot(q1, state_)
        o2_inter_ = tl.dot(q2, state_)

        if USE_DECAY_V:
            o1_inter_ = (o1_inter_ * o1_decay_inter).to(o1_inter_.dtype)
            o2_inter_ = (o2_inter_ * o2_decay_inter).to(o2_inter_.dtype)

        o1 += o1_inter_
        o2 += o2_inter_
        ##### end inter part

        q1_block_ptr += BLOCK_D
        q2_block_ptr += BLOCK_D
        if USE_DECAY_K:
            ldk_trans_block_ptr += BLOCK_D

            ldk1_trans_sum_block_ptr += BLOCK_D
            ldk2_trans_sum_block_ptr += BLOCK_D

            ldq1_sub_block_ptr += BLOCK_D
            ldq2_sub_block_ptr += BLOCK_D

            ldq1_start_block_ptr += BLOCK_D
            ldq2_start_block_ptr += BLOCK_D

        if TRANS_STATE:
            state_block_ptr += BLOCK_D
        else:
            state_block_ptr += BLOCK_D * E
    ##### start sub intra part

    NUM_ATTN_MATRIX = tl.cdiv(N, BLOCK_C_)
    NUM_BLOCK_C = tl.cdiv(BLOCK_N, BLOCK_C_)
    off_c1 = off_block_n * NUM_BLOCK_C + 2 * off_block_c
    off_c2 = off_block_n * NUM_BLOCK_C + 2 * off_block_c + 1
    offset_a1 = (
        off_b * H * NUM_ATTN_MATRIX * BLOCK_C_ * BLOCK_C_
        + off_h * NUM_ATTN_MATRIX * BLOCK_C_ * BLOCK_C_
        + off_c1 * BLOCK_C_ * BLOCK_C_
    )
    offset_a2 = (
        off_b * H * NUM_ATTN_MATRIX * BLOCK_C_ * BLOCK_C_
        + off_h * NUM_ATTN_MATRIX * BLOCK_C_ * BLOCK_C_
        + off_c2 * BLOCK_C_ * BLOCK_C_
    )
    mask_a1 = off_c1 < NUM_ATTN_MATRIX
    mask_a2 = off_c2 < NUM_ATTN_MATRIX

    if TRANS_A:
        a_block_ptr1 = A + offset_a1 + array_c_[None, :] * BLOCK_C_ + array_c_[:, None]
        a_block_ptr2 = A + offset_a2 + array_c_[None, :] * BLOCK_C_ + array_c_[:, None]
    else:
        a_block_ptr1 = A + offset_a1 + array_c_[:, None] * BLOCK_C_ + array_c_[None, :]
        a_block_ptr2 = A + offset_a2 + array_c_[:, None] * BLOCK_C_ + array_c_[None, :]

    a1 = tl.load(a_block_ptr1, mask=mask_a1).to(tl.float32)
    a2 = tl.load(a_block_ptr2, mask=mask_a2).to(tl.float32)

    v1_sub_intra_block_ptr = (
        v_start
        + offset_vo
        + offset_block_vo
        + (offset_block_c1 + array_c_[:, None]) * H * E
        + (offset_block_e + array_e)[None, :]
    )

    v2_sub_intra_block_ptr = (
        v_start
        + offset_vo
        + offset_block_vo
        + (offset_block_c2 + array_c_[:, None]) * H * E
        + (offset_block_e + array_e)[None, :]
    )

    if USE_DECAY_V:
        ldv1_sub_intra_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + (offset_block_c1 + array_c_[:, None]) * H * E
            + (offset_block_e + array_e)[None, :]
        )

        ldv2_sub_intra_block_ptr = (
            LOG_DECAY_V_CUMSUM
            + offset_vo
            + offset_block_vo
            + (offset_block_c2 + array_c_[:, None]) * H * E
            + (offset_block_e + array_e)[None, :]
        )

    v1_sub_intra = tl.load(
        v1_sub_intra_block_ptr, mask=mask_c1[:, None] & mask_e[None, :], other=0.0
    )
    v2_sub_intra = tl.load(
        v2_sub_intra_block_ptr, mask=mask_c2[:, None] & mask_e[None, :], other=0.0
    )
    if SHARE_V:
        v1_sub_intra = 1 - tl.exp(v1_sub_intra.to(tl.float32)).to(v1_sub_intra.dtype)
        v2_sub_intra = 1 - tl.exp(v2_sub_intra.to(tl.float32)).to(v2_sub_intra.dtype)

    if not USE_DECAY_V:
        o1_sub_intra = tl.dot(a1.to(v1_sub_intra.dtype), v1_sub_intra).to(
            v1_sub_intra.dtype
        )
        o2_sub_intra = tl.dot(a2.to(v2_sub_intra.dtype), v2_sub_intra).to(
            v2_sub_intra.dtype
        )
    else:
        if REVERSE:
            mask_a = (array_c_[:, None] <= array_c_[None, :])[:, :, None]
        else:
            mask_a = (array_c_[:, None] >= array_c_[None, :])[:, :, None]

        ld1_vo_sub_intra = tl.load(
            ldv1_sub_intra_block_ptr, mask=mask_c1[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)
        ld1_vo_diff_sub_intra = (
            ld1_vo_sub_intra[:, None, :] - ld1_vo_sub_intra[None, :, :]
        )
        decay1_vo_diff_sub_intra = tl.exp(ld1_vo_diff_sub_intra)
        a1_ = a1[:, :, None] * decay1_vo_diff_sub_intra
        a1_ = tl.where(mask_a, a1_, 0.0)
        o1_sub_intra = tl.sum(a1_ * v1_sub_intra[None, :, :], axis=1).to(
            v1_sub_intra.dtype
        )

        ld2_vo_sub_intra = tl.load(
            ldv2_sub_intra_block_ptr, mask=mask_c2[:, None] & mask_e[None, :], other=0.0
        ).to(tl.float32)
        ld2_vo_diff_sub_intra = (
            ld2_vo_sub_intra[:, None, :] - ld2_vo_sub_intra[None, :, :]
        )
        decay2_vo_diff_sub_intra = tl.exp(ld2_vo_diff_sub_intra)
        a2_ = a2[:, :, None] * decay2_vo_diff_sub_intra
        a2_ = tl.where(mask_a, a2_, 0.0)
        o2_sub_intra = tl.sum(a2_ * v2_sub_intra[None, :, :], axis=1).to(
            v2_sub_intra.dtype
        )

    o1 += o1_sub_intra
    o2 += o2_sub_intra
    #### end sub intra part

    tl.store(
        o1_block_ptr,
        o1.to(o1_block_ptr.dtype.element_ty),
        mask=mask_c1[:, None] & mask_e[None, :],
    )
    tl.store(
        o2_block_ptr,
        o2.to(o2_block_ptr.dtype.element_ty),
        mask=mask_c2[:, None] & mask_e[None, :],
    )

    if COMPUTE_DLD:
        if SHARE_X:
            x_start = LOG_DECAY_V
        else:
            x_start = X

        x1_block_ptr = (
            x_start
            + offset_vo
            + offset_block_vo
            + (offset_block_c1 + array_c_[:, None]) * H * E
            + (offset_block_e + array_e[None, :])
        )
        x2_block_ptr = (
            x_start
            + offset_vo
            + offset_block_vo
            + (offset_block_c2 + array_c_[:, None]) * H * E
            + (offset_block_e + array_e[None, :])
        )
        x1 = tl.load(x1_block_ptr, mask=mask_c1[:, None] & mask_e[None, :], other=0.0)
        x2 = tl.load(x2_block_ptr, mask=mask_c2[:, None] & mask_e[None, :], other=0.0)
        if SHARE_X:
            x1 = 1 - tl.exp(x1.to(tl.float32)).to(x1.dtype)
            x2 = 1 - tl.exp(x2.to(tl.float32)).to(x2.dtype)
        # N E
        dld1 = x1 * o1
        dld2 = x2 * o2

        offset_dld = off_b * N * H * E + off_h * E
        offset_block_dld = offset_block_n * H * E
        dld1_block_ptr = (
            DLOG_DECAY
            + offset_dld
            + offset_block_dld
            + (offset_block_c1 + array_c_[:, None]) * H * E
            + (offset_block_e + array_e[None, :])
        )
        dld2_block_ptr = (
            DLOG_DECAY
            + offset_dld
            + offset_block_dld
            + (offset_block_c2 + array_c_[:, None]) * H * E
            + (offset_block_e + array_e[None, :])
        )
        tl.store(
            dld1_block_ptr,
            dld1.to(dld1_block_ptr.dtype.element_ty),
            mask=mask_c1[:, None] & mask_e[None, :],
        )
        tl.store(
            dld2_block_ptr,
            dld2.to(dld2_block_ptr.dtype.element_ty),
            mask=mask_c2[:, None] & mask_e[None, :],
        )


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": BLOCK_C_LIST,
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
        "USE_CU_SEQLENS",
        "USE_DECAY_K",
        "USE_DECAY_V",
    ],
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
    MAX_BLOCK_N: tl.constexpr,
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
        "USE_CU_SEQLENS",
        "USE_DECAY_K",
        "USE_DECAY_V",
    ],
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
    MAX_BLOCK_N: tl.constexpr,
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
        "USE_CU_SEQLENS",
        "USE_DECAY_K",
        "USE_DECAY_V",
    ],
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
    MAX_BLOCK_N: tl.constexpr,
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
