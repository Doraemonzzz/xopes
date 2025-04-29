import triton
import triton.language as tl

from xopes.utils import generate_configs


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_D": [64, 128],
        }
    ),
    key=[
        "B",
        "MAX_BLOCK_N",
        "H",
        "D",
        "USE_CU_SEQLENS",
    ],
)
@triton.jit
def _laer_parallel_state_parallel(
    K,  # B N D
    V,  # B N D
    STATES,  # B N D
    LOG_DECAY,  # B N D
    LOG_DECAY_CUMSUM,  # B N D
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_PAD: tl.constexpr,
    REVERSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_BLOCK_N: tl.constexpr,
    NUM_BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_block_n = tl.program_id(1)
    off_block_d = tl.program_id(2)

    offset_block_n = off_block_n * BLOCK_N
    offset_block_d = off_block_d * BLOCK_D
    offset_b = off_b * N * D

    # compute block ptr and mask
    if REVERSE:
        if USE_PAD and (off_block_n == NUM_BLOCK_N - 1):
            start = N % BLOCK_N
        else:
            start = BLOCK_N - 1
        kv_start = offset_block_n + start
        decay_start = offset_block_n + start + 1
        stride = -1
    else:
        start = 0
        kv_start = offset_block_n
        decay_start = offset_block_n
        stride = 1

    array_d = offset_block_d + tl.arange(0, BLOCK_D)
    mask_d = array_d < D

    k_block_ptr = K + offset_b + kv_start * D + array_d
    v_block_ptr = V + offset_b + kv_start * D + array_d
    log_decay_block_ptr = LOG_DECAY + offset_b + decay_start * D + array_d
    log_decay_cumsum_block_ptr = LOG_DECAY_CUMSUM + offset_b + kv_start * D + array_d
    state_block_ptr = STATES + offset_b + kv_start * D + array_d

    state = tl.zeros((BLOCK_D,), dtype=tl.float32)
    log_decay_cumsum = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for i in range(BLOCK_N):
        mask_kv_n = (kv_start >= offset_block_n) & (kv_start < N)
        mask = mask_kv_n & mask_d
        k = tl.load(k_block_ptr, mask=mask, other=0.0)
        v = tl.load(v_block_ptr, mask=mask, other=0.0)

        mask_decay_n = (decay_start >= offset_block_n) & (decay_start < N)
        mask_decay = mask_decay_n & mask_d
        log_decay = tl.load(log_decay_block_ptr, mask=mask_decay, other=0.0).to(
            tl.float32
        )

        state = tl.exp(log_decay) * state + k * v

        tl.store(state_block_ptr, state.to(state_block_ptr.dtype.element_ty), mask=mask)

        log_decay_cumsum += log_decay
        tl.store(
            log_decay_cumsum_block_ptr,
            log_decay_cumsum.to(log_decay_cumsum_block_ptr.dtype.element_ty),
            mask=mask,
        )
        log_decay_cumsum_block_ptr += stride * D

        k_block_ptr += stride * D
        v_block_ptr += stride * D
        log_decay_block_ptr += stride * D
        state_block_ptr += stride * D
        kv_start += stride
        decay_start += stride


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_D": [64, 128],
        }
    ),
    key=[
        "B",
        "MAX_BLOCK_N",
        "H",
        "D",
        "USE_CU_SEQLENS",
    ],
)
@triton.jit
def _laer_parallel_intra_inter_fwd(
    Q,  # B N D
    K,  # B N D
    V,  # B N D
    O,  # B N D
    STATE,  # B D
    STATES,  # B N D
    LOG_DECAY,  # B N D
    LOG_DECAY_CUMSUM,  # B N D
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_BLOCK_N: tl.constexpr,
    NUM_BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_block_d = tl.program_id(1)

    offset_block_d = off_block_d * BLOCK_D
    offset_b = off_b * N * D

    array_n = tl.arange(0, BLOCK_N)
    array_d = offset_block_d + tl.arange(0, BLOCK_D)
    mask_d = array_d < D

    ld_block_ptr = LOG_DECAY_CUMSUM + offset_b + array_n[:, None] * D + array_d[None, :]
    q_block_ptr = Q + offset_b + array_n[:, None] * D + array_d[None, :]
    o_block_ptr = O + offset_b + array_n[:, None] * D + array_d[None, :]
    states_block_ptr = STATES + offset_b + array_n[:, None] * D + array_d[None, :]

    if USE_INITIAL_STATE:
        state_block_ptr = STATE + off_b * D + array_d
        state = tl.load(state_block_ptr, mask=mask_d, other=0.0)
    else:
        state = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for i in range(NUM_BLOCK_N):
        mask_n = array_n < N
        mask = mask_n[:, None] & mask_d[None, :]

        current_states = tl.load(states_block_ptr, mask=mask, other=0.0)
        log_decay = tl.load(ld_block_ptr, mask=mask, other=0.0).to(tl.float32)
        decay = tl.exp(log_decay)

        states = decay * state + current_states

        q = tl.load(q_block_ptr, mask=mask, other=0.0)
        o = q * states

        tl.store(
            states_block_ptr, states.to(states_block_ptr.dtype.element_ty), mask=mask
        )
        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask)

        offset_n = min((i + 1) * BLOCK_N, N) - 1
        state_block_ptr = STATES + offset_b + offset_n * D + array_d
        state = tl.load(state_block_ptr, mask=mask_d, other=0.0).to(tl.float32)

        q_block_ptr += BLOCK_N * D
        o_block_ptr += BLOCK_N * D
        states_block_ptr += BLOCK_N * D
        ld_block_ptr += BLOCK_N * D
        array_n += BLOCK_N


@triton.heuristics(
    {
        "MAX_BLOCK_N": lambda args: triton.next_power_of_2(args["N"]),
    }
)
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_D": [64, 128],
        }
    ),
    key=[
        "B",
        "MAX_BLOCK_N",
        "H",
        "D",
        "USE_CU_SEQLENS",
    ],
)
@triton.jit
def _laer_parallel_intra_inter_bwd(
    Q,  # B N D
    K,  # B N D
    V,  # B N D
    DO,  # B N D
    DQ,  # B N D
    DK,  # B N D
    DV,  # B N D
    DLD,  # B N D
    STATE,  # B D
    STATES,  # B N D
    DSTATE,  # B D
    DSTATES,  # B N D
    LOG_DECAY,  # B N D
    LOG_DECAY_REVERSE_CUMSUM,  # B N D
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_DFINAL_STATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_BLOCK_N: tl.constexpr,
    NUM_BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_block_d = tl.program_id(1)

    offset_block_d = off_block_d * BLOCK_D
    offset_b = off_b * N * D

    array_n = NUM_BLOCK_N * BLOCK_N - 1 - tl.arange(0, BLOCK_N)
    array_n_o_states = array_n - 1
    array_d = offset_block_d + tl.arange(0, BLOCK_D)
    mask_d = array_d < D

    ld_block_ptr = LOG_DECAY + offset_b + array_n[:, None] * D + array_d[None, :]
    ld_cumsum_block_ptr = (
        LOG_DECAY_REVERSE_CUMSUM + offset_b + array_n[:, None] * D + array_d[None, :]
    )

    k_block_ptr = K + offset_b + array_n[:, None] * D + array_d[None, :]
    v_block_ptr = V + offset_b + array_n[:, None] * D + array_d[None, :]
    dk_block_ptr = DK + offset_b + array_n[:, None] * D + array_d[None, :]
    dv_block_ptr = DV + offset_b + array_n[:, None] * D + array_d[None, :]
    dld_block_ptr = DLD + offset_b + array_n[:, None] * D + array_d[None, :]
    dstates_block_ptr = DSTATES + offset_b + array_n[:, None] * D + array_d[None, :]

    dq_block_ptr = DQ + offset_b + array_n_o_states[:, None] * D + array_d[None, :]
    do_block_ptr = DO + offset_b + array_n_o_states[:, None] * D + array_d[None, :]
    states_block_ptr = (
        STATES + offset_b + array_n_o_states[:, None] * D + array_d[None, :]
    )  # !!! important

    if USE_DFINAL_STATE:
        dstate_block_ptr = DSTATE + off_b * D + array_d
        dstate = tl.load(dstate_block_ptr, mask=mask_d, other=0.0)
    else:
        dstate = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for i in range(NUM_BLOCK_N):
        mask_n = (array_n >= 0) & (array_n < N)
        mask_n_o_states = (array_n_o_states >= 0) & (array_n_o_states < N)
        mask = mask_n[:, None] & mask_d[None, :]
        mask_o_states = mask_n_o_states[:, None] & mask_d[None, :]

        current_dstates = tl.load(dstates_block_ptr, mask=mask, other=0.0)
        log_decay_cumsum = tl.load(ld_cumsum_block_ptr, mask=mask, other=0.0).to(
            tl.float32
        )
        decay_cumprod = tl.exp(log_decay_cumsum)
        dstates = decay_cumprod * dstate + current_dstates
        k = tl.load(k_block_ptr, mask=mask, other=0.0)
        v = tl.load(v_block_ptr, mask=mask, other=0.0)
        dk = dstates * v
        dv = dstates * k

        # compute dq
        log_decay = tl.load(ld_block_ptr, mask=mask, other=0.0).to(tl.float32)
        decay = tl.exp(log_decay)
        states = tl.load(states_block_ptr, mask=mask_o_states, other=0.0)
        do = tl.load(do_block_ptr, mask=mask_o_states, other=0.0)
        dq = states * do
        dld = dstates * states * decay

        # save
        tl.store(dq_block_ptr, dq.to(dq_block_ptr.dtype.element_ty), mask=mask_o_states)
        tl.store(dk_block_ptr, dk.to(dk_block_ptr.dtype.element_ty), mask=mask)
        tl.store(dv_block_ptr, dv.to(dv_block_ptr.dtype.element_ty), mask=mask)
        tl.store(dld_block_ptr, dld.to(dld_block_ptr.dtype.element_ty), mask=mask)
        tl.store(
            dstates_block_ptr, dstates.to(dstates_block_ptr.dtype.element_ty), mask=mask
        )

        offset_n = (NUM_BLOCK_N - i - 1) * BLOCK_N
        dstate_block_ptr = DSTATES + offset_b + offset_n * D + array_d
        dstate = tl.load(dstate_block_ptr, mask=mask_d, other=0.0)

        dstates_block_ptr -= BLOCK_N * D
        ld_cumsum_block_ptr -= BLOCK_N * D
        k_block_ptr -= BLOCK_N * D
        v_block_ptr -= BLOCK_N * D
        ld_block_ptr -= BLOCK_N * D
        states_block_ptr -= BLOCK_N * D
        do_block_ptr -= BLOCK_N * D
        dq_block_ptr -= BLOCK_N * D
        dk_block_ptr -= BLOCK_N * D
        dv_block_ptr -= BLOCK_N * D
        dld_block_ptr -= BLOCK_N * D
        array_n -= BLOCK_N
        array_n_o_states -= BLOCK_N

    # !!! important
    tl.debug_barrier()

    # compute dq_n = do_n * s_n
    sn_block_ptr = STATES + offset_b + (N - 1) * D + array_d
    sn = tl.load(sn_block_ptr, mask=mask_d, other=0.0)
    do_n_block_ptr = DO + offset_b + (N - 1) * D + array_d
    do_n = tl.load(do_n_block_ptr, mask=mask_d, other=0.0)
    dq_n = sn * do_n
    dq_n_block_ptr = DQ + offset_b + (N - 1) * D + array_d
    tl.store(dq_n_block_ptr, dq_n.to(dq_n_block_ptr.dtype.element_ty), mask=mask_d)

    # compute dld_1
    if USE_INITIAL_STATE:
        s0_block_ptr = STATE + off_b * D + array_d
        s0 = tl.load(s0_block_ptr, mask=mask_d, other=0.0)
    else:
        s0 = tl.zeros((BLOCK_D,), dtype=tl.float32)

    ds1_block_ptr = DSTATES + offset_b + array_d
    ds1 = tl.load(ds1_block_ptr, mask=mask_d, other=0.0)
    ld_1_block_ptr = LOG_DECAY + offset_b + array_d
    decay_1 = tl.exp(tl.load(ld_1_block_ptr, mask=mask_d, other=0.0).to(tl.float32))
    dld_1 = ds1 * s0 * decay_1

    dld_1_block_ptr = DLD + offset_b + array_d
    tl.store(dld_1_block_ptr, dld_1.to(dld_1_block_ptr.dtype.element_ty), mask=mask_d)

    # ds0
    dstate *= decay_1

    final_dstates_block_ptr = DSTATES + offset_b + (N - 1) * D + array_d
    tl.store(
        final_dstates_block_ptr,
        dstate.to(final_dstates_block_ptr.dtype.element_ty),
        mask=mask_d,
    )
