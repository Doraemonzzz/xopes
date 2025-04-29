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
    COMPUTE_LOG_DECAY_CUMSUM: tl.constexpr,
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
    offset_kv = off_b * N * D

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

    k_block_ptr = K + offset_kv + kv_start * D + array_d
    v_block_ptr = V + offset_kv + kv_start * D + array_d
    log_decay_block_ptr = LOG_DECAY + offset_kv + decay_start * D + array_d
    log_decay_cumsum_block_ptr = LOG_DECAY_CUMSUM + offset_kv + kv_start * D + array_d
    state_block_ptr = STATES + offset_kv + kv_start * D + array_d

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

        if COMPUTE_LOG_DECAY_CUMSUM:
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
