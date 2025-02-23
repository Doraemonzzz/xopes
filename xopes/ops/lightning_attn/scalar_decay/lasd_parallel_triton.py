from typing import Optional

import torch
import triton
import triton.language as tl
from einops import repeat

from xopes.utils import contiguous, generate_configs


########## Fwd start ##########
@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": [16, 32, 64, 128],
            "BLOCK_D": [128],
            "BLOCK_E": [128],
        }
    ),
    key=["B", "N", "H", "D", "E", "USE_CU_SEQLENS", "USE_LOG_DECAY"],
)
@triton.jit
def _lasd_parallel_fwd_intra(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    O,  # B N H E
    LOG_DECAY,  # H
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_LOG_DECAY: tl.constexpr,
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
    offset_block_n = off_block_n * BLOCK_N
    offset_block_qk = offset_block_n * H * D
    offset_block_vo = offset_block_n * H * E
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
    array_q = array_c + offset_block_c

    o = tl.zeros([BLOCK_C, BLOCK_E], dtype=tl.float32)
    if USE_LOG_DECAY:
        log_decay = tl.load(LOG_DECAY + off_h).to(tl.float32)

    for i in range(NUM_BLOCK_D):
        mask_d = (i * BLOCK_D + array_d) < D
        q = tl.load(q_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)

        k_trans_block_ptr = (
            K
            + offset_qk
            + offset_block_qk
            + array_c[None, :] * H * D
            + (i * BLOCK_D + array_d)[:, None]
        )
        v_block_ptr = (
            V
            + offset_vo
            + offset_block_vo
            + array_c[:, None] * H * E
            + (offset_block_e + array_e)[None, :]
        )

        for j in range(off_block_c + 1):
            array_kv = j * BLOCK_C + array_c
            mask_kv = array_kv < N

            k_trans = tl.load(
                k_trans_block_ptr, mask=mask_kv[None, :] & mask_d[:, None], other=0.0
            )
            v = tl.load(v_block_ptr, mask=mask_kv[:, None] & mask_e[None, :], other=0.0)

            score = tl.dot(q, k_trans)
            diff = array_q[:, None] - array_kv[None, :]
            if not USE_LOG_DECAY:
                score = tl.where(diff >= 0, score, 0.0)
            else:
                decay = log_decay * diff
                decay = tl.exp(tl.where(diff >= 0, decay, float("-inf")))
                score *= decay
            o += tl.dot(score.to(v.dtype), v)

            k_trans_block_ptr += BLOCK_C * H * D
            v_block_ptr += BLOCK_C * H * E

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
            "BLOCK_C": [16, 32, 64, 128],
            "BLOCK_D": [128],
            "BLOCK_E": [128],
        }
    ),
    key=["B", "N", "H", "D", "E", "USE_CU_SEQLENS", "USE_LOG_DECAY"],
)
@triton.jit
def _lasd_parallel_fwd_state_parallel(
    K,  # B N H D
    V,  # B N H E
    STATES,  # B H L D E
    LOG_DECAY,  # H
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_LOG_DECAY: tl.constexpr,
    USE_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCK_C = tl.cdiv(BLOCK_N, BLOCK_C)
    tl.cdiv(D, BLOCK_D)
    tl.cdiv(E, BLOCK_E)

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
    offset_block_state = off_block_n * D * E

    # compute block ptr and mask
    array_c = tl.arange(0, BLOCK_C)
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)

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

    mask_d = (array_d + offset_block_d) < D
    mask_e = (array_e + offset_block_e) < E

    if USE_LOG_DECAY:
        log_decay = tl.load(LOG_DECAY + off_h).to(tl.float32)
        block_decay = tl.exp(log_decay * BLOCK_C)
        k_decay = tl.exp(log_decay * (BLOCK_C - 1 - tl.arange(0, BLOCK_C)))
        if USE_PAD:
            M = N % BLOCK_C
            last_decay = tl.exp(log_decay * M)
            array = M - 1 - tl.arange(0, BLOCK_C)
            array = tl.where(
                array >= 0, array, 0
            )  # !!! important, otherwise the decay will be nan, nan * 0 != 0
            last_k_decay = tl.exp(log_decay * array)

    state = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)
    array_c = offset_block_n + tl.arange(0, BLOCK_C)

    cnt = offset_block_n
    for i in range(NUM_BLOCK_C):
        mask_c = (array_c + i * BLOCK_C) < N
        k_trans = tl.load(
            k_trans_block_ptr, mask=mask_c[None, :] & mask_d[:, None], other=0.0
        )
        v = tl.load(v_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0)

        if USE_LOG_DECAY:
            if cnt < N:
                # last step
                if USE_PAD:
                    if off_block_n == NUM_BLOCK_N - 1:
                        if cnt + BLOCK_C >= N:
                            k_trans = (k_trans * last_k_decay[None, :]).to(
                                k_trans.dtype
                            )
                            state = last_decay * state + tl.dot(k_trans, v)
                        else:
                            k_trans = (k_trans * k_decay[None, :]).to(k_trans.dtype)
                            state = block_decay * state + tl.dot(k_trans, v)
                    else:
                        k_trans = (k_trans * k_decay[None, :]).to(k_trans.dtype)
                        state = block_decay * state + tl.dot(k_trans, v)
                else:
                    k_trans = (k_trans * k_decay[None, :]).to(k_trans.dtype)
                    state = block_decay * state + tl.dot(k_trans, v)
        else:
            state += tl.dot(k_trans, v)

        k_trans_block_ptr += BLOCK_C * H * D
        v_block_ptr += BLOCK_C * H * E
        cnt += BLOCK_C

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
    key=["B", "N", "H", "D", "E", "USE_CU_SEQLENS", "USE_LOG_DECAY"],
)
@triton.jit
def _lasd_parallel_fwd_state_reduce(
    STATE,  # B H D E
    STATES,  # B H L D E
    LOG_DECAY,  # H
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_LOG_DECAY: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    tl.cdiv(D, BLOCK_D)
    tl.cdiv(E, BLOCK_E)

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

    # compute array for block ptr
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)

    # (BLOCK_D, BLOCK_E)
    states_block_ptr = (
        STATES
        + offset_states
        + (offset_block_d + array_d[:, None]) * E
        + (offset_block_e + array_e[None, :])
    )

    mask_d = (array_d + offset_block_d) < D
    mask_e = (array_e + offset_block_e) < E
    mask = mask_d[:, None] & mask_e[None, :]

    if USE_LOG_DECAY:
        log_decay = tl.load(LOG_DECAY + off_h).to(tl.float32)
        block_decay = tl.exp(log_decay * BLOCK_N)
        L = N % BLOCK_N
        if L == 0:
            last_decay = block_decay
        else:
            last_decay = tl.exp(log_decay * L)

    # compute
    if USE_INITIAL_STATE:
        state_block_ptr = (
            STATE
            + offset_state
            + (offset_block_d + array_d[:, None]) * E
            + (offset_block_e + array_e[None, :])
        )
        state = tl.load(state_block_ptr, mask=mask, other=0.0).to(tl.float32)
    else:
        state = tl.zeros((BLOCK_D, BLOCK_E), dtype=tl.float32)

    for i in range(NUM_BLOCK_N):
        current_state = tl.load(states_block_ptr, mask=mask, other=0.0)

        tl.store(
            states_block_ptr, state.to(states_block_ptr.dtype.element_ty), mask=mask
        )

        if USE_LOG_DECAY:
            if i == NUM_BLOCK_N - 1:
                state = last_decay * state + current_state
            else:
                state = block_decay * state + current_state
        else:
            state += current_state
        states_block_ptr += D * E

    tl.store(states_block_ptr, state.to(states_block_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_C": [16, 32, 64, 128],
            "BLOCK_D": [128],
            "BLOCK_E": [128],
        }
    ),
    key=["B", "N", "H", "D", "E", "USE_CU_SEQLENS", "USE_LOG_DECAY"],
)
@triton.jit
def _lasd_parallel_fwd_inter(
    Q,  # B N H D
    O,  # B N H E
    STATES,  # B H L D E
    LOG_DECAY,  # H
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_LOG_DECAY: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    tl.cdiv(BLOCK_N, BLOCK_C)
    NUM_BLOCK_D = tl.cdiv(D, BLOCK_D)
    tl.cdiv(E, BLOCK_E)

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
    state_block_ptr = (
        STATES
        + offset_state
        + offset_block_state
        + array_d[:, None] * E
        + (offset_block_e + array_e)[None, :]
    )
    mask_e = (offset_block_e + array_e) < E
    mask_c = (offset_block_n + offset_block_c + array_c) < N

    if USE_LOG_DECAY:
        log_decay = tl.load(LOG_DECAY + off_h).to(tl.float32)
        array_c = tl.arange(0, BLOCK_C)
        q_decay = tl.exp(log_decay * (offset_block_c + array_c[:, None] + 1))

    o = tl.load(o_block_ptr, mask=mask_c[:, None] & mask_e[None, :], other=0.0).to(
        tl.float32
    )
    for i in range(NUM_BLOCK_D):
        mask_d = (array_d + i * BLOCK_D) < D
        q = tl.load(q_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)
        state = tl.load(
            state_block_ptr, mask=mask_d[:, None] & mask_e[None, :], other=0.0
        ).to(q.dtype)

        o_ = tl.dot(q, state)
        if USE_LOG_DECAY:
            o_ *= q_decay
        o += o_

        q_block_ptr += BLOCK_D
        state_block_ptr += BLOCK_D * E

    tl.store(
        o_block_ptr,
        o.to(o_block_ptr.dtype.element_ty),
        mask=mask_c[:, None] & mask_e[None, :],
    )


def lasd_parallel_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    b, n, h, d = q.shape
    e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_initial_state = initial_state is not None
    use_ld = ld is not None

    if use_cu_seqlens:
        o = torch.empty((1, n, h, e), dtype=q.dtype, device=q.device)
    else:
        o = torch.empty((b, n, h, e), dtype=q.dtype, device=q.device)

    MAX_BLOCK_N = triton.next_power_of_2(n)
    MAX_BLOCK_C = MAX_BLOCK_N
    MAX_BLOCK_E = triton.next_power_of_2(e)
    MAX_BLOCK_D = triton.next_power_of_2(d)

    if n <= 512:
        BLOCK_N = min(MAX_BLOCK_N, 128)
    else:
        BLOCK_N = 256

    NUM_BLOCK_N = triton.cdiv(n, BLOCK_N)
    use_pad = n % BLOCK_N != 0

    # Step1: Compute intra in parallel, for each chunk, parallel over sub-chunk
    def grid_partial(MAX_BLOCK_C, MAX_BLOCK_E):
        def grid(meta):
            meta["BLOCK_C"] = min(meta["BLOCK_C"], MAX_BLOCK_C)
            meta["BLOCK_E"] = min(meta["BLOCK_E"], MAX_BLOCK_E)
            return (
                b * h * NUM_BLOCK_N,
                triton.cdiv(BLOCK_N, meta["BLOCK_C"]),
                triton.cdiv(e, meta["BLOCK_E"]),
            )

        return grid

    grid = grid_partial(MAX_BLOCK_C, MAX_BLOCK_E)

    _lasd_parallel_fwd_intra[grid](
        Q=q,
        K=k,
        V=v,
        O=o,
        LOG_DECAY=ld,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_LOG_DECAY=use_ld,
        BLOCK_N=BLOCK_N,
    )

    # Step2: Compute local states in parallel
    states = torch.empty(
        (b, h, NUM_BLOCK_N + 1, d, e), dtype=torch.float32, device=q.device
    )

    def grid_partial(MAX_BLOCK_D, MAX_BLOCK_E):
        def grid(meta):
            meta["BLOCK_D"] = min(meta["BLOCK_D"], MAX_BLOCK_D)
            meta["BLOCK_E"] = min(meta["BLOCK_E"], MAX_BLOCK_E)
            return (
                b * h * NUM_BLOCK_N,
                triton.cdiv(d, meta["BLOCK_D"]),
                triton.cdiv(e, meta["BLOCK_E"]),
            )

        return grid

    grid = grid_partial(MAX_BLOCK_D, MAX_BLOCK_E)

    _lasd_parallel_fwd_state_parallel[grid](
        K=k,
        V=v,
        STATES=states,
        LOG_DECAY=ld,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_LOG_DECAY=use_ld,
        USE_PAD=use_pad,
        BLOCK_N=BLOCK_N,
    )

    # Step3: Update local states to get global states
    def grid_partial(MAX_BLOCK_D, MAX_BLOCK_E):
        def grid(meta):
            meta["BLOCK_D"] = min(meta["BLOCK_D"], MAX_BLOCK_D)
            meta["BLOCK_E"] = min(meta["BLOCK_E"], MAX_BLOCK_E)
            return (
                b * h,
                triton.cdiv(d, meta["BLOCK_D"]),
                triton.cdiv(e, meta["BLOCK_E"]),
            )

        return grid

    grid = grid_partial(MAX_BLOCK_D, MAX_BLOCK_E)

    _lasd_parallel_fwd_state_reduce[grid](
        STATE=initial_state,
        STATES=states,
        LOG_DECAY=ld,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_LOG_DECAY=use_ld,
        USE_INITIAL_STATE=use_initial_state,
        BLOCK_N=BLOCK_N,
    )

    # Step4: Compute inter in parallel, for each chunk, parallel over sub-chunk
    def grid_partial(MAX_BLOCK_C, MAX_BLOCK_D, MAX_BLOCK_E):
        def grid(meta):
            meta["BLOCK_C"] = min(meta["BLOCK_C"], MAX_BLOCK_C)
            meta["BLOCK_D"] = min(meta["BLOCK_D"], MAX_BLOCK_D)
            meta["BLOCK_E"] = min(meta["BLOCK_E"], MAX_BLOCK_E)
            return (
                b * h * NUM_BLOCK_N,
                triton.cdiv(BLOCK_N, meta["BLOCK_C"]),
                triton.cdiv(e, meta["BLOCK_E"]),
            )

        return grid

    grid = grid_partial(MAX_BLOCK_C, MAX_BLOCK_D, MAX_BLOCK_E)

    _lasd_parallel_fwd_inter[grid](
        Q=q,
        O=o,
        STATES=states,
        LOG_DECAY=ld,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_LOG_DECAY=use_ld,
        BLOCK_N=BLOCK_N,
    )

    return o, states


########## Fwd end ##########

########## Bwd start ##########
def lasd_parallel_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    dfinal_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    b, n, h, d = q.shape
    e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    dq = torch.empty((b, n, h, d), dtype=q.dtype, device=q.device)
    dk = torch.empty((b, n, h, d), dtype=k.dtype, device=k.device)
    dv = torch.empty((b, n, h, e), dtype=v.dtype, device=v.device)

    dq, _ = lasd_parallel_fwd(
        q=do,
        k=v,
        v=k,
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )

    q = torch.flip(q, [1]).contiguous()
    k = torch.flip(k, [1]).contiguous()
    v = torch.flip(v, [1]).contiguous()
    do = torch.flip(do, [1]).contiguous()

    dk, _ = lasd_parallel_fwd(
        q=v,
        k=do,
        v=q,
        ld=ld,
        initial_state=dfinal_state,
        cu_seqlens=cu_seqlens,
    )
    dk = torch.flip(dk, [1])

    dv, dfinal_states = lasd_parallel_fwd(
        q=k,
        k=q,
        v=do,
        ld=ld,
        initial_state=dfinal_state,
        cu_seqlens=cu_seqlens,
    )
    dv = torch.flip(dv, [1])

    need_dfinal_state = (
        dfinal_state is not None
        and initial_state is not None
        and initial_state.requires_grad
    )

    return dq, dk, dv, dfinal_states[:, :, -1, :, :] if need_dfinal_state else None


########## Bwd end ##########


class LasdParallelFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        ld,
        initial_state=None,
        cu_seqlens=None,
    ):
        # Forward computation
        output, states = lasd_parallel_fwd(
            q=q,
            k=k,
            v=v,
            ld=ld,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

        # Save tensors needed for backward
        # ctx.save_for_backward(q, k, v, ld, initial_state, states, cu_seqlens)
        ctx.save_for_backward(q, k, v, ld, initial_state, cu_seqlens)
        final_state = states[:, :, -1, :, :]
        del states

        return output, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dfinal_state):
        q, k, v, ld, initial_state, cu_seqlens = ctx.saved_tensors

        dq, dk, dv, dinitial_state = lasd_parallel_bwd(
            q=q,
            k=k,
            v=v,
            do=do,
            ld=ld,
            initial_state=initial_state,
            dfinal_state=dfinal_state,
            cu_seqlens=cu_seqlens,
        )

        return (
            dq,
            dk,
            dv,
            None,
            dinitial_state,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def lasd_parallel_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """
    Apply Lightning Attention Parallel with Scalar Decay in Triton.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (H,)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training
        eps: Epsilon for numerical stability
    Returns:
        output: Tensor of shape (B, N, H, E)
        state: Tensor of shape (B, H, D, E)
    """
    if initial_state is not None:
        b = q.shape[0]
        use_cu_seqlens = cu_seqlens is not None
        if use_cu_seqlens:
            b = cu_seqlens.shape[0] - 1
        # treat for varlen training
        if initial_state.shape[0] == 1:
            initial_state = initial_state.squeeze(0)
        if len(initial_state.shape) == 3:
            initial_state = repeat(initial_state, "h d e -> b h d e", b=b).contiguous()

    return LasdParallelFunction.apply(
        q,
        k,
        v,
        ld,
        initial_state,
        cu_seqlens,
    )


if __name__ == "__main__":
    import torch.nn.functional as F

    b, n, h, d = 2, 16, 12, 64
    e = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    q = torch.randn(b, n, h, d, device=device, dtype=dtype).requires_grad_(True)
    k = torch.randn(b, n, h, d, device=device, dtype=dtype).requires_grad_(True)
    v = torch.randn(b, n, h, e, device=device, dtype=dtype).requires_grad_(True)
    ld = F.logsigmoid(torch.randn(h, device=device))
    initial_state = torch.randn(b, h, d, e, device=device, dtype=dtype).requires_grad_(
        True
    )
    output, final_state = lasd_parallel_triton(q, k, v, ld, initial_state)
    loss = output.sum() + final_state.sum()
    loss.backward()
