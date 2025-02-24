from typing import Optional

import torch
import triton
import triton.language as tl

from xopes.utils import generate_configs


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
def _lasd_parallel_intra(
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

    if REVERSE:
        stride = -1
        NUM_BLOCK_C = tl.cdiv(BLOCK_N, BLOCK_C)
        NUM_LOOP = NUM_BLOCK_C - off_block_c
    else:
        stride = 1
        NUM_LOOP = off_block_c + 1

    for i in range(NUM_BLOCK_D):
        mask_d = (i * BLOCK_D + array_d) < D
        q = tl.load(q_block_ptr, mask=mask_c[:, None] & mask_d[None, :], other=0.0)

        if REVERSE:
            # TODO: test this
            # array_kv = BLOCK_N - 1 - array_c

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

        for j in range(NUM_LOOP):
            mask_kv = (array_kv < N) & (array_kv >= 0)

            k_trans = tl.load(
                k_trans_block_ptr, mask=mask_kv[None, :] & mask_d[:, None], other=0.0
            )
            v = tl.load(v_block_ptr, mask=mask_kv[:, None] & mask_e[None, :], other=0.0)

            score = tl.dot(q, k_trans)
            diff = (array_q[:, None] - array_kv[None, :]) * stride
            if not USE_LOG_DECAY:
                score = tl.where(diff >= 0, score, 0.0)
            else:
                decay = log_decay * diff
                decay = tl.exp(tl.where(diff >= 0, decay, float("-inf")))
                score *= decay
            o += tl.dot(score.to(v.dtype), v)

            k_trans_block_ptr += BLOCK_C * H * D * stride
            v_block_ptr += BLOCK_C * H * E * stride
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
            "BLOCK_C": [16, 32, 64, 128],
            "BLOCK_D": [128],
            "BLOCK_E": [128],
        }
    ),
    key=["B", "N", "H", "D", "E", "USE_CU_SEQLENS", "USE_LOG_DECAY"],
)
@triton.jit
def _lasd_parallel_state_parallel(
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
    REVERSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCK_C = tl.cdiv(BLOCK_N, BLOCK_C)
    # tl.cdiv(D, BLOCK_D)
    # tl.cdiv(E, BLOCK_E)

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

    # # for reverse, local loop start from the last block
    if REVERSE:
        # index map: 0: Sn-1, 1: Sn-2, 2: Sn-3, ..., n-1: S0, n: Null
        stride = -1
        # array_c = BLOCK_N - 1 - array_c
        array_c = BLOCK_N - BLOCK_C + array_c
        # offset_block_state = (NUM_BLOCK_N - 1 - off_block_n) * D * E
        # tl.device_print("offset_block_state", offset_block_state)
    else:
        # index map: 0: S0, 1: S1, 2: S2, ..., n-1: Sn-1, n: Null
        stride = 1
        array_c = array_c
        # offset_block_state = off_block_n * D * E

    # index map, assume index starts from 1
    # reverse = false: 1: S1, 2: S2, ..., n: Sn, n + 1: Null
    # reverse = true:  1: dS1, 2: dS2, ..., n: dSn, n + 1: Null
    # stride = 1
    # array_c = array_c
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

    mask_d = (array_d + offset_block_d) < D
    mask_e = (array_e + offset_block_e) < E

    if USE_LOG_DECAY:
        log_decay = tl.load(LOG_DECAY + off_h).to(tl.float32)
        block_decay = tl.exp(log_decay * BLOCK_C)
        if REVERSE:
            k_decay = tl.exp(log_decay * tl.arange(0, BLOCK_C))
        else:
            k_decay = tl.exp(log_decay * (BLOCK_C - 1 - tl.arange(0, BLOCK_C)))

        if USE_PAD:
            M = N % BLOCK_C
            last_decay = tl.exp(log_decay * M)
            if REVERSE:
                array = tl.arange(0, BLOCK_C)
            else:
                array = M - 1 - tl.arange(0, BLOCK_C)
            array = tl.where(
                array >= 0, array, 0
            )  # !!! important, otherwise the decay will be nan, nan * 0 != 0
            last_k_decay = tl.exp(log_decay * array)

    state = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)
    # array_c = offset_block_n + tl.arange(0, BLOCK_C)

    cnt = offset_block_n
    for i in range(NUM_BLOCK_C):
        mask_c = (offset_block_n + array_c) < N
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

        k_trans_block_ptr += BLOCK_C * H * D * stride
        v_block_ptr += BLOCK_C * H * E * stride
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
    key=["B", "N", "H", "D", "E", "USE_CU_SEQLENS", "USE_LOG_DECAY"],
)
@triton.jit
def _lasd_parallel_state_reduce(
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
    REVERSE: tl.constexpr,
    TRANS: tl.constexpr,
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

    # compute array for block ptr
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)

    # if reverse, start from the last block
    if REVERSE:
        stride = -1
        states_start = NUM_BLOCK_N * D * E
    else:
        stride = 1
        states_start = 0
    # stride = 1
    # states_start = 0

    # (BLOCK_D, BLOCK_E)
    states_block_ptr = (
        STATES
        + offset_states
        + states_start
        + (offset_block_d + array_d[:, None]) * E
        + (offset_block_e + array_e[None, :])
    )

    mask_d = (array_d + offset_block_d) < D
    mask_e = (array_e + offset_block_e) < E
    mask = mask_d[:, None] & mask_e[None, :]

    c = 1.0
    if USE_LOG_DECAY:
        log_decay = tl.load(LOG_DECAY + off_h).to(tl.float32)
        block_decay = tl.exp(log_decay * BLOCK_N)
        L = N % BLOCK_N
        if L == 0:
            last_decay = block_decay
        else:
            last_decay = tl.exp(log_decay * L)

        # !!! important
        if REVERSE:
            c = tl.exp(-log_decay)

    # compute
    if USE_INITIAL_STATE:
        if TRANS:
            state_block_ptr = (
                STATE
                + offset_state
                + (offset_block_d + array_d[None, :]) * E
                + (offset_block_e + array_e[:, None])
            )
        else:
            state_block_ptr = (
                STATE
                + offset_state
                + (offset_block_d + array_d[:, None]) * E
                + (offset_block_e + array_e[None, :])
            )
        # !!! important
        state = tl.load(state_block_ptr, mask=mask, other=0.0).to(tl.float32) * c
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
        states_block_ptr += D * E * stride

    # !!! important
    if REVERSE:
        state *= 1 / c

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
def _lasd_parallel_inter(
    Q,  # B N H D
    O,  # B N H E
    STATES,  # B H L D E if not trans_states, B H L E D if trans_states
    LOG_DECAY,  # H
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_LOG_DECAY: tl.constexpr,
    REVERSE: tl.constexpr,
    TRANS_STATES: tl.constexpr,
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

    offset_state = off_bh * (NUM_BLOCK_N + 1) * D * E
    # if REVERSE:
    #     # if reverse, the i'th q chunk aligns with the i + 1'th state:
    #     # q chunk:        0, 1, ..., n - 1
    #     # state chunk: 0, 1, 2, ..., n
    #     offset_block_state = (off_block_n + 1) * D * E
    # else:
    #     # if not reverse, the i'th q chunk aligns with the i'th state:
    #     # q chunk:     0, 1, ..., n - 1
    #     # state chunk: 0, 1, ..., n - 1, n
    #     offset_block_state = off_block_n * D * E
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
    if TRANS_STATES:
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
    # tl.static_print("aaa", q_block_ptr, state_block_ptr)
    # state_block_ptr = (
    #     STATES
    #     + offset_state
    #     + offset_block_state
    #     + array_d[:, None] * E
    #     + (offset_block_e + array_e)[None, :]
    # )
    mask_e = (offset_block_e + array_e) < E
    mask_c = (offset_block_n + offset_block_c + array_c) < N

    if USE_LOG_DECAY:
        log_decay = tl.load(LOG_DECAY + off_h).to(tl.float32)
        array_c = tl.arange(0, BLOCK_C)
        if REVERSE:
            q_decay = tl.exp(
                log_decay * (BLOCK_N - (offset_block_c + array_c[:, None]))
            )
        else:
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


########## pytorch implementation reference ##########
def lasd_intra_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
):
    def _lasd_intra(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ld: Optional[torch.Tensor] = None,
    ):
        b, n, h, d = k.shape
        e = v.shape[-1]
        dtype = q.dtype

        q = q.float()
        k = k.float()
        v = v.float()

        if ld is None:
            decay = (
                torch.ones((), dtype=torch.float32, device=q.device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
        else:
            decay = torch.exp(ld.float()).unsqueeze(-1).unsqueeze(-1)

        state = torch.zeros(b, h, d, e, dtype=torch.float32, device=q.device)
        o = []
        for i in range(n):
            state_ = torch.einsum(
                "b h d, b h e -> b h d e", k[:, i, :, :], v[:, i, :, :]
            )
            state = decay * state + state_
            o_ = torch.einsum(
                "b h d, b h d e -> b h e", q[:, i, :, :], state
            ).unsqueeze(1)
            o.append(o_)
        o = torch.cat(o, dim=1)

        return o.to(dtype)

    if reverse:
        q = torch.flip(q, dims=[1])
        k = torch.flip(k, dims=[1])
        v = torch.flip(v, dims=[1])

    o = _lasd_intra(q, k, v, ld)

    if reverse:
        o = torch.flip(o, dims=[1])

    return o


def compute_states(
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    BLOCK_N: int = 256,
    reverse: bool = False,
):
    b, n, h, d = k.shape
    e = v.shape[-1]

    if ld is None:
        decay = (
            torch.ones((), dtype=torch.float32, device=k.device)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
    else:
        decay = torch.exp(ld.float()).unsqueeze(-1).unsqueeze(-1)

    # local state
    l = (n + BLOCK_N - 1) // BLOCK_N
    local_states = []
    for i in range(l):
        start = i * BLOCK_N
        end = min(start + BLOCK_N, n)
        m = end - start
        k_i = k[:, start:end, :, :]
        v_i = v[:, start:end, :, :]
        state_i = torch.zeros(b, h, d, e, dtype=torch.float32, device=k.device)
        for j in range(m):
            state_ = torch.einsum(
                "b h d, b h e -> b h d e", k_i[:, j, :, :], v_i[:, j, :, :]
            )
            state_i = decay * state_i + state_
        local_states.append(state_i.unsqueeze(2))
    local_states = torch.cat(local_states, dim=2)

    # global state
    if initial_state is None:
        state = torch.zeros(b, h, d, e, dtype=torch.float32, device=k.device)
    else:
        state = initial_state.float()

    global_states = [state.unsqueeze(2)]
    for i in range(n):
        state_ = torch.einsum("b h d, b h e -> b h d e", k[:, i, :, :], v[:, i, :, :])
        state = decay * state + state_
        if (i + 1) % BLOCK_N == 0 or (i == n - 1):
            global_states.append(state.unsqueeze(2))
    global_states = torch.cat(global_states, dim=2)

    return local_states, global_states
