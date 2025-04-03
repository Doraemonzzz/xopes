from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import repeat

from xopes.ops.cumsum import cumsum_fn
from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
        }
    ),
    key=[
        "B",
        "H",
        "D",
        "E",
        "USE_INITIAL_STATE",
        "USE_CU_SEQLENS",
    ],
)
@triton.jit
def _lavd_recurrence_fwd(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    LOG_DECAY_K,  # B N H D
    LOG_DECAY_V,  # B N H E
    STATE,  # B H D E
    CU_SEQLENS,  # M
    O,  # B N H E
    FINAL_STATE,  # B H D E
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_DECAY_K: tl.constexpr,
    USE_DECAY_V: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    SHARE_K: tl.constexpr,
    SHARE_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    # compute offset
    if not USE_CU_SEQLENS:
        offset_qk = off_b * N * H * D + off_h * D
        offset_vo = off_b * N * H * E + off_h * E
    else:
        start = tl.load(CU_SEQLENS + off_b)
        end = tl.load(CU_SEQLENS + off_b + 1)
        N = end - start
        offset_qk = start * H * D + off_h * D
        offset_vo = start * H * E + off_h * E
    offset_state = off_b * H * D * E + off_h * D * E

    # compute block ptr
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    q_block_ptr = Q + offset_qk + array_d
    if not SHARE_K:
        k_block_ptr = K + offset_qk + array_d
    if not SHARE_V:
        v_block_ptr = V + offset_vo + array_e
    o_block_ptr = O + offset_vo + array_e
    if USE_DECAY_K:
        log_decay_k_block_ptr = LOG_DECAY_K + offset_qk + array_d
    if USE_DECAY_V:
        log_decay_v_block_ptr = LOG_DECAY_V + offset_vo + array_e
    mask_d = array_d < D
    mask_e = array_e < E

    if USE_INITIAL_STATE:
        state_block_ptr = STATE + offset_state + array_d[:, None] * E + array_e[None, :]
        state = tl.load(
            state_block_ptr, mask=mask_d[:, None] & mask_e[None, :], other=0.0
        ).to(
            tl.float32
        )  # D E
    else:
        state = tl.zeros((BLOCK_D, BLOCK_E), dtype=tl.float32)

    final_state_block_ptr = (
        FINAL_STATE + offset_state + array_d[:, None] * E + array_e[None, :]
    )

    # compute
    for i in range(N):
        # load
        q = tl.load(q_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
        if not SHARE_K:
            k = tl.load(k_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
            if USE_DECAY_K:
                log_decay_k = tl.load(log_decay_k_block_ptr, mask=mask_d, other=0.0).to(
                    tl.float32
                )
                decay_k = tl.exp(log_decay_k)
        else:
            log_decay_k = tl.load(log_decay_k_block_ptr, mask=mask_d, other=0.0).to(
                tl.float32
            )
            decay_k = tl.exp(log_decay_k)
            k = 1 - decay_k

        if not SHARE_V:
            v = tl.load(v_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
            if USE_DECAY_V:
                log_decay_v = tl.load(log_decay_v_block_ptr, mask=mask_e, other=0.0).to(
                    tl.float32
                )
                decay_v = tl.exp(log_decay_v)
        else:
            log_decay_v = tl.load(log_decay_v_block_ptr, mask=mask_e, other=0.0).to(
                tl.float32
            )
            decay_v = tl.exp(log_decay_v)
            v = 1 - decay_v

        if USE_DECAY_K:
            state = decay_k[:, None] * state
        if USE_DECAY_V:
            state = state * decay_v[None, :]

        state += k[:, None] * v[None, :]
        o = tl.sum(q[:, None] * state, axis=0)

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask_e)

        # update
        q_block_ptr += H * D
        o_block_ptr += H * E
        if not SHARE_K:
            k_block_ptr += H * D
        if not SHARE_V:
            v_block_ptr += H * E
        if USE_DECAY_K:
            log_decay_k_block_ptr += H * D
        if USE_DECAY_V:
            log_decay_v_block_ptr += H * E

    tl.store(
        final_state_block_ptr,
        state.to(final_state_block_ptr.dtype.element_ty),
        mask=mask_d[:, None] & mask_e[None, :],
    )


def lavd_recurrence_fwd(
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    b, n, h, d = q.shape
    if ldv is not None:
        e = ldv.shape[-1]
    else:
        e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    share_k = k is None
    share_v = v is None
    use_initial_state = initial_state is not None
    final_state = torch.empty((b, h, d, e), dtype=torch.float32, device=q.device)
    BLOCK_D = triton.next_power_of_2(d)
    BLOCK_E = triton.next_power_of_2(e)

    if use_cu_seqlens:
        o = torch.empty((1, n, h, e), dtype=q.dtype, device=q.device)
    else:
        o = torch.empty((b, n, h, e), dtype=q.dtype, device=q.device)

    def grid(meta):
        return (b, h)

    _lavd_recurrence_fwd[grid](
        Q=q,
        K=k,
        V=v,
        LOG_DECAY_K=ldk,
        LOG_DECAY_V=ldv,
        STATE=initial_state,
        CU_SEQLENS=cu_seqlens,
        O=o,
        FINAL_STATE=final_state,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_DECAY_K=use_ldk,
        USE_DECAY_V=use_ldv,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        SHARE_K=share_k,
        SHARE_V=share_v,
        BLOCK_D=BLOCK_D,
        BLOCK_E=BLOCK_E,
    )

    return o, final_state


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
        }
    ),
    key=["B", "H", "D", "E", "USE_INITIAL_STATE", "USE_CU_SEQLENS"],
)
@triton.jit
def _lavd_recurrence_bwd_dq(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    LOG_DECAY_K,  # B N H D
    LOG_DECAY_V,  # B N H E
    STATE,  # B H D E
    CU_SEQLENS,  # M
    FINAL_STATE,  # B H D E
    DO,  # B N H E
    DSTATE,  # B H D E
    DQ,  # B N H D
    DK,  # B N H D
    DV,  # B N H E
    DINITIAL_STATE,  # B H D E
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_DECAY_K: tl.constexpr,
    USE_DECAY_V: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_DFINAL_STATE: tl.constexpr,
    SHARE_K: tl.constexpr,
    SHARE_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    # compute offset
    if not USE_CU_SEQLENS:
        offset_qk = off_b * N * H * D + off_h * D
        offset_vo = off_b * N * H * E + off_h * E
    else:
        start = tl.load(CU_SEQLENS + off_b)
        end = tl.load(CU_SEQLENS + off_b + 1)
        N = end - start
        offset_qk = start * H * D + off_h * D
        offset_vo = start * H * E + off_h * E
    offset_state = off_b * H * D * E + off_h * D * E

    # compute block ptr
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    if not SHARE_K:
        k_block_ptr = K + offset_qk + array_d
    if not SHARE_V:
        v_block_ptr = V + offset_vo + array_e
    do_block_ptr = DO + offset_vo + array_e
    dq_block_ptr = DQ + offset_qk + array_d
    if USE_DECAY_K:
        log_decay_k_block_ptr = LOG_DECAY_K + offset_qk + array_d
    if USE_DECAY_V:
        log_decay_v_block_ptr = LOG_DECAY_V + offset_vo + array_e
    mask_d = array_d < D
    mask_e = array_e < E

    if USE_INITIAL_STATE:
        state_block_ptr = STATE + offset_state + array_d[:, None] * E + array_e[None, :]
        state = tl.load(
            state_block_ptr, mask=mask_d[:, None] & mask_e[None, :], other=0.0
        ).to(
            tl.float32
        )  # D E
    else:
        state = tl.zeros((BLOCK_D, BLOCK_E), dtype=tl.float32)

    # compute
    for i in range(N):
        # load
        do = tl.load(do_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        if not SHARE_K:
            k = tl.load(k_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
            if USE_DECAY_K:
                log_decay_k = tl.load(log_decay_k_block_ptr, mask=mask_d, other=0.0).to(
                    tl.float32
                )
                decay_k = tl.exp(log_decay_k)
        else:
            log_decay_k = tl.load(log_decay_k_block_ptr, mask=mask_d, other=0.0).to(
                tl.float32
            )
            decay_k = tl.exp(log_decay_k)
            k = 1 - decay_k

        if not SHARE_V:
            v = tl.load(v_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
            if USE_DECAY_V:
                log_decay_v = tl.load(log_decay_v_block_ptr, mask=mask_e, other=0.0).to(
                    tl.float32
                )
                decay_v = tl.exp(log_decay_v)
        else:
            log_decay_v = tl.load(log_decay_v_block_ptr, mask=mask_e, other=0.0).to(
                tl.float32
            )
            decay_v = tl.exp(log_decay_v)
            v = 1 - decay_v

        if USE_DECAY_K:
            state = decay_k[:, None] * state
        if USE_DECAY_V:
            state = state * decay_v[None, :]

        state += k[:, None] * v[None, :]
        dq = tl.sum(do[None, :] * state, axis=-1)

        tl.store(dq_block_ptr, dq.to(dq_block_ptr.dtype.element_ty), mask=mask_d)

        # update
        do_block_ptr += H * E
        dq_block_ptr += H * D
        if not SHARE_K:
            k_block_ptr += H * D
        if not SHARE_V:
            v_block_ptr += H * E
        if USE_DECAY_K:
            log_decay_k_block_ptr += H * D
        if USE_DECAY_V:
            log_decay_v_block_ptr += H * E

    final_state_block_ptr = (
        FINAL_STATE + offset_state + array_d[:, None] * E + array_e[None, :]
    )
    tl.store(
        final_state_block_ptr,
        state.to(final_state_block_ptr.dtype.element_ty),
        mask=mask_d[:, None] & mask_e[None, :],
    )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
        }
    ),
    key=["B", "H", "D", "E", "USE_INITIAL_STATE", "USE_CU_SEQLENS"],
)
@triton.jit
def _lavd_recurrence_bwd_dk_dv(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    LOG_DECAY_K,  # B N H D
    LOG_DECAY_V,  # B N H E
    STATE,  # B H D E
    CU_SEQLENS,  # M
    FINAL_STATE,  # B H D E
    DO,  # B N H E
    DSTATE,  # B H D E
    DQ,  # B N H D
    DK,  # B N H D
    DV,  # B N H E
    DINITIAL_STATE,  # B H D E
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_DECAY_K: tl.constexpr,
    USE_DECAY_V: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_DFINAL_STATE: tl.constexpr,
    SHARE_K: tl.constexpr,
    SHARE_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    # compute offset
    if not USE_CU_SEQLENS:
        offset_qk = off_b * N * H * D + N * H * D + off_h * D
        offset_vo = off_b * N * H * E + N * H * E + off_h * E
        off_b * N * H + N * H + off_h
    else:
        start = tl.load(CU_SEQLENS + off_b + 1)
        end = tl.load(CU_SEQLENS + off_b)
        N = start - end
        offset_qk = start * H * D + off_h * D
        offset_vo = start * H * E + off_h * E
        start * H + off_h
    offset_state = off_b * H * D * E + off_h * D * E

    # compute block ptr
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    q_block_ptr = Q + offset_qk + array_d
    if not SHARE_K:
        k_block_ptr = K + offset_qk + array_d
    if not SHARE_V:
        v_block_ptr = V + offset_vo + array_e
    do_block_ptr = DO + offset_vo + array_e
    dk_block_ptr = DK + offset_qk + array_d
    dv_block_ptr = DV + offset_vo + array_e
    if USE_DECAY_K:
        log_decay_k_block_ptr = LOG_DECAY_K + offset_qk + array_d
    if USE_DECAY_V:
        log_decay_v_block_ptr = LOG_DECAY_V + offset_vo + array_e
    mask_d = array_d < D
    mask_e = array_e < E

    if USE_DFINAL_STATE:
        dstate_block_ptr = (
            DSTATE + offset_state + array_d[:, None] * E + array_e[None, :]
        )
        dstate = tl.load(
            dstate_block_ptr, mask=mask_d[:, None] & mask_e[None, :], other=0.0
        ).to(
            tl.float32
        )  # D E
    else:
        dstate = tl.zeros((BLOCK_D, BLOCK_E), dtype=tl.float32)

    if SHARE_K:
        decay_k = tl.full((BLOCK_D,), 1.0, dtype=tl.float32)
    if SHARE_V:
        decay_v = tl.full((BLOCK_E,), 1.0, dtype=tl.float32)
    # compute
    for i in range(N):
        # update
        q_block_ptr -= H * D
        if not SHARE_K:
            k_block_ptr -= H * D

        if not SHARE_V:
            v_block_ptr -= H * E

        do_block_ptr -= H * E
        dk_block_ptr -= H * D
        dv_block_ptr -= H * E
        # load
        do = tl.load(do_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        q = tl.load(q_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
        # !!! IMPORTANT
        if i > 0:
            if USE_DECAY_K:
                if not SHARE_K:
                    log_decay_k_block_ptr -= H * D
                    log_decay_k = tl.load(
                        log_decay_k_block_ptr, mask=mask_d, other=0.0
                    ).to(tl.float32)
                    decay_k = tl.exp(log_decay_k)
                dstate = decay_k[:, None] * dstate

            if USE_DECAY_V:
                if not SHARE_V:
                    log_decay_v_block_ptr -= H * E
                    log_decay_v = tl.load(
                        log_decay_v_block_ptr, mask=mask_e, other=0.0
                    ).to(tl.float32)
                    decay_v = tl.exp(log_decay_v)
                dstate = dstate * decay_v[None, :]

        dstate += q[:, None] * do[None, :]
        # compute k and v
        if not SHARE_K:
            k = tl.load(k_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
        else:
            log_decay_k_block_ptr -= H * D
            log_decay_k = tl.load(log_decay_k_block_ptr, mask=mask_d, other=0.0).to(
                tl.float32
            )
            decay_k = tl.exp(log_decay_k)
            k = 1 - decay_k
        if not SHARE_V:
            v = tl.load(v_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        else:
            log_decay_v_block_ptr -= H * E
            log_decay_v = tl.load(log_decay_v_block_ptr, mask=mask_e, other=0.0).to(
                tl.float32
            )
            decay_v = tl.exp(log_decay_v)
            v = 1 - decay_v
        dk = tl.sum(dstate * v[None, :], axis=-1)
        dv = tl.sum(dstate * k[:, None], axis=0)
        tl.store(dk_block_ptr, dk.to(dk_block_ptr.dtype.element_ty), mask=mask_d)
        tl.store(dv_block_ptr, dv.to(dv_block_ptr.dtype.element_ty), mask=mask_e)

    # !!! IMPORTANT
    if USE_DECAY_K:
        if not SHARE_K:
            log_decay_k_block_ptr -= H * D
            log_decay_k = tl.load(log_decay_k_block_ptr).to(tl.float32)
            decay_k = tl.exp(log_decay_k)
        dstate = decay_k[:, None] * dstate

    if USE_DECAY_V:
        if not SHARE_V:
            log_decay_v_block_ptr -= H * E
            log_decay_v = tl.load(log_decay_v_block_ptr).to(tl.float32)
            decay_v = tl.exp(log_decay_v)
        dstate = dstate * decay_v[None, :]

    dinitial_state_block_ptr = (
        DINITIAL_STATE + offset_state + array_d[:, None] * E + array_e[None, :]
    )
    tl.store(
        dinitial_state_block_ptr,
        dstate.to(dinitial_state_block_ptr.dtype.element_ty),
        mask=mask_d[:, None] & mask_e[None, :],
    )


def lavd_recurrence_bwd(
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    do: Optional[torch.Tensor] = None,
    o: Optional[torch.Tensor] = None,
    dfinal_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    b, n, h, d = q.shape
    if ldv is not None:
        e = ldv.shape[-1]
    else:
        e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    share_k = k is None
    share_v = v is None
    use_initial_state = initial_state is not None
    use_dfinal_state = dfinal_state is not None
    BLOCK_D = triton.next_power_of_2(d)
    BLOCK_E = triton.next_power_of_2(e)

    dq = torch.empty_like(q)
    if share_k:
        dk = torch.empty_like(ldk)
        k = 1 - torch.exp(ldk.float())
    else:
        dk = torch.empty_like(k)

    if share_v:
        dv = torch.empty_like(ldv)
        v = 1 - torch.exp(ldv.float())
    else:
        dv = torch.empty_like(v)

    dinitial_state = torch.empty(b, h, d, e, device=q.device, dtype=torch.float32)

    def grid(meta):
        return (b, h)

    _lavd_recurrence_bwd_dq[grid](
        Q=q,
        K=k,
        V=v,
        LOG_DECAY_K=ldk,
        LOG_DECAY_V=ldv,
        STATE=initial_state,
        CU_SEQLENS=cu_seqlens,
        FINAL_STATE=final_state,
        DO=do,
        DSTATE=dfinal_state,
        DQ=dq,
        DK=dk,
        DV=dv,
        DINITIAL_STATE=dinitial_state,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_DECAY_K=use_ldk,
        USE_DECAY_V=use_ldv,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        USE_DFINAL_STATE=use_dfinal_state,
        SHARE_K=share_k,
        SHARE_V=share_v,
        BLOCK_D=BLOCK_D,
        BLOCK_E=BLOCK_E,
    )

    _lavd_recurrence_bwd_dk_dv[grid](
        Q=q,
        K=k,
        V=v,
        LOG_DECAY_K=ldk,
        LOG_DECAY_V=ldv,
        STATE=initial_state,
        CU_SEQLENS=cu_seqlens,
        FINAL_STATE=final_state,
        DO=do,
        DSTATE=dfinal_state,
        DQ=dq,
        DK=dk,
        DV=dv,
        DINITIAL_STATE=dinitial_state,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_DECAY_K=use_ldk,
        USE_DECAY_V=use_ldv,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        USE_DFINAL_STATE=use_dfinal_state,
        SHARE_K=share_k,
        SHARE_V=share_v,
        BLOCK_D=BLOCK_D,
        BLOCK_E=BLOCK_E,
    )

    if dfinal_state is not None:
        # b h 1 d
        dldk_state = (final_state * dfinal_state).sum(dim=-1).unsqueeze(1)
        # b h 1 e
        dldv_state = (final_state * dfinal_state).sum(dim=-2).unsqueeze(1)

    dldk = None
    if use_ldk:
        dldk = q * dq - k * dk
        if cu_seqlens is not None:
            dldk = dldk.squeeze(0)
            b = cu_seqlens.shape[0] - 1
            array = []
            for i in range(b):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                dldk_ = cumsum_fn(dldk[start:end], dim=0, reverse=True)
                if dfinal_state is not None:
                    dldk_ = dldk_ + dldk_state[i]
                array.append(dld_)
            dldk = torch.cat(array, dim=0)
            dldk = dldk.unsqueeze(0)
        else:
            dldk = cumsum_fn(dldk, dim=1, reverse=True)

            if dfinal_state is not None:
                dldk = dldk + dldk_state

    dldv = None
    if use_ldv:
        dldv = o * do - v * dv
        if cu_seqlens is not None:
            dldv = dldv.squeeze(0)
            b = cu_seqlens.shape[0] - 1
            array = []
            for i in range(b):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                dldv_ = cumsum_fn(dldv[start:end], dim=0, reverse=True)
                if dfinal_state is not None:
                    dldv_ = dldv_ + dldv_state[i]
                array.append(dld_)
            dldv = torch.cat(array, dim=0)
            dldv = dldv.unsqueeze(0)
        else:
            dldv = cumsum_fn(dldv, dim=1, reverse=True)

            if dfinal_state is not None:
                dldv = dldv + dldv_state

    if share_k:
        # k = 1 - exp(ldk)
        dldk += dk * (-torch.exp(ldk))
        dk = None

    if share_v:
        # v = 1 - exp(ldv)
        dldv += dv * (-torch.exp(ldv))
        dv = None

    dinitial_state = dinitial_state if use_initial_state else None

    return dq, dk, dv, dldk, dldv, dinitial_state


class LavdRecurrenceFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        ldk,
        ldv,
        use_ldk=True,
        use_ldv=False,
        initial_state=None,
        cu_seqlens=None,
    ):
        # Forward computation
        output, final_state = lavd_recurrence_fwd(
            q=q,
            k=k,
            v=v,
            ldk=ldk,
            ldv=ldv,
            use_ldk=use_ldk,
            use_ldv=use_ldv,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

        # Save tensors needed for backward
        ctx.save_for_backward(
            q, k, v, ldk, ldv, initial_state, final_state, cu_seqlens, output
        )
        ctx.use_ldk = use_ldk
        ctx.use_ldv = use_ldv

        return output, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dfinal_state):
        (
            q,
            k,
            v,
            ldk,
            ldv,
            initial_state,
            final_state,
            cu_seqlens,
            output,
        ) = ctx.saved_tensors
        use_ldk = ctx.use_ldk
        use_ldv = ctx.use_ldv

        dq, dk, dv, dldk, dldv, dinitial_state = lavd_recurrence_bwd(
            q=q,
            k=k,
            v=v,
            ldk=ldk,
            ldv=ldv,
            use_ldk=use_ldk,
            use_ldv=use_ldv,
            initial_state=initial_state,
            final_state=final_state,
            do=do,
            o=output,
            dfinal_state=dfinal_state,
            cu_seqlens=cu_seqlens,
        )

        return (
            dq,
            dk,
            dv,
            dldk,
            dldv,
            None,
            None,
            dinitial_state,
            None,
        )


def lavd_recurrence_triton(
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Lightning Attention with Vector Decay in Pytorch.

    Args:
        q: Query tensor, shape (B, N, H, D)
        k: Key tensor, shape (B, N, H, D)
        v: Value tensor, shape (B, N, H, E)
        ldk: Log Decay vector for key, shape (B, N, H, D), if not provided uses log(1 - exp(k))
        ldv: Log Decay vector for value, shape (B, N, H, E), if not provided uses log(1 - exp(v))
        use_ldk: Whether to use log decay for key
        use_ldv: Whether to use log decay for value
        initial_state: Initial state tensor, shape (B, H, D, E) or (H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        Output tensor, shape (B, N, H, E)
        State tensor, shape (B, H, D, E)
    """
    b = q.shape[0]
    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1
    if initial_state is not None:
        initial_state = initial_state.squeeze(0)
        # treat for varlen training
        if len(initial_state.shape) == 3:
            initial_state = repeat(initial_state, "h d e -> b h d e", b=b).contiguous()

    return LavdRecurrenceFunction.apply(
        q,
        k,
        v,
        ldk,
        ldv,
        use_ldk,
        use_ldv,
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
    ld = F.logsigmoid(torch.randn(b, n, h, device=device))
    initial_state = torch.randn(b, h, d, e, device=device, dtype=dtype).requires_grad_(
        True
    )
    output, final_state = lavd_recurrence_triton(q, k, v, ld, initial_state)
    loss = output.sum() + final_state.sum()
    loss.backward()
