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
def _krcl_recurrence_fwd(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    O,  # B N H E
    LOG_DECAY,  # B N H
    ALPHA,  # B N H
    BETA,  # B N H
    STATE, # B H D E
    FINAL_STATE,  # B H D E
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_Q: tl.constexpr,
    USE_ALPHA: tl.constexpr,
    USE_BETA: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    # compute offset
    if not USE_CU_SEQLENS:
        offset_qk = off_b * N * H * D + off_h * D
        offset_vo = off_b * N * H * E + off_h * E
        offset_log_decay = off_b * N * H + off_h
    else:
        start = tl.load(CU_SEQLENS + off_b)
        end = tl.load(CU_SEQLENS + off_b + 1)
        N = end - start
        offset_qk = start * H * D + off_h * D
        offset_vo = start * H * E + off_h * E
        offset_log_decay = start * H + off_h
    offset_state = off_b * H * D * E + off_h * D * E

    # compute block ptr
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    if USE_Q:
        q_block_ptr = Q + offset_qk + array_d
    k_block_ptr = K + offset_qk + array_d
    v_block_ptr = V + offset_vo + array_e
    o_block_ptr = O + offset_vo + array_e
    log_decay_block_ptr = LOG_DECAY + offset_log_decay
    if USE_ALPHA:
        alpha_block_ptr = ALPHA + offset_log_decay
    if USE_BETA:
        beta_block_ptr = BETA + offset_log_decay
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
        k = tl.load(k_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
        if USE_Q:
            q = tl.load(q_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
        else:
            q = k
        v = tl.load(v_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        if USE_ALPHA:
            alpha = tl.load(alpha_block_ptr).to(tl.float32)
            q = q * alpha
        if USE_BETA:
            beta = tl.load(beta_block_ptr).to(tl.float32)
            k = k * beta
        log_decay = tl.load(log_decay_block_ptr).to(tl.float32)
        decay = tl.exp(log_decay)

        # update state
        state *= decay
        o = v - tl.sum(q[:, None] * state, axis=0)
        state_ = k[:, None] * o[None, :]
        state += state_

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask_e)

        # update
        if USE_Q:
            q_block_ptr += H * D
        k_block_ptr += H * D
        v_block_ptr += H * E
        o_block_ptr += H * E
        log_decay_block_ptr += H
        if USE_ALPHA:
            alpha_block_ptr += H
        if USE_BETA:
            beta_block_ptr += H

    tl.store(
        final_state_block_ptr,
        state.to(final_state_block_ptr.dtype.element_ty),
        mask=mask_d[:, None] & mask_e[None, :],
    )


def krcl_recurrence_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    b, n, h, d = q.shape
    e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_initial_state = initial_state is not None
    use_q = q is not None
    use_alpha = alpha is not None
    use_beta = beta is not None
    final_state = torch.empty((b, h, d, e), dtype=torch.float32, device=q.device)
    BLOCK_D = triton.next_power_of_2(d)
    BLOCK_E = triton.next_power_of_2(e)

    if use_cu_seqlens:
        o = torch.empty((1, n, h, e), dtype=q.dtype, device=q.device)
    else:
        o = torch.empty((b, n, h, e), dtype=q.dtype, device=q.device)

    def grid(meta):
        return (b, h)

    print("aaa", torch.mean(alpha).item(), torch.mean(beta).item(), torch.norm(q).item(), torch.norm(k).item(), torch.norm(v).item())
    print(torch.norm(q - k).item())

    _krcl_recurrence_fwd[grid](
        Q=q,
        K=k,
        V=v,
        O=o,
        LOG_DECAY=ld,
        ALPHA=alpha,
        BETA=beta,
        STATE=initial_state,
        FINAL_STATE=final_state,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_Q=use_q,
        USE_ALPHA=use_alpha,
        USE_BETA=use_beta,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
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
def _krcl_recurrence_bwd_dk_dv_p(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    O,  # B N H E
    LOG_DECAY,  # B N H
    ALPHA,  # B N H
    BETA,  # B N H
    STATE,  # B H D E
    FINAL_STATE,  # B H D E
    DQ,  # B N H D
    DK,  # B N H D
    DV,  # B N H E
    DO,  # B N H E
    DALPHA,  # B N H
    DBETA,  # B N H
    DSTATE,  # B H D E
    DINITIAL_STATE,  # B H D E
    QDQ,  # B N H
    KDK,  # B N H
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_Q: tl.constexpr,
    USE_ALPHA: tl.constexpr,
    USE_BETA: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_DFINAL_STATE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    # compute offset
    if not USE_CU_SEQLENS:
        offset_qk = off_b * N * H * D + N * H * D + off_h * D
        offset_vo = off_b * N * H * E + N * H * E + off_h * E
        offset_log_decay = off_b * N * H + N * H + off_h
    else:
        start = tl.load(CU_SEQLENS + off_b + 1)
        end = tl.load(CU_SEQLENS + off_b)
        N = start - end
        offset_qk = start * H * D + off_h * D
        offset_vo = start * H * E + off_h * E
        offset_log_decay = start * H + off_h
    offset_state = off_b * H * D * E + off_h * D * E

    # compute block ptr
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    if USE_Q:
        q_block_ptr = Q + offset_qk + array_d
    k_block_ptr = K + offset_qk + array_d
    v_block_ptr = V + offset_vo + array_e
    o_block_ptr = O + offset_vo + array_e
    do_block_ptr = DO + offset_vo + array_e
    dk_block_ptr = DK + offset_qk + array_d
    dv_block_ptr = DV + offset_vo + array_e
    log_decay_block_ptr = LOG_DECAY + offset_log_decay
    if USE_ALPHA:
        alpha_block_ptr = ALPHA + offset_log_decay
        dalpha_block_ptr = DALPHA + offset_log_decay
    if USE_BETA:
        beta_block_ptr = BETA + offset_log_decay
        dbeta_block_ptr = DBETA + offset_log_decay
    kdk_block_ptr = KDK + offset_log_decay
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

    # compute
    for i in range(N):
        # update
        if USE_Q:
            q_block_ptr -= H * D
        k_block_ptr -= H * D
        v_block_ptr -= H * E
        o_block_ptr -= H * E
        do_block_ptr -= H * E
        dk_block_ptr -= H * D
        dv_block_ptr -= H * E
        log_decay_block_ptr -= H
        kdk_block_ptr -= H
        if USE_ALPHA:
            alpha_block_ptr -= H
        if USE_BETA:
            beta_block_ptr -= H
            dbeta_block_ptr -= H

        # load
        do = tl.load(do_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        k = tl.load(k_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
        if USE_Q:
            q = tl.load(q_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
        else:
            q = k
        v = tl.load(v_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        if USE_ALPHA:
            alpha = tl.load(alpha_block_ptr).to(tl.float32)
            q = q * alpha
        if USE_BETA:
            beta = tl.load(beta_block_ptr).to(tl.float32)
            k = k * beta
        o = tl.load(o_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        log_decay = tl.load(log_decay_block_ptr).to(tl.float32)
        decay = tl.exp(log_decay)

        # compute p, do, dk
        p = tl.sum(dstate * k[:, None], axis=0)
        dv = do + p
        dk = tl.sum(dstate * o[None, :], axis=-1)
        kdk = tl.sum(dk * k, axis=-1)

        if USE_BETA:
            dbeta = kdk / beta
            dk = dk * beta

        # store
        tl.store(dk_block_ptr, dk.to(dk_block_ptr.dtype.element_ty), mask=mask_d)
        tl.store(dv_block_ptr, dv.to(dv_block_ptr.dtype.element_ty), mask=mask_e)
        tl.store(kdk_block_ptr, kdk.to(kdk_block_ptr.dtype.element_ty))
        if USE_BETA:
            tl.store(dbeta_block_ptr, dbeta.to(dbeta_block_ptr.dtype.element_ty))
        # compute dk
        dstate = decay * (dstate - q[:, None] * dv[None, :])

    dinitial_state_block_ptr = (
        DINITIAL_STATE + offset_state + array_d[:, None] * E + array_e[None, :]
    )
    tl.store(
        dinitial_state_block_ptr,
        dstate.to(dinitial_state_block_ptr.dtype.element_ty),
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
def _krcl_recurrence_bwd_dq(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    O,  # B N H E
    LOG_DECAY,  # B N H
    ALPHA,  # B N H
    BETA,  # B N H
    STATE,  # B H D E
    FINAL_STATE,  # B H D E
    DQ,  # B N H D
    DK,  # B N H D
    DV,  # B N H E
    DO,  # B N H E
    DALPHA,  # B N H
    DBETA,  # B N H
    DSTATE,  # B H D E
    DINITIAL_STATE,  # B H D E
    QDQ,  # B N H
    KDK,  # B N H
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_Q: tl.constexpr,
    USE_ALPHA: tl.constexpr,
    USE_BETA: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_DFINAL_STATE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    # compute offset
    if not USE_CU_SEQLENS:
        offset_qk = off_b * N * H * D + off_h * D
        offset_vo = off_b * N * H * E + off_h * E
        offset_log_decay = off_b * N * H + off_h
    else:
        start = tl.load(CU_SEQLENS + off_b)
        end = tl.load(CU_SEQLENS + off_b + 1)
        N = end - start
        offset_qk = start * H * D + off_h * D
        offset_vo = start * H * E + off_h * E
        offset_log_decay = start * H + off_h
    offset_state = off_b * H * D * E + off_h * D * E

    # compute block ptr
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    if USE_Q:
        q_block_ptr = Q + offset_qk + array_d
    k_block_ptr = K + offset_qk + array_d
    o_block_ptr = O + offset_vo + array_e
    dq_block_ptr = DQ + offset_qk + array_d
    dv_block_ptr = DV + offset_vo + array_e
    if USE_ALPHA:
        alpha_block_ptr = ALPHA + offset_log_decay
        dalpha_block_ptr = DALPHA + offset_log_decay
    if USE_BETA:
        beta_block_ptr = BETA + offset_log_decay
        dbeta_block_ptr = DBETA + offset_log_decay
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

    log_decay_block_ptr = LOG_DECAY + offset_log_decay
    qdq_block_ptr = QDQ + offset_log_decay

    # compute
    for i in range(N):
        # load
        k = tl.load(k_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
        if USE_Q:
            q = tl.load(q_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
        else:
            q = k
        if USE_ALPHA:
            alpha = tl.load(alpha_block_ptr).to(tl.float32)
            q = q * alpha
        if USE_BETA:
            beta = tl.load(beta_block_ptr).to(tl.float32)
            k = k * beta
        o = tl.load(o_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        dv = tl.load(dv_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        log_decay = tl.load(log_decay_block_ptr).to(tl.float32)
        decay = tl.exp(log_decay)

        # compute dq
        dq = -decay * tl.sum(state * dv[None, :], axis=-1)
        qdq = tl.sum(dq * q, axis=-1)
        if USE_ALPHA:
            dalpha = qdq / alpha
            dq = dq * alpha

        # save
        tl.store(dq_block_ptr, dq.to(dq_block_ptr.dtype.element_ty), mask=mask_d)
        tl.store(qdq_block_ptr, qdq.to(qdq_block_ptr.dtype.element_ty))
        if USE_ALPHA:
            tl.store(dalpha_block_ptr, dalpha.to(dalpha_block_ptr.dtype.element_ty))

        # update state
        state *= decay
        state_ = k[:, None] * o[None, :]
        state += state_

        # update
        if USE_Q:
            q_block_ptr += H * D
        k_block_ptr += H * D
        o_block_ptr += H * E
        dv_block_ptr += H * E
        dq_block_ptr += H * D
        log_decay_block_ptr += H
        qdq_block_ptr += H
        if USE_ALPHA:
            alpha_block_ptr += H
            dalpha_block_ptr += H
        if USE_BETA:
            beta_block_ptr += H


def krcl_recurrence_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    ld: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    dfinal_state: torch.Tensor,
    cu_seqlens: torch.LongTensor,
):
    b, n, h, d = k.shape
    e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_initial_state = initial_state is not None
    use_q = q is not None
    use_alpha = alpha is not None
    use_beta = beta is not None
    use_dfinal_state = dfinal_state is not None
    BLOCK_D = triton.next_power_of_2(d)
    BLOCK_E = triton.next_power_of_2(e)

    dq = torch.empty_like(k)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    qdq = torch.empty_like(ld, dtype=torch.float32)
    kdk = torch.empty_like(ld, dtype=torch.float32)
    dalpha = torch.empty_like(alpha) if alpha is not None else None
    dbeta = torch.empty_like(beta) if beta is not None else None
    dinitial_state = torch.empty(b, h, d, e, device=q.device, dtype=torch.float32)

    def grid(meta):
        return (b, h)

    _krcl_recurrence_bwd_dk_dv_p[grid](
        Q=q,
        K=k,
        V=v,
        O=o,
        LOG_DECAY=ld,
        ALPHA=alpha,
        BETA=beta,
        STATE=initial_state,
        FINAL_STATE=final_state,
        DQ=dq,
        DK=dk,
        DV=dv,
        DO=do,
        DALPHA=dalpha,
        DBETA=dbeta,
        DSTATE=dfinal_state,
        DINITIAL_STATE=dinitial_state,
        QDQ=qdq,
        KDK=kdk,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_Q=use_q,
        USE_ALPHA=use_alpha,
        USE_BETA=use_beta,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        USE_DFINAL_STATE=use_dfinal_state,
        BLOCK_D=BLOCK_D,
        BLOCK_E=BLOCK_E,
    )

    _krcl_recurrence_bwd_dq[grid](
        Q=q,
        K=k,
        V=v,
        O=o,
        LOG_DECAY=ld,
        ALPHA=alpha,
        BETA=beta,
        STATE=initial_state,
        FINAL_STATE=final_state,
        DQ=dq,
        DK=dk,
        DV=dv,
        DO=do,
        DALPHA=dalpha,
        DBETA=dbeta,
        DSTATE=dfinal_state,
        DINITIAL_STATE=dinitial_state,
        QDQ=qdq,
        KDK=kdk,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_Q=use_q,
        USE_ALPHA=use_alpha,
        USE_BETA=use_beta,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        USE_DFINAL_STATE=use_dfinal_state,
        BLOCK_D=BLOCK_D,
        BLOCK_E=BLOCK_E,
    )

    if dfinal_state is not None:
        dld_state = (final_state * dfinal_state).sum(dim=-1).sum(dim=-1).unsqueeze(1)

    dld = qdq - kdk
    if cu_seqlens is not None:
        dld = dld.squeeze(0)
        b = cu_seqlens.shape[0] - 1
        array = []
        for i in range(b):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            dld_ = cumsum_fn(dld[start:end], dim=0, reverse=True)
            if dfinal_state is not None:
                dld_ = dld_ + dld_state[i]
            array.append(dld_)
        dld = torch.cat(array, dim=0)
        dld = dld.unsqueeze(0)
    else:
        dld = cumsum_fn(dld, dim=1, reverse=True)

        if dfinal_state is not None:
            dld = dld + dld_state

    dinitial_state = dinitial_state if use_initial_state else None

    return dq, dk, dv, dld, dalpha, dbeta, dinitial_state


class KrclRecurrenceFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        ld,
        alpha,
        beta,
        initial_state=None,
        cu_seqlens=None,
    ):
        # Forward computation
        o, final_state = krcl_recurrence_fwd(
            q=q,
            k=k,
            v=v,
            ld=ld,
            alpha=alpha,
            beta=beta,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

        # Save tensors needed for backward
        ctx.save_for_backward(q, k, v, o, ld, alpha, beta, initial_state, final_state, cu_seqlens)

        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dfinal_state):
        q, k, v, o, ld, alpha, beta, initial_state, final_state, cu_seqlens = ctx.saved_tensors

        dq, dk, dv, dld, dalpha, dbeta, dinitial_state = krcl_recurrence_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            do=do,
            ld=ld,
            alpha=alpha,
            beta=beta,
            initial_state=initial_state,
            final_state=final_state,
            dfinal_state=dfinal_state,
            cu_seqlens=cu_seqlens,
        )

        return (dq, dk, dv, dld, dalpha, dbeta, dinitial_state, None)


def krcl_recurrence_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Kernel Regression with Causal Linear Recurrence in Triton.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (B, N, H)
        alpha: Alpha tensor of shape (B, N, H)
        beta: Beta tensor of shape (B, N, H)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        o: Tensor of shape (B, N, H, E)
        state: Tensor of shape (B, H, D, E)
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

    return KrclRecurrenceFunction.apply(
        q,
        k,
        v,
        ld,
        alpha,
        beta,
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
    alpha = torch.randn(b, n, h, device=device, dtype=dtype).requires_grad_(True)
    beta = torch.randn(b, n, h, device=device, dtype=dtype).requires_grad_(True)
    ld = F.logsigmoid(torch.randn(b, n, h, device=device))
    initial_state = torch.randn(b, h, d, e, device=device, dtype=dtype).requires_grad_(
        True
    )
    o, final_state = krcl_recurrence_triton(q, k, v, ld, alpha, beta, initial_state)
    loss = o.sum() + final_state.sum()
    loss.backward()
