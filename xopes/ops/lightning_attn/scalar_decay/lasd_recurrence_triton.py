from typing import Optional

import torch
import triton
import triton.language as tl
from einops import repeat

from xopes.utils import contiguous, generate_configs


@triton.jit
def _activation_fwd(X, ACT: tl.constexpr):
    if ACT != "none":
        if ACT == "relu":
            X = tl.where(X >= 0, X, 0)
        elif ACT == "sigmoid":
            X = tl.sigmoid(X)
        elif ACT == "silu":
            X = X * tl.sigmoid(X)
        elif ACT == "softmax":
            X_max = tl.max(X, axis=-1)
            X_minus_max = X - X_max
            # softmax
            numerator = tl.exp(X_minus_max)
            denominator = tl.sum(numerator, axis=-1, keep_dims=True)
            X = numerator / denominator
    return X


@triton.jit
def _activation_bwd(X, DX, ACT: tl.constexpr):
    if ACT == "relu":
        DX = tl.where(X >= 0, DX, 0)
    elif ACT == "sigmoid":
        sigmoid = tl.sigmoid(X)
        DX = DX * sigmoid * (1 - sigmoid)
    elif ACT == "silu":
        sigmoid = tl.sigmoid(X)
        DX = DX * sigmoid * (1 + X * (1 - sigmoid))
    elif ACT == "softmax":
        X_max = tl.max(X, axis=-1)
        # for stable
        X_minus_max = X - X_max
        # softmax
        numerator = tl.exp(X_minus_max)
        denominator = tl.sum(numerator, axis=-1, keep_dims=True)
        O = numerator / denominator
        # scalar
        c = tl.sum(O * DX, axis=-1, keep_dims=True)
        DX = O * DX - c * O

    return DX


@triton.jit
def _normalization_fwd(X, USE_NORM: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
    if USE_NORM:
        sigma = tl.sqrt(tl.sum(X * X, axis=-1) / D + EPS)
        O = (1 / D**0.5) * X / sigma
    else:
        O = X

    return O


@triton.jit
def _normalization_bwd(
    X, DX, USE_NORM: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr
):
    if USE_NORM:
        sigma = tl.sqrt(tl.sum(X * X, axis=-1) / D + EPS)
        R = X / sigma
        DR = DX * (1 / D**0.5)
        DX = 1 / sigma * (DR - R * tl.sum(R * DR, axis=-1) / D)

    return DX


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
        }
    ),
    key=["B", "H", "D", "E", "USE_INITIAL_STATE", "USE_CU_SEQLENS", "USE_LOG_DECAY"],
)
@triton.jit
def _lasd_recurrence_fwd(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    STATE,  # B H D E
    CU_SEQLENS,  # M
    O,  # B N H E
    FINAL_STATE,  # B H D E
    LOG_DECAY,  # H
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    Q_ACT: tl.constexpr,
    K_ACT: tl.constexpr,
    V_ACT: tl.constexpr,
    Q_NORM: tl.constexpr,
    K_NORM: tl.constexpr,
    V_NORM: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_LOG_DECAY: tl.constexpr,
    EPS: tl.constexpr,
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
    array_d = tl.arange(0, D)
    array_e = tl.arange(0, E)
    q_block_ptr = Q + offset_qk + array_d
    k_block_ptr = K + offset_qk + array_d
    v_block_ptr = V + offset_vo + array_e
    o_block_ptr = O + offset_vo + array_e

    if USE_INITIAL_STATE:
        state_block_ptr = STATE + offset_state + array_d[:, None] * E + array_e[None, :]
        state = tl.load(state_block_ptr).to(tl.float32)  # D E
    else:
        state = tl.zeros((D, E), dtype=tl.float32)

    final_state_block_ptr = (
        FINAL_STATE + offset_state + array_d[:, None] * E + array_e[None, :]
    )

    if USE_LOG_DECAY:
        log_decay_block_ptr = LOG_DECAY + off_h
        log_decay = tl.load(log_decay_block_ptr).to(tl.float32)
        decay = tl.exp(log_decay)

    # compute
    for i in range(N):
        # load
        q = tl.load(q_block_ptr).to(tl.float32)
        k = tl.load(k_block_ptr).to(tl.float32)
        v = tl.load(v_block_ptr).to(tl.float32)
        q = _normalization_fwd(q, Q_NORM, D, EPS)
        k = _normalization_fwd(k, K_NORM, D, EPS)
        v = _normalization_fwd(v, V_NORM, E, EPS)
        q = _activation_fwd(q, Q_ACT)
        k = _activation_fwd(k, K_ACT)
        v = _activation_fwd(v, V_ACT)
        if USE_LOG_DECAY:
            state = decay * state + k[:, None] * v[None, :]
        else:
            state += k[:, None] * v[None, :]
        o = tl.sum(q[:, None] * state, axis=0)

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty))

        # update
        q_block_ptr += H * D
        k_block_ptr += H * D
        v_block_ptr += H * E
        o_block_ptr += H * E

    tl.store(final_state_block_ptr, state.to(final_state_block_ptr.dtype.element_ty))


def lasd_recurrence_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    q_act: str = "none",
    k_act: str = "none",
    v_act: str = "none",
    q_norm: bool = False,
    k_norm: bool = False,
    v_norm: bool = False,
    eps: float = 1e-6,
):
    b, n, h, d = q.shape
    e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_initial_state = initial_state is not None
    final_state = torch.empty((b, h, d, e), dtype=torch.float32, device=q.device)
    use_ld = ld is not None

    def grid(meta):
        return (b, h)

    if use_cu_seqlens:
        # o = torch.empty((1, cu_seqlens[-1], h, e), dtype=q.dtype, device=q.device)
        o = torch.empty((1, n, h, e), dtype=q.dtype, device=q.device)
    else:
        o = torch.empty((b, n, h, e), dtype=q.dtype, device=q.device)

    _lasd_recurrence_fwd[grid](
        Q=q,
        K=k,
        V=v,
        STATE=initial_state,
        CU_SEQLENS=cu_seqlens,
        O=o,
        FINAL_STATE=final_state,
        LOG_DECAY=ld,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        Q_ACT=q_act,
        K_ACT=k_act,
        V_ACT=v_act,
        Q_NORM=q_norm,
        K_NORM=k_norm,
        V_NORM=v_norm,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        USE_LOG_DECAY=use_ld,
        EPS=eps,
    )

    return o, final_state


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
        }
    ),
    key=["B", "H", "D", "E", "USE_INITIAL_STATE"],
)
@triton.jit
def _lasd_recurrence_bwd_dq(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    STATE,  # B H D E
    CU_SEQLENS,  # M
    FINAL_STATE,  # B H D E
    LOG_DECAY,  # H
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
    Q_ACT: tl.constexpr,
    K_ACT: tl.constexpr,
    V_ACT: tl.constexpr,
    Q_NORM: tl.constexpr,
    K_NORM: tl.constexpr,
    V_NORM: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_DFINAL_STATE: tl.constexpr,
    USE_LOG_DECAY: tl.constexpr,
    EPS: tl.constexpr,
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
    array_d = tl.arange(0, D)
    array_e = tl.arange(0, E)
    q_block_ptr = Q + offset_qk + array_d
    k_block_ptr = K + offset_qk + array_d
    v_block_ptr = V + offset_vo + array_e
    do_block_ptr = DO + offset_vo + array_e
    dq_block_ptr = DQ + offset_qk + array_d

    if USE_INITIAL_STATE:
        state_block_ptr = STATE + offset_state + array_d[:, None] * E + array_e[None, :]
        state = tl.load(state_block_ptr).to(tl.float32)  # D E
    else:
        state = tl.zeros((D, E), dtype=tl.float32)

    if USE_LOG_DECAY:
        log_decay_block_ptr = LOG_DECAY + off_h
        log_decay = tl.load(log_decay_block_ptr).to(tl.float32)
        decay = tl.exp(log_decay)

    # compute
    for i in range(N):
        # load
        do = tl.load(do_block_ptr).to(tl.float32)
        k = tl.load(k_block_ptr).to(tl.float32)
        v = tl.load(v_block_ptr).to(tl.float32)
        k = _normalization_fwd(k, K_NORM, D, EPS)
        v = _normalization_fwd(v, V_NORM, E, EPS)
        k = _activation_fwd(k, K_ACT)
        v = _activation_fwd(v, V_ACT)
        if USE_LOG_DECAY:
            state = decay * state + k[:, None] * v[None, :]
        else:
            state += k[:, None] * v[None, :]
        dq = tl.sum(do[None, :] * state, axis=-1)

        if Q_ACT != "none" or Q_NORM:
            q = tl.load(q_block_ptr).to(tl.float32)
            dq = _activation_bwd(q, dq, Q_ACT)
            dq = _normalization_bwd(q, dq, Q_NORM, D, EPS)
            q_block_ptr += H * D

        tl.store(dq_block_ptr, dq.to(dq_block_ptr.dtype.element_ty))

        # update
        k_block_ptr += H * D
        v_block_ptr += H * E
        do_block_ptr += H * E
        dq_block_ptr += H * D


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
        }
    ),
    key=["B", "H", "D", "E", "USE_INITIAL_STATE"],
)
@triton.jit
def _lasd_recurrence_bwd_dk_dv(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    STATE,  # B H D E
    CU_SEQLENS,  # M
    FINAL_STATE,  # B H D E
    LOG_DECAY,  # H
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
    Q_ACT: tl.constexpr,
    K_ACT: tl.constexpr,
    V_ACT: tl.constexpr,
    Q_NORM: tl.constexpr,
    K_NORM: tl.constexpr,
    V_NORM: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_DFINAL_STATE: tl.constexpr,
    USE_LOG_DECAY: tl.constexpr,
    EPS: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    # compute offset
    if not USE_CU_SEQLENS:
        offset_qk = off_b * N * H * D + N * H * D + off_h * D
        offset_vo = off_b * N * H * E + N * H * E + off_h * E
    else:
        start = tl.load(CU_SEQLENS + off_b + 1)
        end = tl.load(CU_SEQLENS + off_b)
        N = start - end
        offset_qk = start * H * D + off_h * D
        offset_vo = start * H * E + off_h * E
    offset_state = off_b * H * D * E + off_h * D * E

    # compute block ptr
    array_d = tl.arange(0, D)
    array_e = tl.arange(0, E)
    q_block_ptr = Q + offset_qk + array_d
    k_block_ptr = K + offset_qk + array_d
    v_block_ptr = V + offset_vo + array_e
    do_block_ptr = DO + offset_vo + array_e
    dk_block_ptr = DK + offset_qk + array_d
    dv_block_ptr = DV + offset_vo + array_e

    if USE_DFINAL_STATE:
        dstate_block_ptr = (
            DSTATE + offset_state + array_d[:, None] * E + array_e[None, :]
        )
        dstate = tl.load(dstate_block_ptr).to(tl.float32)  # D E
    else:
        dstate = tl.zeros((D, E), dtype=tl.float32)

    if USE_LOG_DECAY:
        log_decay_block_ptr = LOG_DECAY + off_h
        log_decay = tl.load(log_decay_block_ptr).to(tl.float32)
        decay = tl.exp(log_decay)

    # compute
    for i in range(N):
        # update
        q_block_ptr -= H * D
        k_block_ptr -= H * D
        v_block_ptr -= H * E
        do_block_ptr -= H * E
        dk_block_ptr -= H * D
        dv_block_ptr -= H * E

        # load
        do = tl.load(do_block_ptr).to(tl.float32)
        q = tl.load(q_block_ptr).to(tl.float32)
        q = _normalization_fwd(q, Q_NORM, D, EPS)
        q = _activation_fwd(q, Q_ACT)
        # !!! IMPORTANT
        if i > 0:
            if USE_LOG_DECAY:
                dstate = decay * dstate
        dstate += q[:, None] * do[None, :]
        # compute k and v
        k = tl.load(k_block_ptr).to(tl.float32)
        v = tl.load(v_block_ptr).to(tl.float32)
        k_ = _normalization_fwd(k, K_NORM, D, EPS)
        v_ = _normalization_fwd(v, V_NORM, E, EPS)
        k_ = _activation_fwd(k_, K_ACT)
        v_ = _activation_fwd(v_, V_ACT)
        dk = tl.sum(dstate * v_[None, :], axis=-1)
        dv = tl.sum(dstate * k_[:, None], axis=0)
        # norm and act
        dk = _activation_bwd(k, dk, K_ACT)
        dv = _activation_bwd(v, dv, V_ACT)
        dk = _normalization_bwd(k, dk, K_NORM, D, EPS)
        dv = _normalization_bwd(v, dv, V_NORM, E, EPS)

        tl.store(dk_block_ptr, dk.to(dk_block_ptr.dtype.element_ty))
        tl.store(dv_block_ptr, dv.to(dv_block_ptr.dtype.element_ty))

    # !!! IMPORTANT
    if USE_LOG_DECAY:
        dstate = decay * dstate

    if USE_INITIAL_STATE:
        dinitial_state_block_ptr = (
            DINITIAL_STATE + offset_state + array_d[:, None] * E + array_e[None, :]
        )
        tl.store(
            dinitial_state_block_ptr,
            dstate.to(dinitial_state_block_ptr.dtype.element_ty),
        )


def lasd_recurrence_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    do: torch.Tensor,
    dfinal_state: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    q_act: str = "none",
    k_act: str = "none",
    v_act: str = "none",
    q_norm: bool = False,
    k_norm: bool = False,
    v_norm: bool = False,
    eps: float = 1e-6,
):
    b, n, h, d = q.shape
    e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_initial_state = initial_state is not None
    use_dfinal_state = dfinal_state is not None
    use_ld = ld is not None

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    if use_initial_state:
        dinitial_state = torch.empty(b, h, d, e, device=q.device, dtype=torch.float32)
    else:
        dinitial_state = None

    def grid(meta):
        return (b, h)

    _lasd_recurrence_bwd_dq[grid](
        Q=q,
        K=k,
        V=v,
        STATE=initial_state,
        CU_SEQLENS=cu_seqlens,
        FINAL_STATE=final_state,
        LOG_DECAY=ld,
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
        Q_ACT=q_act,
        K_ACT=k_act,
        V_ACT=v_act,
        Q_NORM=q_norm,
        K_NORM=k_norm,
        V_NORM=v_norm,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        USE_DFINAL_STATE=use_dfinal_state,
        USE_LOG_DECAY=use_ld,
        EPS=eps,
    )

    _lasd_recurrence_bwd_dk_dv[grid](
        Q=q,
        K=k,
        V=v,
        STATE=initial_state,
        CU_SEQLENS=cu_seqlens,
        FINAL_STATE=final_state,
        LOG_DECAY=ld,
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
        Q_ACT=q_act,
        K_ACT=k_act,
        V_ACT=v_act,
        Q_NORM=q_norm,
        K_NORM=k_norm,
        V_NORM=v_norm,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        USE_DFINAL_STATE=use_dfinal_state,
        USE_LOG_DECAY=use_ld,
        EPS=eps,
    )

    return dq, dk, dv, dinitial_state


class LasdRecurrenceFunction(torch.autograd.Function):
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
        q_act="none",
        k_act="none",
        v_act="none",
        q_norm=False,
        k_norm=False,
        v_norm=False,
        eps=1e-6,
    ):
        # Save non-tensor inputs for backward
        ctx.q_act = q_act
        ctx.k_act = k_act
        ctx.v_act = v_act
        ctx.q_norm = q_norm
        ctx.k_norm = k_norm
        ctx.v_norm = v_norm
        ctx.eps = eps

        # Forward computation
        output, final_state = lasd_recurrence_fwd(
            q=q,
            k=k,
            v=v,
            ld=ld,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            q_act=q_act,
            k_act=k_act,
            v_act=v_act,
            q_norm=q_norm,
            k_norm=k_norm,
            v_norm=v_norm,
            eps=eps,
        )

        # Save tensors needed for backward
        ctx.save_for_backward(q, k, v, ld, initial_state, final_state, cu_seqlens)

        return output, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dfinal_state):
        q, k, v, ld, initial_state, final_state, cu_seqlens = ctx.saved_tensors
        q_act = ctx.q_act
        k_act = ctx.k_act
        v_act = ctx.v_act
        q_norm = ctx.q_norm
        k_norm = ctx.k_norm
        v_norm = ctx.v_norm
        eps = ctx.eps

        dq, dk, dv, dinitial_state = lasd_recurrence_bwd(
            q=q,
            k=k,
            v=v,
            ld=ld,
            initial_state=initial_state,
            final_state=final_state,
            do=do,
            dfinal_state=dfinal_state,
            cu_seqlens=cu_seqlens,
            q_act=q_act,
            k_act=k_act,
            v_act=v_act,
            q_norm=q_norm,
            k_norm=k_norm,
            v_norm=v_norm,
            eps=eps,
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


def lasd_recurrence_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    q_act: str = "none",
    k_act: str = "none",
    v_act: str = "none",
    q_norm: bool = False,
    k_norm: bool = False,
    v_norm: bool = False,
    eps: float = 1e-6,
):
    """
    Apply Lightning Attention with Scalar Decay in Triton.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (H,)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training
        q_act: Activation function for query
        k_act: Activation function for key
        v_act: Activation function for value
        q_norm: Normalize query
        k_norm: Normalize key
        v_norm: Normalize value
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

    return LasdRecurrenceFunction.apply(
        q,
        k,
        v,
        ld,
        initial_state,
        cu_seqlens,
        q_act,
        k_act,
        v_act,
        q_norm,
        k_norm,
        v_norm,
        eps,
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
    output, final_state = lasd_recurrence_triton(q, k, v, ld, initial_state)
    loss = output.sum() + final_state.sum()
    loss.backward()
