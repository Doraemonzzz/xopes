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
def _ilav_recurrence_fwd(
    Q,  # B N H D
    K,  # B N H D
    O,  # B N H E
    STATE,  # B H D E
    CU_SEQLENS,  # M
    V,  # B N H E
    FINAL_STATE,  # B H D E
    LOG_DECAY,  # B N H
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
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
    q_block_ptr = Q + offset_qk + array_d
    k_block_ptr = K + offset_qk + array_d
    v_block_ptr = V + offset_vo + array_e
    o_block_ptr = O + offset_vo + array_e
    log_decay_block_ptr = LOG_DECAY + offset_log_decay
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
        k = tl.load(k_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
        o = tl.load(o_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        log_decay = tl.load(log_decay_block_ptr).to(tl.float32)
        decay = tl.exp(log_decay)

        # update state
        state *= decay
        v = o - tl.sum(q[:, None] * state, axis=0)

        state_ = (1 - decay) * k[:, None] * v[None, :]
        state += state_

        tl.store(v_block_ptr, v.to(v_block_ptr.dtype.element_ty), mask=mask_e)

        # update
        q_block_ptr += H * D
        k_block_ptr += H * D
        v_block_ptr += H * E
        o_block_ptr += H * E
        log_decay_block_ptr += H

    tl.store(
        final_state_block_ptr,
        state.to(final_state_block_ptr.dtype.element_ty),
        mask=mask_d[:, None] & mask_e[None, :],
    )


def ilav_recurrence_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    o: torch.Tensor,
    ld: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    b, n, h, d = q.shape
    e = o.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_initial_state = initial_state is not None
    final_state = torch.empty((b, h, d, e), dtype=torch.float32, device=q.device)
    BLOCK_D = triton.next_power_of_2(d)
    BLOCK_E = triton.next_power_of_2(e)

    if use_cu_seqlens:
        v = torch.empty((1, n, h, e), dtype=q.dtype, device=q.device)
    else:
        v = torch.empty((b, n, h, e), dtype=q.dtype, device=q.device)

    def grid(meta):
        return (b, h)

    _ilav_recurrence_fwd[grid](
        Q=q,
        K=k,
        O=o,
        STATE=initial_state,
        CU_SEQLENS=cu_seqlens,
        V=v,
        FINAL_STATE=final_state,
        LOG_DECAY=ld,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        BLOCK_D=BLOCK_D,
        BLOCK_E=BLOCK_E,
    )

    return v, final_state


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
        }
    ),
    key=["B", "H", "D", "E", "USE_INITIAL_STATE", "USE_CU_SEQLENS"],
)
@triton.jit
def _ilav_recurrence_bwd_dk(
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
    q_block_ptr = Q + offset_qk + array_d
    k_block_ptr = K + offset_qk + array_d
    v_block_ptr = V + offset_vo + array_e
    dk_block_ptr = DK + offset_qk + array_d
    dv_block_ptr = DV + offset_vo + array_e
    log_decay_block_ptr = LOG_DECAY + offset_log_decay
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
        q_block_ptr -= H * D
        k_block_ptr -= H * D
        v_block_ptr -= H * E
        dk_block_ptr -= H * D
        dv_block_ptr -= H * E
        log_decay_block_ptr -= H

        # load
        dv = tl.load(dv_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        q = tl.load(q_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
        v = tl.load(v_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        log_decay = tl.load(log_decay_block_ptr).to(tl.float32)
        decay = tl.exp(log_decay)

        # compute dk
        dk = (1 - decay) * tl.sum(v[None, :] * dstate, axis=-1)

        # update dstate
        dstate = decay * (dstate - q[:, None] * dv[None, :])

        tl.store(dk_block_ptr, dk.to(dk_block_ptr.dtype.element_ty), mask=mask_d)

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
def _ilav_recurrence_bwd_dq(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    STATE,  # B H D E
    CU_SEQLENS,  # M
    FINAL_STATE,  # B H D E
    LOG_DECAY,  # B N H
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
    Q + offset_qk + array_d
    k_block_ptr = K + offset_qk + array_d
    v_block_ptr = V + offset_vo + array_e
    dq_block_ptr = DQ + offset_qk + array_d
    dv_block_ptr = DV + offset_vo + array_e
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

    # compute
    for i in range(N):
        # load
        dv = tl.load(dv_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        k = tl.load(k_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
        v = tl.load(v_block_ptr, mask=mask_e, other=0.0).to(tl.float32)
        log_decay = tl.load(log_decay_block_ptr).to(tl.float32)
        decay = tl.exp(log_decay)

        # compute dq
        dq = -decay * tl.sum(dv[None, :] * state, axis=-1)

        # update state
        state = decay * state + (1 - decay) * k[:, None] * v[None, :]

        tl.store(dq_block_ptr, dq.to(dq_block_ptr.dtype.element_ty), mask=mask_d)

        # update
        k_block_ptr += H * D
        v_block_ptr += H * E
        dv_block_ptr += H * E
        dq_block_ptr += H * D
        log_decay_block_ptr += H


def ilav_recurrence_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    dv: torch.Tensor,
    dfinal_state: torch.Tensor,
    cu_seqlens: torch.LongTensor,
):
    b, n, h, d = q.shape
    e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_initial_state = initial_state is not None
    use_dfinal_state = dfinal_state is not None
    BLOCK_D = triton.next_power_of_2(d)
    BLOCK_E = triton.next_power_of_2(e)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    do = dv
    dinitial_state = torch.empty(b, h, d, e, device=q.device, dtype=torch.float32)

    def grid(meta):
        return (b, h)

    _ilav_recurrence_bwd_dk[grid](
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
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        USE_DFINAL_STATE=use_dfinal_state,
        BLOCK_D=BLOCK_D,
        BLOCK_E=BLOCK_E,
    )

    _ilav_recurrence_bwd_dq[grid](
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
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        USE_DFINAL_STATE=use_dfinal_state,
        BLOCK_D=BLOCK_D,
        BLOCK_E=BLOCK_E,
    )

    if dfinal_state is not None:
        dld_state = (final_state * dfinal_state).sum(dim=-1).sum(dim=-1).unsqueeze(1)

    dld = (q * dq - k * dk).sum(dim=-1)
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

    # dld = ld

    dinitial_state = dinitial_state if use_initial_state else None

    return dq, dk, do, dld, dinitial_state


class IlavRecurrenceFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        k,
        o,
        ld,
        initial_state=None,
        cu_seqlens=None,
    ):
        # Forward computation
        v, final_state = ilav_recurrence_fwd(
            q=q,
            k=k,
            o=o,
            ld=ld,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

        # Save tensors needed for backward
        ctx.save_for_backward(q, k, v, ld, initial_state, final_state, cu_seqlens)

        return v, final_state

    @staticmethod
    @contiguous
    def backward(ctx, dv, dfinal_state):
        q, k, v, ld, initial_state, final_state, cu_seqlens = ctx.saved_tensors

        dq, dk, do, dld, dinitial_state = ilav_recurrence_bwd(
            q=q,
            k=k,
            v=v,
            ld=ld,
            initial_state=initial_state,
            final_state=final_state,
            dv=dv,
            dfinal_state=dfinal_state,
            cu_seqlens=cu_seqlens,
        )

        return (
            dq,
            dk,
            do,
            dld,
            dinitial_state,
            None,
        )


def ilav_recurrence_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    o: torch.Tensor,
    ld: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Implicit Attention Recurrence with V in Triton.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        o: Output tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (B, N, H)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        output: Tensor of shape (B, N, H, E)
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

    return IlavRecurrenceFunction.apply(
        q,
        k,
        o,
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
    ld = F.logsigmoid(torch.randn(b, n, h, device=device))
    initial_state = torch.randn(b, h, d, e, device=device, dtype=dtype).requires_grad_(
        True
    )
    output, final_state = ilav_recurrence_triton(q, k, v, ld, initial_state)
    loss = output.sum() + final_state.sum()
    loss.backward()
