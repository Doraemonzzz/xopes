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
            "BLOCK_D": [
                128,
                256,
            ],
        }
    ),
    key=[
        "B",
        "D",
        "USE_INITIAL_STATE",
        "USE_CU_SEQLENS",
    ],
)
@triton.jit
def _laer_recurrence_fwd(
    Q,  # B N D
    K,  # B N D
    V,  # B N D
    STATE,  # B D
    CU_SEQLENS,  # M
    O,  # B N D
    FINAL_STATE,  # B D
    LOG_DECAY,  # B N D
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    MAX_BLOCK_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)

    # compute offset
    if not USE_CU_SEQLENS:
        offset = off_b * N * D
    else:
        start = tl.load(CU_SEQLENS + off_b)
        end = tl.load(CU_SEQLENS + off_b + 1)
        N = end - start
        offset = start * D
    offset_state = off_b * D

    # compute block ptr
    array_d = off_h * BLOCK_D + tl.arange(0, BLOCK_D)
    q_block_ptr = Q + offset + array_d
    k_block_ptr = K + offset + array_d
    v_block_ptr = V + offset + array_d
    o_block_ptr = O + offset + array_d
    log_decay_block_ptr = LOG_DECAY + offset + array_d
    mask_d = array_d < D

    if USE_INITIAL_STATE:
        state_block_ptr = STATE + offset_state + array_d
        state = tl.load(state_block_ptr, mask=mask_d, other=0.0).to(
            tl.float32
        )  # BLOCK_D
    else:
        state = tl.zeros((BLOCK_D,), dtype=tl.float32)

    final_state_block_ptr = FINAL_STATE + offset_state + array_d

    # compute
    for i in range(N):
        # load
        q = tl.load(q_block_ptr, mask=mask_d, other=0.0)
        k = tl.load(k_block_ptr, mask=mask_d, other=0.0)
        v = tl.load(v_block_ptr, mask=mask_d, other=0.0)
        log_decay = tl.load(log_decay_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
        decay = tl.exp(log_decay)

        state = decay * state + k * v
        o = q * state

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask_d)

        # update
        q_block_ptr += D
        k_block_ptr += D
        v_block_ptr += D
        o_block_ptr += D
        log_decay_block_ptr += D

    tl.store(
        final_state_block_ptr,
        state.to(final_state_block_ptr.dtype.element_ty),
        mask=mask_d,
    )


def laer_recurrence_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    b, n, d = q.shape
    dtype = q.dtype
    device = q.device
    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_initial_state = initial_state is not None
    final_state = torch.empty((b, d), dtype=dtype, device=device)
    MAX_BLOCK_D = triton.next_power_of_2(d)

    if use_cu_seqlens:
        o = torch.empty((1, n, d), dtype=dtype, device=device)
    else:
        o = torch.empty((b, n, d), dtype=dtype, device=device)

    def grid(meta):
        return (
            b,
            triton.cdiv(d, meta["BLOCK_D"]),
        )

    _laer_recurrence_fwd[grid](
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
        D=d,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        MAX_BLOCK_D=MAX_BLOCK_D,
    )

    return o, final_state


@triton.autotune(
    generate_configs({"num_warps": [4, 8, 16, 32], "BLOCK_D": [128, 256]}),
    key=["B", "D", "USE_INITIAL_STATE", "USE_CU_SEQLENS"],
)
@triton.jit
def _laer_recurrence_bwd_dq(
    Q,  # B N D
    K,  # B N D
    V,  # B N D
    STATE,  # B D
    CU_SEQLENS,  # M
    FINAL_STATE,  # B D
    LOG_DECAY,  # B N D
    DO,  # B N D
    DSTATE,  # B D
    DQ,  # B N D
    DK,  # B N D
    DV,  # B N D
    DINITIAL_STATE,  # B D
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_DFINAL_STATE: tl.constexpr,
    MAX_BLOCK_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)

    # compute offset
    if not USE_CU_SEQLENS:
        offset = off_b * N * D
    else:
        start = tl.load(CU_SEQLENS + off_b)
        end = tl.load(CU_SEQLENS + off_b + 1)
        N = end - start
        offset = start * D
    offset_state = off_b * D

    # compute block ptr
    array_d = off_h * BLOCK_D + tl.arange(0, BLOCK_D)
    k_block_ptr = K + offset + array_d
    v_block_ptr = V + offset + array_d
    do_block_ptr = DO + offset + array_d
    dq_block_ptr = DQ + offset + array_d
    log_decay_block_ptr = LOG_DECAY + offset + array_d
    mask_d = array_d < D

    if USE_INITIAL_STATE:
        state_block_ptr = STATE + offset_state + array_d
        state = tl.load(state_block_ptr, mask=mask_d, other=0.0).to(
            tl.float32
        )  # BLOCK_D
    else:
        state = tl.zeros((BLOCK_D,), dtype=tl.float32)

    final_state_block_ptr = FINAL_STATE + offset_state + array_d

    # compute
    for i in range(N):
        # load
        do = tl.load(do_block_ptr, mask=mask_d, other=0.0)
        k = tl.load(k_block_ptr, mask=mask_d, other=0.0)
        v = tl.load(v_block_ptr, mask=mask_d, other=0.0)
        log_decay = tl.load(log_decay_block_ptr, mask=mask_d, other=0.0).to(tl.float32)
        decay = tl.exp(log_decay)

        state = decay * state + k * v
        dq = do * state

        tl.store(dq_block_ptr, dq.to(dq_block_ptr.dtype.element_ty), mask=mask_d)

        # update
        k_block_ptr += D
        v_block_ptr += D
        do_block_ptr += D
        dq_block_ptr += D
        log_decay_block_ptr += D

    tl.store(
        final_state_block_ptr,
        state.to(final_state_block_ptr.dtype.element_ty),
        mask=mask_d,
    )


@triton.autotune(
    generate_configs({"num_warps": [4, 8, 16, 32], "BLOCK_D": [128, 256]}),
    key=["B", "D", "USE_INITIAL_STATE", "USE_CU_SEQLENS"],
)
@triton.jit
def _laer_recurrence_bwd_dk_dv(
    Q,  # B N D
    K,  # B N D
    V,  # B N D
    STATE,  # B D
    CU_SEQLENS,  # M
    FINAL_STATE,  # B D
    LOG_DECAY,  # B N D
    DO,  # B N D
    DSTATE,  # B D
    DQ,  # B N D
    DK,  # B N D
    DV,  # B N D
    DINITIAL_STATE,  # B D
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_DFINAL_STATE: tl.constexpr,
    MAX_BLOCK_D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)

    # compute offset
    if not USE_CU_SEQLENS:
        offset = off_b * N * D + N * D
    else:
        start = tl.load(CU_SEQLENS + off_b + 1)
        end = tl.load(CU_SEQLENS + off_b)
        N = start - end
        offset = start * D + N * D
    offset_state = off_b * D

    # compute block ptr
    array_d = off_h * BLOCK_D + tl.arange(0, BLOCK_D)
    q_block_ptr = Q + offset + array_d
    k_block_ptr = K + offset + array_d
    v_block_ptr = V + offset + array_d
    do_block_ptr = DO + offset + array_d
    dk_block_ptr = DK + offset + array_d
    dv_block_ptr = DV + offset + array_d
    log_decay_block_ptr = LOG_DECAY + offset + array_d
    mask_d = array_d < D

    if USE_DFINAL_STATE:
        dstate_block_ptr = DSTATE + offset_state + array_d
        dstate = tl.load(dstate_block_ptr, mask=mask_d, other=0.0).to(
            tl.float32
        )  # BLOCK_D
    else:
        dstate = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # compute
    for i in range(N):
        # update
        q_block_ptr -= D
        k_block_ptr -= D
        v_block_ptr -= D
        do_block_ptr -= D
        dk_block_ptr -= D
        dv_block_ptr -= D
        # load
        do = tl.load(do_block_ptr, mask=mask_d, other=0.0)
        q = tl.load(q_block_ptr, mask=mask_d, other=0.0)
        # !!! IMPORTANT
        if i > 0:
            log_decay_block_ptr -= D
            log_decay = tl.load(log_decay_block_ptr).to(tl.float32)
            decay = tl.exp(log_decay)
            dstate = decay * dstate

        dstate += q * do
        # compute k and v
        k = tl.load(k_block_ptr, mask=mask_d, other=0.0)
        v = tl.load(v_block_ptr, mask=mask_d, other=0.0)
        dk = dstate * v
        dv = dstate * k

        tl.store(dk_block_ptr, dk.to(dk_block_ptr.dtype.element_ty), mask=mask_d)
        tl.store(dv_block_ptr, dv.to(dv_block_ptr.dtype.element_ty), mask=mask_d)

    # !!! IMPORTANT
    log_decay_block_ptr -= D
    log_decay = tl.load(log_decay_block_ptr).to(tl.float32)
    decay = tl.exp(log_decay)
    dstate = decay * dstate

    dinitial_state_block_ptr = DINITIAL_STATE + offset_state + array_d
    tl.store(
        dinitial_state_block_ptr,
        dstate.to(dinitial_state_block_ptr.dtype.element_ty),
        mask=mask_d,
    )


def laer_recurrence_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    do: torch.Tensor,
    dfinal_state: torch.Tensor,
    cu_seqlens: torch.LongTensor,
):
    b, n, d = q.shape
    dtype = q.dtype
    device = q.device

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1

    use_initial_state = initial_state is not None
    use_dfinal_state = dfinal_state is not None
    MAX_BLOCK_D = triton.next_power_of_2(d)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dinitial_state = torch.empty(b, d, device=device, dtype=dtype)

    def grid(meta):
        return (
            b,
            triton.cdiv(d, meta["BLOCK_D"]),
        )

    _laer_recurrence_bwd_dq[grid](
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
        D=d,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        USE_DFINAL_STATE=use_dfinal_state,
        MAX_BLOCK_D=MAX_BLOCK_D,
    )

    _laer_recurrence_bwd_dk_dv[grid](
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
        D=d,
        USE_CU_SEQLENS=use_cu_seqlens,
        USE_INITIAL_STATE=use_initial_state,
        USE_DFINAL_STATE=use_dfinal_state,
        MAX_BLOCK_D=MAX_BLOCK_D,
    )

    if dfinal_state is not None:
        dld_state = (final_state * dfinal_state).unsqueeze(1)

    dld = q * dq - k * dk
    dld = cumsum_fn(dld, dim=1, reverse=True)

    if dfinal_state is not None:
        dld = dld + dld_state

    dinitial_state = dinitial_state if use_initial_state else None

    return dq, dk, dv, dld, dinitial_state


class LaerRecurrenceFunction(torch.autograd.Function):
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
        output, final_state = laer_recurrence_fwd(
            q=q,
            k=k,
            v=v,
            ld=ld,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

        # Save tensors needed for backward
        ctx.save_for_backward(q, k, v, ld, initial_state, final_state, cu_seqlens)

        return output, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dfinal_state):
        q, k, v, ld, initial_state, final_state, cu_seqlens = ctx.saved_tensors

        dq, dk, dv, dld, dinitial_state = laer_recurrence_bwd(
            q=q,
            k=k,
            v=v,
            ld=ld,
            initial_state=initial_state,
            final_state=final_state,
            do=do,
            dfinal_state=dfinal_state,
            cu_seqlens=cu_seqlens,
        )

        return (
            dq,
            dk,
            dv,
            dld,
            dinitial_state,
            None,
        )


def laer_recurrence_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Lightning Attention with Element-wise Recurrence.

    Args:
        q: Query tensor of shape (B, N, D)
        k: Key tensor of shape (B, N, D)
        v: Value tensor of shape (B, N, D)
        ld: Logarithmic decay tensor of shape (B, N, D)
        initial_state: Initial state tensor of shape (B, D)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        output: Tensor of shape (B, N, D)
        state: Tensor of shape (B, D)
    """
    b = q.shape[0]
    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = cu_seqlens.shape[0] - 1
    if initial_state is not None:
        initial_state = initial_state.squeeze(0)
        # treat for varlen training
        if len(initial_state.shape) == 1:
            initial_state = repeat(initial_state, "d -> b d", b=b).contiguous()

    return LaerRecurrenceFunction.apply(
        q,
        k,
        v,
        ld,
        initial_state,
        cu_seqlens,
    )


if __name__ == "__main__":
    import torch.nn.functional as F

    b, n, d = 2, 16, 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    q = torch.randn(b, n, d, device=device, dtype=dtype).requires_grad_(True)
    k = torch.randn(b, n, d, device=device, dtype=dtype).requires_grad_(True)
    v = torch.randn(b, n, d, device=device, dtype=dtype).requires_grad_(True)
    ld = F.logsigmoid(torch.randn(b, n, d, device=device))
    initial_state = torch.randn(b, d, device=device, dtype=dtype).requires_grad_(True)
    output, final_state = laer_recurrence_triton(q, k, v, ld, initial_state)
    loss = output.sum() + final_state.sum()
    loss.backward()
