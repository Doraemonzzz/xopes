import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs({"BLOCK_E": [16, 32, 64, 128], "num_warps": [2, 4, 8]}),
    key=["n", "d"],
)
@triton.jit
def _grpe_recurrence_fwd(
    Q,
    K,
    V,
    M,
    O,
    S_INITIAL_STATE,
    S_FINAL_STATE,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    OUTPUT_FINAL_STATE: tl.constexpr,  # whether to output final state
    BLOCK_E: tl.constexpr,
):
    """
    q: (1, d)
    k: (1, d)
    v: (1, BLOCK_E)
    m: (d, d)
    s: (d, BLOCK_E)
    """
    off_bh = tl.program_id(0)
    off_e = tl.program_id(1)
    # compute offset
    off_qk = off_bh * n * d
    off_ov = off_bh * n * e
    off_m = off_bh * n * d * d
    off_e = off_e * BLOCK_E
    off_s = off_bh * d * e

    # compute block ptr
    q_trans_block_ptr = Q + off_qk + tl.arange(0, d)[:, None]
    k_trans_block_ptr = K + off_qk + tl.arange(0, d)[:, None]
    v_block_ptr = V + off_ov + off_e + tl.arange(0, BLOCK_E)[None, :]
    m_block_ptr = M + off_m + tl.arange(0, d)[:, None] * d + tl.arange(0, d)[None, :]
    o_block_ptr = O + off_ov + off_e + tl.arange(0, BLOCK_E)[None, :]

    mask = (off_e + tl.arange(0, BLOCK_E)[None, :]) < e

    if USE_INITIAL_STATE:
        s_block_ptr = (
            S_INITIAL_STATE
            + off_s
            + tl.arange(0, d)[:, None] * e
            + off_e
            + tl.arange(0, BLOCK_E)[None, :]
        )

        s = tl.load(s_block_ptr, mask=mask, other=0).to(tl.float32)
    else:
        s = tl.zeros([d, BLOCK_E], dtype=tl.float32)

    for i in range(n):
        q_trans = tl.load(q_trans_block_ptr).to(tl.float32)
        k_trans = tl.load(k_trans_block_ptr).to(tl.float32)
        v = tl.load(v_block_ptr, mask=mask, other=0).to(tl.float32)
        m = tl.load(m_block_ptr).to(tl.float32)

        s = tl.dot(m, s) + k_trans.to(v.dtype) * v
        o = q_trans * s
        # d e -> 1 e
        o = tl.sum(o, axis=0)[None, :]

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask)

        q_trans_block_ptr += d
        k_trans_block_ptr += d
        v_block_ptr += e
        m_block_ptr += d * d
        o_block_ptr += e

    if OUTPUT_FINAL_STATE:
        s_final_block_ptr = (
            S_FINAL_STATE
            + off_s
            + tl.arange(0, d)[:, None] * e
            + off_e
            + tl.arange(0, BLOCK_E)[None, :]
        )

        tl.store(
            s_final_block_ptr,
            s.to(s_final_block_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def _grpe_recurrence_bwd(
    Q,
    K,
    V,
    M,
    DO,
    DQ,
    DK,
    DV,
    DM,
    DS,
    S_INITIAL_STATE,
    S_FINAL_STATE,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    OUTPUT_FINAL_STATE: tl.constexpr,  # whether to output final state
):
    """
    q: (d, 1)
    k: (d, 1)
    v: (1, e)
    do: (1, e)
    dq: (d, 1)
    dk: (d, 1)
    dv: (1, e)
    m: (d, d)
    s: (d, e)
    """
    off_bh = tl.program_id(0)
    # compute offset
    off_qk = off_bh * n * d
    off_ov = off_bh * n * e
    off_m = off_bh * n * d * d
    off_s = off_bh * d * e

    # compute block ptr
    # fwd
    q_trans_block_ptr = Q + off_qk + tl.arange(0, d)[:, None]
    k_trans_block_ptr = K + off_qk + tl.arange(0, d)[:, None]
    v_block_ptr = V + off_ov + tl.arange(0, e)[None, :]
    m_block_ptr = M + off_m + tl.arange(0, d)[:, None] * d + tl.arange(0, d)[None, :]
    # o_block_ptr = O + off_ov + tl.arange(0, e)[None, :]

    # bwd
    do_block_ptr = DO + off_ov + tl.arange(0, e)[None, :]
    dq_trans_block_ptr = DQ + off_qk + tl.arange(0, d)[:, None]
    dk_trans_block_ptr = DK + off_qk + tl.arange(0, d)[:, None]
    dv_block_ptr = DV + off_ov + tl.arange(0, e)[None, :]

    if USE_INITIAL_STATE:
        s_block_ptr = (
            S_INITIAL_STATE
            + off_s
            + tl.arange(0, d)[:, None] * e
            + tl.arange(0, e)[None, :]
        )

        s = tl.load(s_block_ptr).to(tl.float32)
    else:
        s = tl.zeros([d, e], dtype=tl.float32)

    for i in range(n):
        # q_trans = tl.load(q_trans_block_ptr).to(tl.float32)
        do = tl.load(do_block_ptr).to(tl.float32)
        k_trans = tl.load(k_trans_block_ptr).to(tl.float32)
        v = tl.load(v_block_ptr).to(tl.float32)
        m = tl.load(m_block_ptr).to(tl.float32)

        s = tl.dot(m, s) + k_trans.to(v.dtype) * v
        # o = q_trans * s
        dq_trans = s * do
        # d e -> d 1
        dq_trans = tl.sum(dq_trans, axis=1)[:, None]

        tl.store(dq_trans_block_ptr, dq_trans.to(dq_trans_block_ptr.dtype.element_ty))

        # q_trans_block_ptr += d
        k_trans_block_ptr += d
        v_block_ptr += e
        m_block_ptr += d * d
        do_block_ptr += e
        dq_trans_block_ptr += d

    ds = tl.zeros([d, e], dtype=tl.float32)
    do_block_ptr = DO + off_ov + n * e + tl.arange(0, e)[None, :]
    q_trans_block_ptr = Q + off_qk + n * d + tl.arange(0, d)[:, None]
    k_trans_block_ptr = K + off_qk + n * d + tl.arange(0, d)[:, None]
    v_block_ptr = V + off_ov + n * e + tl.arange(0, e)[None, :]
    m_trans_block_ptr = (
        M + off_m + n * d * d + tl.arange(0, d)[:, None] * d + tl.arange(0, d)[None, :]
    )

    dk_trans_block_ptr = DK + off_qk + n * d + tl.arange(0, d)[:, None]
    dv_block_ptr = DV + off_ov + n * e + tl.arange(0, e)[None, :]

    for i in range(n - 1, -1, -1):
        do_block_ptr -= e
        dq_trans_block_ptr -= d
        k_trans_block_ptr -= d
        v_block_ptr -= e
        m_block_ptr -= d * d

        dk_trans_block_ptr -= d
        dv_block_ptr -= e

        q_trans = tl.load(q_trans_block_ptr).to(tl.float32)
        do = tl.load(do_block_ptr).to(tl.float32)
        k_trans = tl.load(k_trans_block_ptr).to(tl.float32)
        v = tl.load(v_block_ptr).to(tl.float32)
        tl.load(m_trans_block_ptr).to(tl.float32)

        ds = tl.dot(m, ds) + q_trans.to(v.dtype) * do
        # o = q_trans * s
        dk_trans = ds * v
        # d e -> d 1
        dk_trans = tl.sum(dk_trans, axis=1)[:, None]

        dv = ds * k_trans
        dv = tl.sum(dv, axis=0)[None, :]

        tl.store(dk_trans_block_ptr, dk_trans.to(dk_trans_block_ptr.dtype.element_ty))
        tl.store(dv_block_ptr, dv.to(dv_block_ptr.dtype.element_ty))

    # if OUTPUT_FINAL_STATE:
    #     s_final_block_ptr = (
    #         S_FINAL_STATE
    #         + off_s
    #         + tl.arange(0, d)[:, None] * e
    #         + off_e
    #         + tl.arange(0, BLOCK_E)[None, :]
    #     )

    #     tl.store(
    #         s_final_block_ptr,
    #         s.to(s_final_block_ptr.dtype.element_ty),
    #         mask=mask,
    #     )


class GrpeRecurrenceFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx, q, k, v, alpha, beta, gamma, initial_state=None, output_final_state=None
    ):
        b, h, n, d = q.shape
        e = v.shape[-1]

        # m = exp(alpha + beta * gamma * gamma ^ T)
        identity = torch.eye(d, device=torch.cuda.current_device())
        order_one_term = alpha.unsqueeze(-1) * identity
        order_two_term = (
            beta.unsqueeze(-1).unsqueeze(-1) * gamma.unsqueeze(-1) * gamma.unsqueeze(-2)
        )
        log_m = order_one_term + order_two_term
        m = torch.matrix_exp(log_m)

        o = torch.empty((b, h, n, e), dtype=q.dtype, device=torch.cuda.current_device())

        if initial_state is not None:
            s_initial_state = initial_state
        else:
            s_initial_state = None

        if output_final_state:
            s_final_state = torch.empty(
                (b, h, d, e), dtype=torch.float32, device=torch.cuda.current_device()
            )
        else:
            s_final_state = None

        USE_INITIAL_STATE = initial_state is not None
        OUTPUT_FINAL_STATE = output_final_state

        def grid(meta):
            return (b * h, triton.cdiv(e, meta["BLOCK_E"]))

        _grpe_recurrence_fwd[grid](
            q,
            k,
            v,
            m,
            o,
            s_initial_state,
            s_final_state,
            b,
            h,
            n,
            d,
            e,
            USE_INITIAL_STATE,
            OUTPUT_FINAL_STATE,
        )

        if OUTPUT_FINAL_STATE:
            final_state = s_final_state
        else:
            final_state = None

        ctx.save_for_backward(q, k, v, alpha, beta, gamma, initial_state)
        ctx.output_final_state = output_final_state

        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, ds):
        q, k, v, alpha, beta, gamma, initial_state = ctx.saved_tensors
        output_final_state = ctx.output_final_state

        b, h, n, d = q.shape
        e = v.shape[-1]

        # m = exp(alpha + beta * gamma * gamma ^ T)
        identity = torch.eye(d, device=torch.cuda.current_device())
        order_one_term = alpha.unsqueeze(-1) * identity
        order_two_term = (
            beta.unsqueeze(-1).unsqueeze(-1) * gamma.unsqueeze(-1) * gamma.unsqueeze(-2)
        )
        log_m = order_one_term + order_two_term
        m = torch.matrix_exp(log_m)

        dq = torch.empty_like(q, dtype=q.dtype, device=torch.cuda.current_device())
        dk = torch.empty_like(k, dtype=q.dtype, device=torch.cuda.current_device())
        dv = torch.empty_like(v, dtype=q.dtype, device=torch.cuda.current_device())
        ds_ = (
            torch.empty((b, h, d, e), dtype=q.dtype, device=torch.cuda.current_device())
            if initial_state is not None
            else None
        )
        dalpha = torch.empty_like(
            alpha, dtype=q.dtype, device=torch.cuda.current_device()
        )
        dbeta = torch.empty_like(
            beta, dtype=q.dtype, device=torch.cuda.current_device()
        )
        dgamma = torch.empty_like(
            gamma, dtype=q.dtype, device=torch.cuda.current_device()
        )
        dm = torch.empty_like(m, dtype=q.dtype, device=torch.cuda.current_device())

        if initial_state is not None:
            s_initial_state = initial_state
        else:
            s_initial_state = None

        if output_final_state:
            s_final_state = torch.empty(
                (b, h, d, e), dtype=torch.float32, device=torch.cuda.current_device()
            )
        else:
            s_final_state = None

        USE_INITIAL_STATE = initial_state is not None
        OUTPUT_FINAL_STATE = output_final_state

        grid = (b * h,)

        _grpe_recurrence_bwd[grid](
            q,
            k,
            v,
            m,
            do,
            dq,
            dk,
            dv,
            dm,
            ds_,
            s_initial_state,
            s_final_state,
            b,
            h,
            n,
            d,
            e,
            USE_INITIAL_STATE,
            OUTPUT_FINAL_STATE,
        )

        return dq, dk, dv, dalpha, dbeta, dgamma, ds_, None


def grpe_recurrence_triton(
    q, k, v, alpha, beta, gamma, initial_state=None, output_final_state=False
):
    o, final_state = GrpeRecurrenceFunction.apply(
        q, k, v, alpha, beta, gamma, initial_state, output_final_state
    )
    return o, final_state


if __name__ == "__main__":
    import torch.nn.functional as F

    b, h, n, d, e = 2, 8, 128, 64, 32
    dtype = torch.float32
    device = torch.cuda.current_device()
    q = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    k = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    v = (torch.randn((b, h, n, e), dtype=dtype, device=device)).requires_grad_()
    lower_bound = 0.95
    alpha = torch.log(
        lower_bound
        + (1 - lower_bound)
        * F.sigmoid(torch.randn((b, h, n, d), dtype=dtype, device=device))
    ).requires_grad_()
    beta = torch.log(
        lower_bound
        + (1 - lower_bound)
        * F.sigmoid(torch.randn((b, h, n), dtype=dtype, device=device))
    ).requires_grad_()
    gamma = F.normalize(
        torch.randn((b, h, n, d), dtype=dtype, device=device)
    ).requires_grad_()
    initial_state = None
    output_final_state = False

    o_recurrence_torch, final_state_recurrence_torch = grpe_recurrence_triton(
        q, k, v, alpha, beta, gamma, initial_state, output_final_state
    )
