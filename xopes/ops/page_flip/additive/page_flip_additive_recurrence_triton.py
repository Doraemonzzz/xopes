import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, next_power_of_two


@triton.jit
def _page_flip_additive_prepare(
    W,  # B N H D
    INIT_STATE0,  # B H D / H D
    INIT_STATE1,  # B H D / H D
    U,  # B (N + 1) H D
    S,  # B (N + 1) H D
    FINAL_STATE0,  # B H D
    FINAL_STATE1,  # B H D
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    B_STATE: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_INIT_STATE: tl.constexpr,
    OUTPUT_FINAL_STATE: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    # compute offset
    offset_w = off_b * N * H * D + off_h * D
    offset_us = off_b * (N + 1) * H * D + off_h * D
    # mask
    d_mask = tl.arange(0, BLOCK) < D
    # compute block ptr
    w_block_ptr = W + offset_w + tl.arange(0, BLOCK)
    u_block_ptr = U + offset_us + tl.arange(0, BLOCK)
    s_block_ptr = S + offset_us + tl.arange(0, BLOCK)

    if USE_INIT_STATE:
        # compute offset
        if B_STATE:
            offset_state = off_b * H * D + off_h * D
        else:
            offset_state = off_h * D

        state0_block_ptr = INIT_STATE0 + offset_state + tl.arange(0, BLOCK)
        state1_block_ptr = INIT_STATE1 + offset_state + tl.arange(0, BLOCK)

        state0 = tl.load(state0_block_ptr, mask=d_mask, other=0.0).to(tl.float32)
        state1 = tl.load(state1_block_ptr, mask=d_mask, other=0.0).to(tl.float32)
    else:
        state0 = tl.zeros([BLOCK], dtype=tl.float32)
        state1 = tl.zeros([BLOCK], dtype=tl.float32)

    # save init state
    tl.store(u_block_ptr, state0, mask=d_mask)
    tl.store(s_block_ptr, state1, mask=d_mask)
    u_block_ptr += H * D
    s_block_ptr += H * D

    for i in range(N):
        w = tl.load(w_block_ptr, mask=d_mask, other=0.0)
        state0 += w
        state1 += state0

        tl.store(u_block_ptr, state0, mask=d_mask)
        tl.store(s_block_ptr, state1, mask=d_mask)

        w_block_ptr += H * D
        u_block_ptr += H * D
        s_block_ptr += H * D

    if OUTPUT_FINAL_STATE:
        # compute offset
        offset_state = off_b * H * D + off_h * D
        final_state0_block_ptr = FINAL_STATE0 + offset_state + tl.arange(0, BLOCK)
        final_state1_block_ptr = FINAL_STATE1 + offset_state + tl.arange(0, BLOCK)

        tl.store(
            final_state0_block_ptr,
            state0.to(final_state0_block_ptr.dtype.element_ty),
            mask=d_mask,
        )
        tl.store(
            final_state1_block_ptr,
            state1.to(final_state1_block_ptr.dtype.element_ty),
            mask=d_mask,
        )


@triton.jit
def _page_flip_additive_recrurrence_fwd(
    Q,  # B N H D
    V,  # B N H E
    W,  # B N H D
    U,  # B (N + 1) H D
    S,  # B (N + 1) H D
    K,  # B N H D
    O_GATE,  # B N H E
    INIT_STATE0,  # B H D E
    INIT_STATE1,  # B H D E
    O,  # B N H E
    P,  # B (N + 1) H D E
    L,  # B (N + 1) H D E
    FINAL_STATE0,  # B H D E
    FINAL_STATE1,  # B H D E
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    B_STATE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    USE_NORMALIZE: tl.constexpr,
    USE_K: tl.constexpr,
    USE_O_GATE: tl.constexpr,
    USE_INIT_STATE: tl.constexpr,
    OUTPUT_FINAL_STATE: tl.constexpr,
    OUTPUT_HIDDEN_STATE: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    # compute offset
    offset_qk = off_b * N * H * D + off_h * D
    offset_w = off_b * N * H * D + off_h * D
    offset_us = off_b * (N + 1) * H * D + off_h * D
    offset_vo = off_b * N * H * E + off_h * E
    # mask
    d_mask = tl.arange(0, BLOCK_D) < D
    e_mask = tl.arange(0, BLOCK_E) < E
    # compute block ptr
    q_block_ptr = Q + offset_qk + tl.arange(0, BLOCK_D)
    v_block_ptr = V + offset_vo + tl.arange(0, BLOCK_E)
    W + offset_w + tl.arange(0, BLOCK_D)
    u_block_ptr = U + offset_us + tl.arange(0, BLOCK_D)
    s_block_ptr = S + offset_us + tl.arange(0, BLOCK_D)
    o_block_ptr = O + offset_vo + tl.arange(0, BLOCK_E)
    if OUTPUT_HIDDEN_STATE:
        offset_pl = off_b * (N + 1) * H * D * E + off_h * D * E
        p_block_ptr = (
            P
            + offset_pl
            + tl.arange(0, BLOCK_D)[:, None] * E
            + tl.arange(0, BLOCK_E)[None, :]
        )
        l_block_ptr = (
            L
            + offset_pl
            + tl.arange(0, BLOCK_D)[:, None] * E
            + tl.arange(0, BLOCK_E)[None, :]
        )
    if USE_K or (not USE_NORMALIZE):
        k_block_ptr = K + offset_qk + tl.arange(0, BLOCK_D)
    if USE_O_GATE:
        o_gate_block_ptr = O_GATE + offset_vo + tl.arange(0, BLOCK_E)

    if USE_INIT_STATE:
        # compute offset
        if B_STATE:
            offset_state = off_b * H * D * E + off_h * D * E
        else:
            offset_state = off_h * D * E

        state0_block_ptr = (
            INIT_STATE0
            + offset_state
            + tl.arange(0, BLOCK_D)[:, None] * E
            + tl.arange(0, BLOCK_E)[None, :]
        )
        state1_block_ptr = (
            INIT_STATE1
            + offset_state
            + tl.arange(0, BLOCK_D)[:, None] * E
            + tl.arange(0, BLOCK_E)[None, :]
        )

        state0 = tl.load(
            state0_block_ptr, mask=d_mask[:, None] & e_mask[None, :], other=0.0
        ).to(tl.float32)
        state1 = tl.load(
            state1_block_ptr, mask=d_mask[:, None] & e_mask[None, :], other=0.0
        ).to(tl.float32)
    else:
        state0 = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)
        state1 = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)

    u = tl.load(u_block_ptr, mask=d_mask, other=1).to(tl.float32)
    s = tl.load(s_block_ptr, mask=d_mask, other=1).to(tl.float32)
    u_block_ptr += H * D
    s_block_ptr += H * D

    if OUTPUT_HIDDEN_STATE:
        tl.store(
            p_block_ptr,
            state0.to(p_block_ptr.dtype.element_ty),
            mask=d_mask[:, None] & e_mask[None, :],
        )
        tl.store(
            l_block_ptr,
            state1.to(l_block_ptr.dtype.element_ty),
            mask=d_mask[:, None] & e_mask[None, :],
        )
        p_block_ptr += H * D * E
        l_block_ptr += H * D * E

    for i in range(N):
        q = tl.load(q_block_ptr, mask=d_mask, other=0.0).to(tl.float32)
        v = tl.load(v_block_ptr, mask=e_mask, other=0.0).to(tl.float32)
        u1 = tl.load(u_block_ptr, mask=d_mask, other=1).to(tl.float32)
        s1 = tl.load(s_block_ptr, mask=d_mask, other=1).to(tl.float32)

        decay_state0 = u / u1
        decay_state1 = s / s1

        if USE_NORMALIZE:
            x = 1 - decay_state0

            if USE_K:
                k = tl.load(k_block_ptr, mask=d_mask, other=0.0).to(tl.float32)
                x *= k
                k_block_ptr += H * D

            state = x[:, None] * v[None, :]
        else:
            k = tl.load(k_block_ptr, mask=d_mask, other=0.0).to(tl.float32)
            state = k[:, None] * v[None, :]
            k_block_ptr += H * D

        state0 = decay_state0[:, None] * state0 + state
        state1 = decay_state1[:, None] * state1 + (1 - decay_state1[:, None]) * state0
        # d, d e -> e
        o = tl.sum(q[:, None] * state1, axis=0)

        if USE_O_GATE:
            o_gate = tl.load(o_gate_block_ptr, mask=e_mask, other=0.0).to(tl.float32)
            o *= o_gate
            o_gate_block_ptr += H * E

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=e_mask)
        if OUTPUT_HIDDEN_STATE:
            tl.store(
                p_block_ptr,
                state0.to(p_block_ptr.dtype.element_ty),
                mask=d_mask[:, None] & e_mask[None, :],
            )
            tl.store(
                l_block_ptr,
                state1.to(l_block_ptr.dtype.element_ty),
                mask=d_mask[:, None] & e_mask[None, :],
            )
            p_block_ptr += H * D * E
            l_block_ptr += H * D * E

        q_block_ptr += H * D
        v_block_ptr += H * E
        u_block_ptr += H * D
        s_block_ptr += H * D
        o_block_ptr += H * E
        u = u1
        s = s1

    if OUTPUT_FINAL_STATE:
        # compute offset
        offset_state = off_b * H * D * E + off_h * D * E
        final_state0_block_ptr = (
            FINAL_STATE0
            + offset_state
            + tl.arange(0, BLOCK_D)[:, None] * E
            + tl.arange(0, BLOCK_E)[None, :]
        )
        final_state1_block_ptr = (
            FINAL_STATE1
            + offset_state
            + tl.arange(0, BLOCK_D)[:, None] * E
            + tl.arange(0, BLOCK_E)[None, :]
        )

        tl.store(
            final_state0_block_ptr,
            state0.to(final_state0_block_ptr.dtype.element_ty),
            mask=d_mask[:, None] & e_mask[None, :],
        )
        tl.store(
            final_state1_block_ptr,
            state1.to(final_state1_block_ptr.dtype.element_ty),
            mask=d_mask[:, None] & e_mask[None, :],
        )


class PageFlipAdditiveRecurrenceTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        v,
        w,
        k=None,
        o_gate=None,
        initial_state=None,
        output_final_state=False,
        use_normalize=False,
        output_hidden_state=False,
    ):
        use_k = k is not None
        use_o_gate = o_gate is not None
        b, n, h, d = q.shape
        e = v.shape[-1]
        o = torch.empty((b, n, h, e), dtype=q.dtype, device=torch.cuda.current_device())
        u = torch.empty(
            (b, n + 1, h, d), dtype=q.dtype, device=torch.cuda.current_device()
        )
        s = torch.empty(
            (b, n + 1, h, d), dtype=q.dtype, device=torch.cuda.current_device()
        )
        use_init_state = initial_state is not None

        if use_init_state:
            state0 = initial_state[0]
            state1 = initial_state[1]
            state2 = initial_state[2]
            state3 = initial_state[3]
        else:
            state0 = None
            state1 = None
            state2 = None
            state3 = None

        if output_final_state:
            o_state0 = torch.empty(
                (b, h, d), dtype=torch.float32, device=torch.cuda.current_device()
            )
            o_state1 = torch.empty(
                (b, h, d), dtype=torch.float32, device=torch.cuda.current_device()
            )
            o_state2 = torch.empty(
                (b, h, d, e), dtype=torch.float32, device=torch.cuda.current_device()
            )
            o_state3 = torch.empty(
                (b, h, d, e), dtype=torch.float32, device=torch.cuda.current_device()
            )
        else:
            o_state0 = None
            o_state1 = None
            o_state2 = None
            o_state3 = None

        if output_hidden_state:
            p = torch.empty(
                (b, n + 1, h, d, e),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            l = torch.empty(
                (b, n + 1, h, d, e),
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
        else:
            p = None
            l = None

        grid = (b, h)
        BLOCK_D = max(next_power_of_two(d), 16)
        BLOCK_E = max(next_power_of_two(e), 16)

        # whether use batch init state
        b_state = False
        if use_init_state and len(state0.shape) == 3:
            b_state = True

        _page_flip_additive_prepare[grid](
            W=w,
            INIT_STATE0=state0,
            INIT_STATE1=state1,
            U=u,
            S=s,
            FINAL_STATE0=o_state0,
            FINAL_STATE1=o_state1,
            B=b,
            N=n,
            H=h,
            D=d,
            B_STATE=b_state,
            BLOCK=BLOCK_D,
            USE_INIT_STATE=use_init_state,
            OUTPUT_FINAL_STATE=output_final_state,
        )

        _page_flip_additive_recrurrence_fwd[grid](
            Q=q,
            V=v,
            W=w,
            U=u,
            S=s,
            K=k,
            O_GATE=o_gate,
            INIT_STATE0=state2,
            INIT_STATE1=state3,
            O=o,
            P=p,
            L=l,
            FINAL_STATE0=o_state2,
            FINAL_STATE1=o_state3,
            B=b,
            N=n,
            H=h,
            D=d,
            E=e,
            B_STATE=b_state,
            BLOCK_D=BLOCK_D,
            BLOCK_E=BLOCK_E,
            USE_NORMALIZE=use_normalize,
            USE_K=use_k,
            USE_O_GATE=use_o_gate,
            USE_INIT_STATE=use_init_state,
            OUTPUT_FINAL_STATE=output_final_state,
            OUTPUT_HIDDEN_STATE=output_hidden_state,
        )

        # final_state = [o_state0, o_state1, o_state2, o_state3]
        final_state = [u, s, p, l]

        # ctx.save_for_backward(q, v, w, k, o_gate, initial_state, final_state, u, s)
        ctx.output_final_state = output_final_state
        ctx.use_normalize = use_normalize

        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do):
        q, v, w, k, o_gate, initial_state, final_state, u, s = ctx.saved_tensors
        ctx.output_final_state
        ctx.use_normalize

        k is not None
        o_gate is not None
        b, n, h, d = q.shape
        e = v.shape[-1]

        dq = torch.empty(
            (b, n, h, d), dtype=q.dtype, device=torch.cuda.current_device()
        )
        dv = torch.empty(
            (b, n, h, e), dtype=q.dtype, device=torch.cuda.current_device()
        )
        dw = torch.empty(
            (b, n, h, d), dtype=q.dtype, device=torch.cuda.current_device()
        )
        if k is not None:
            dk = torch.empty(
                (b, n, h, d), dtype=q.dtype, device=torch.cuda.current_device()
            )
        else:
            dk = None
        if o_gate is not None:
            do_gate = torch.empty(
                (b, n, h, e), dtype=q.dtype, device=torch.cuda.current_device()
            )
        else:
            do_gate = None

        use_init_state = initial_state[0] is not None
        if use_init_state:
            dstate0 = torch.empty_like(
                initial_state[0], dtype=q.dtype, device=torch.cuda.current_device()
            )
            dstate1 = torch.empty_like(
                initial_state[1], dtype=q.dtype, device=torch.cuda.current_device()
            )
            dstate2 = torch.empty_like(
                initial_state[2], dtype=q.dtype, device=torch.cuda.current_device()
            )
            dstate3 = torch.empty_like(
                initial_state[3], dtype=q.dtype, device=torch.cuda.current_device()
            )
        else:
            pass

        # whether use batch init state
        if USE_INIT_STATE and len(state0.shape) == 3:
            pass

        # compute dq

        # compute dk, dv

        # compute ds

        # compute du

        # compute de

        return dq, dv, dw, dk, do_gate, dinit_state, None, None


def page_flip_additive_recurrence_triton(
    q,
    v,
    w,
    k=None,
    o_gate=None,
    initial_state=None,
    output_final_state=False,
    use_normalize=False,
    output_hidden_state=False,
):
    o, o_state = PageFlipAdditiveRecurrenceTriton.apply(
        q,
        v,
        w,
        k,
        o_gate,
        initial_state,
        output_final_state,
        use_normalize,
        output_hidden_state,
    )

    return o, o_state


if __name__ == "__main__":
    b, n, h, d, e = 2, 512, 8, 128, 64
    dtype = torch.float32
    q = torch.randn((b, n, h, d), dtype=dtype).cuda()
    k = torch.randn((b, n, h, d), dtype=dtype).cuda()
    v = torch.randn((b, n, h, e), dtype=dtype).cuda()
    w = torch.randn((b, n, h, d), dtype=dtype).cuda()
    o_gate = torch.randn((b, n, h, e), dtype=dtype).cuda()
    k = None
    use_normalize = True
    output_hidden_state = True
    o, final_state = page_flip_additive_recurrence_triton(
        q,
        v,
        w,
        k=k,
        o_gate=o_gate,
        use_normalize=use_normalize,
        output_hidden_state=output_hidden_state,
    )
    print(o.shape)
