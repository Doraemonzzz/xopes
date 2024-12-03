import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, next_power_of_two


@triton.jit
def _page_flip_prepare(
    W,  # B N H D
    INIT_STATE1,  # B H D
    INIT_STATE2,  # B H D
    W_CUM_CUM,  # B N H D
    FINAL_STATE1,  # B H D
    FINAL_STATE2,  # B H D
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_INIT_STATE: tl.constexpr,
    OUTPUT_FINAL_STATE: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    # compute offset
    offset_w = off_b * N * H * D + off_h * D
    offset_w_cum_cum = off_b * (N + 1) * H * D + off_h * D
    # mask
    d_mask = tl.arange(0, BLOCK) < D
    # compute block ptr
    w_block_ptr = W + offset_w + tl.arange(0, BLOCK)
    w_cum_cum_block_ptr = W_CUM_CUM + offset_w_cum_cum + tl.arange(0, BLOCK)

    if USE_INIT_STATE:
        # compute offset
        offset_state = off_b * H * D + off_h * D
        state1_block_ptr = INIT_STATE1 + offset_state + tl.arange(0, BLOCK)
        state2_block_ptr = INIT_STATE2 + offset_state + tl.arange(0, BLOCK)

        w_state = tl.load(state1_block_ptr, mask=d_mask, other=0.0).to(tl.float32)
        w_cum_state = tl.load(state2_block_ptr, mask=d_mask, other=0.0).to(tl.float32)
    else:
        w_state = tl.zeros([BLOCK], dtype=tl.float32)
        w_cum_state = tl.zeros([BLOCK], dtype=tl.float32)

    # save init state
    tl.store(w_cum_cum_block_ptr, w_cum_state, mask=d_mask)
    w_cum_cum_block_ptr += H * D

    for i in range(N):
        w = tl.load(w_block_ptr, mask=d_mask, other=0.0)
        w_state += w
        w_cum_state += w_state

        tl.store(w_cum_cum_block_ptr, w_cum_state, mask=d_mask)

        w_block_ptr += H * D
        w_cum_cum_block_ptr += H * D

    if OUTPUT_FINAL_STATE:
        # compute offset
        offset_state = off_b * H * D + off_h * D
        final_state1_block_ptr = FINAL_STATE1 + offset_state + tl.arange(0, BLOCK)
        final_state2_block_ptr = FINAL_STATE2 + offset_state + tl.arange(0, BLOCK)

        tl.store(
            final_state1_block_ptr,
            w_state.to(final_state1_block_ptr.dtype.element_ty),
            mask=d_mask,
        )
        tl.store(
            final_state2_block_ptr,
            w_cum_state.to(final_state2_block_ptr.dtype.element_ty),
            mask=d_mask,
        )


@triton.jit
def _page_flip_recrurrence_fwd(
    Q,  # B N H D
    V,  # B N H E
    W,  # B N H D
    W_CUM_CUM,  # B (N + 1) H D
    K,  # B N H D
    O_GATE,  # B N H E
    INIT_STATE1,  # B H D E
    INIT_STATE2,  # B H D E
    O,  # B N H E
    FINAL_STATE1,  # B H D E
    FINAL_STATE2,  # B H D E
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    DECAY_TYPE: tl.constexpr,
    USE_NORMALIZE: tl.constexpr,
    USE_K: tl.constexpr,
    USE_O_GATE: tl.constexpr,
    USE_INIT_STATE: tl.constexpr,
    OUTPUT_FINAL_STATE: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    # compute offset
    offset_qk = off_b * N * H * D + off_h * D
    offset_w = off_b * N * H * D + off_h * D
    offset_w_cum_cum = off_b * (N + 1) * H * D + off_h * D
    offset_vo = off_b * N * H * E + off_h * E
    # mask
    d_mask = tl.arange(0, BLOCK_D) < D
    e_mask = tl.arange(0, BLOCK_E) < E
    # compute block ptr
    q_block_ptr = Q + offset_qk + tl.arange(0, BLOCK_D)
    v_block_ptr = V + offset_vo + tl.arange(0, BLOCK_E)
    w_block_ptr = W + offset_w + tl.arange(0, BLOCK_D)
    w_cum_cum_block_ptr = W_CUM_CUM + offset_w_cum_cum + tl.arange(0, BLOCK_D)
    o_block_ptr = O + offset_vo + tl.arange(0, BLOCK_E)
    if USE_K:
        k_block_ptr = K + offset_qk + tl.arange(0, BLOCK_D)
    if USE_O_GATE:
        o_gate_block_ptr = O_GATE + offset_vo + tl.arange(0, BLOCK_E)

    if USE_INIT_STATE:
        # compute offset
        offset_state = off_b * H * D * E + off_h * D * E
        state1_block_ptr = (
            INIT_STATE1
            + offset_state
            + tl.arange(0, BLOCK_D)[:, None] * E
            + tl.arange(0, BLOCK_E)[None, :]
        )
        state2_block_ptr = (
            INIT_STATE2
            + offset_state
            + tl.arange(0, BLOCK_D)[:, None] * E
            + tl.arange(0, BLOCK_E)[None, :]
        )

        state1 = tl.load(
            state1_block_ptr, mask=d_mask[:, None] & e_mask[None, :], other=0.0
        ).to(tl.float32)
        state2 = tl.load(
            state2_block_ptr, mask=d_mask[:, None] & e_mask[None, :], other=0.0
        ).to(tl.float32)
    else:
        state1 = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)
        state2 = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)

    w1 = tl.load(w_cum_cum_block_ptr, mask=d_mask, other=1).to(tl.float32)
    w_cum_cum_block_ptr += H * D

    for i in range(N):
        q = tl.load(q_block_ptr, mask=d_mask, other=0.0).to(tl.float32)
        v = tl.load(v_block_ptr, mask=e_mask, other=0.0).to(tl.float32)
        w2 = tl.load(w_cum_cum_block_ptr, mask=d_mask, other=0.0).to(tl.float32)

        if DECAY_TYPE == "additive":
            r = w1 / w2
        else:
            r = tl.exp(w1 - w2)

        if USE_NORMALIZE or (not USE_K):
            w = tl.load(w_block_ptr, mask=d_mask, other=0.0).to(tl.float32)
            if DECAY_TYPE == "additive":
                norm_factor = w / w2
            else:
                norm_factor = tl.exp(w - w2)

            if USE_K:
                k = tl.load(k_block_ptr, mask=d_mask, other=0.0).to(tl.float32)
                norm_factor *= k

            s = norm_factor[:, None] * v[None, :]

            w_block_ptr += H * D
        else:
            k = tl.load(k_block_ptr, mask=d_mask, other=0.0).to(tl.float32)
            s = k[:, None] * v[None, :]

        state1 = r[:, None] * state1 + s
        state2 = r[:, None] * state2 + state1
        # d, d e -> e
        o = tl.sum(q[:, None] * state2, axis=0)

        if USE_O_GATE:
            o_gate = tl.load(o_gate_block_ptr, mask=e_mask, other=0.0).to(tl.float32)
            o *= o_gate
            o_gate_block_ptr += H * E

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=e_mask)

        q_block_ptr += H * D
        v_block_ptr += H * E
        w_cum_cum_block_ptr += H * D
        o_block_ptr += H * E
        w1 = w2

    if OUTPUT_FINAL_STATE:
        # compute offset
        offset_state = off_b * H * D * E + off_h * D * E
        final_state1_block_ptr = (
            FINAL_STATE1
            + offset_state
            + tl.arange(0, BLOCK_D)[:, None] * E
            + tl.arange(0, BLOCK_E)[None, :]
        )
        final_state2_block_ptr = (
            FINAL_STATE2
            + offset_state
            + tl.arange(0, BLOCK_D)[:, None] * E
            + tl.arange(0, BLOCK_E)[None, :]
        )

        tl.store(
            final_state1_block_ptr,
            state1.to(final_state1_block_ptr.dtype.element_ty),
            mask=d_mask[:, None] & e_mask[None, :],
        )
        tl.store(
            final_state2_block_ptr,
            state2.to(final_state2_block_ptr.dtype.element_ty),
            mask=d_mask[:, None] & e_mask[None, :],
        )


class PageFlipRecurrenceTriton(torch.autograd.Function):
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
        decay_type="additive",
        use_normalize=False,
    ):
        use_k = k is not None
        use_o_gate = o_gate is not None
        b, n, h, d = q.shape
        e = v.shape[-1]
        o = torch.empty((b, n, h, e), dtype=q.dtype, device=torch.cuda.current_device())
        w_cum_cum = torch.empty(
            (b, n + 1, h, d), dtype=q.dtype, device=torch.cuda.current_device()
        )
        use_init_state = initial_state is not None
        if use_init_state:
            state1 = initial_state[0]
            state2 = initial_state[1]
            state3 = initial_state[2]
            state4 = initial_state[3]
        else:
            state1 = None
            state2 = None
            state3 = None
            state4 = None

        if output_final_state:
            o_state1 = torch.empty(
                (b, h, d), dtype=torch.float32, device=torch.cuda.current_device()
            )
            o_state2 = torch.empty(
                (b, h, d), dtype=torch.float32, device=torch.cuda.current_device()
            )
            o_state3 = torch.empty(
                (b, h, d, e), dtype=torch.float32, device=torch.cuda.current_device()
            )
            o_state4 = torch.empty(
                (b, h, d, e), dtype=torch.float32, device=torch.cuda.current_device()
            )
        else:
            o_state1 = None
            o_state2 = None
            o_state3 = None
            o_state4 = None

        grid = (b, h)
        BLOCK_D = max(next_power_of_two(d), 16)
        BLOCK_E = max(next_power_of_two(e), 16)

        _page_flip_prepare[grid](
            W=w,
            INIT_STATE1=state1,
            INIT_STATE2=state2,
            W_CUM_CUM=w_cum_cum,
            FINAL_STATE1=o_state1,
            FINAL_STATE2=o_state2,
            B=b,
            N=n,
            H=h,
            D=d,
            BLOCK=BLOCK_D,
            USE_INIT_STATE=use_init_state,
            OUTPUT_FINAL_STATE=output_final_state,
        )

        _page_flip_recrurrence_fwd[grid](
            Q=q,
            V=v,
            W=w,
            W_CUM_CUM=w_cum_cum,
            K=k,
            O_GATE=o_gate,
            INIT_STATE1=state3,
            INIT_STATE2=state4,
            O=o,
            FINAL_STATE1=o_state3,
            FINAL_STATE2=o_state4,
            B=b,
            N=n,
            H=h,
            D=d,
            E=e,
            BLOCK_D=BLOCK_D,
            BLOCK_E=BLOCK_E,
            DECAY_TYPE=decay_type,
            USE_NORMALIZE=use_normalize,
            USE_K=use_k,
            USE_O_GATE=use_o_gate,
            USE_INIT_STATE=use_init_state,
            OUTPUT_FINAL_STATE=output_final_state,
        )

        # ctx.save_for_backward(x, theta, x_stat1, x_stat2)
        # ctx.offset = offset
        # ctx.act = act
        # ctx.dim = dim

        return o, o_state1, o_state2, o_state3, o_state4

    # @staticmethod
    # @contiguous
    # def backward(ctx, do):
    #     x, theta, x_stat1, x_stat2 = ctx.saved_tensors
    #     offset = ctx.offset
    #     act = ctx.act
    #     dim = ctx.dim

    #     dx = lrpe_cosine_1d_bp_bwd_triton(
    #         x, theta, do, x_stat1, x_stat2, offset, act, dim
    #     )

    #     return dx, None, None, None, None


def page_flip_recurrence_triton(
    q,
    v,
    w,
    k=None,
    o_gate=None,
    initial_state=None,
    output_final_state=False,
    decay_type="additive",
    use_normalize=False,
):
    o, o_state1, o_state2, o_state3, o_state4 = PageFlipRecurrenceTriton.apply(
        q, v, w, k, o_gate, initial_state, output_final_state, decay_type, use_normalize
    )

    return o, [o_state1, o_state2, o_state3, o_state4]


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
    o, final_state = page_flip_recurrence_triton(
        q, v, w, k=k, o_gate=o_gate, use_normalize=use_normalize
    )
    print(o.shape)
