from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs

MIN_CHUNK_SIZE = 64
MAX_BLOCK_SIZE = 128


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_N": [MIN_CHUNK_SIZE, MIN_CHUNK_SIZE * 2, MIN_CHUNK_SIZE * 4],
            "BLOCK_E": [MAX_BLOCK_SIZE, MAX_BLOCK_SIZE // 2, MAX_BLOCK_SIZE // 4],
        }
    ),
    key=["B", "N", "H", "D", "E"],
)
@triton.jit
def _lavd_intra_fwd(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    LDK,  # B N H D
    LDV,  # B N H E
    O,  # B N H E
    STATES,  # B M H D E, store from index 1
    LOG_PI,  # B N H D
    LOG_RHO,  # B N H E
    BLOCK_SIZE,  # 1
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    USE_LDK: tl.constexpr,
    USE_LDV: tl.constexpr,
    HAS_LDK: tl.constexpr,
    HAS_LDV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_c = tl.program_id(1)
    off_e = tl.program_id(2)
    off_b = off_bh // H
    off_h = off_bh % H

    # compute offset
    offset_n = off_c * BLOCK_N
    offset_e = off_e * BLOCK_E
    offset_qk = off_b * N * H * D + offset_n * H * D + off_h * D
    offset_vo = off_b * N * H * E + offset_n * H * E + off_h * E
    # store from index 1
    offset_state = off_b * M * H * D * E + (off_c + 1) * H * D * E + off_h * D * E

    # compute block ptr
    array_nd = tl.full([BLOCK_D], value=offset_n, dtype=tl.int64)
    array_ne = tl.full([BLOCK_E], value=offset_n, dtype=tl.int64)
    array_d = tl.arange(0, BLOCK_D)
    array_e = offset_e + tl.arange(0, BLOCK_E)
    mask_d = array_d < D
    mask_e = array_e < E
    q_block_ptr = Q + offset_qk + array_d
    k_block_ptr = K + offset_qk + array_d
    v_block_ptr = V + offset_vo + array_e
    o_block_ptr = O + offset_vo + array_e
    if USE_LDK and HAS_LDK:
        ldk_block_ptr = LDK + offset_qk + array_d
    if USE_LDV and HAS_LDV:
        ldv_block_ptr = LDV + offset_vo + array_e
    state_block_ptr = STATES + offset_state + array_d[:, None] * E + array_e[None, :]

    # compute
    state = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)
    if USE_LDK:
        log_pi = tl.zeros([BLOCK_D], dtype=tl.float32)
        log_pi_block_ptr = LOG_PI + offset_qk + array_d

    if USE_LDV:
        log_rho = tl.zeros([BLOCK_E], dtype=tl.float32)
        log_rho_block_ptr = LOG_RHO + offset_vo + array_e

    for i in range(BLOCK_N):
        if offset_n < N:
            # mask
            mask_nd = array_nd < N
            mask_ne = array_ne < N

            q = tl.load(q_block_ptr, mask=mask_d, other=0)
            k = tl.load(k_block_ptr, mask=mask_d, other=0)
            v = tl.load(v_block_ptr, mask=mask_e, other=0)
            q = tl.where(mask_nd, q, 0)
            k = tl.where(mask_nd, k, 0)
            v = tl.where(mask_ne, v, 0)
            state_ = k[:, None] * v[None, :]

            if USE_LDK:
                if HAS_LDK:
                    ldk = tl.load(ldk_block_ptr, mask=mask_d, other=0).to(tl.float32)
                    log_pi += ldk
                    lambda_ = tl.exp(ldk)
                    ldk_block_ptr += H * D
                else:
                    lambda_ = 1 - k.to(tl.float32)
                    log_pi += tl.log(lambda_)
                lambda_ = tl.where(mask_nd, lambda_, 1)
                log_pi = tl.where(mask_nd, log_pi, 0)
                state = lambda_[:, None] * state

                tl.store(
                    log_pi_block_ptr,
                    log_pi.to(log_pi_block_ptr.dtype.element_ty),
                    mask=mask_d,
                )
                log_pi_block_ptr += H * D

            if USE_LDV:
                if HAS_LDV:
                    ldv = tl.load(ldv_block_ptr, mask=mask_e, other=0).to(tl.float32)
                    log_rho += ldv
                    gamma_ = tl.exp(ldv)
                    ldv_block_ptr += H * E
                else:
                    gamma_ = 1 - v.to(tl.float32)
                    log_rho += tl.log(gamma_)
                gamma_ = tl.where(mask_ne, gamma_, 1)
                log_rho = tl.where(mask_ne, log_rho, 0)
                state = state * gamma_[None, :]

                tl.store(
                    log_rho_block_ptr,
                    log_rho.to(log_rho_block_ptr.dtype.element_ty),
                    mask=mask_e,
                )
                log_rho_block_ptr += H * E

            state += state_
            o = tl.sum(q[:, None] * state, axis=0)
            tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask_e)

            q_block_ptr += H * D
            k_block_ptr += H * D
            v_block_ptr += H * E
            o_block_ptr += H * E
            array_nd += 1
            array_ne += 1
            offset_n += 1

    tl.store(
        state_block_ptr,
        state.to(state_block_ptr.dtype.element_ty),
        mask=mask_d[:, None] & mask_e[None, :],
    )

    if off_bh == 0:
        if off_c == 0:
            if off_e == 0:
                block_size_ptr = BLOCK_SIZE
                tl.store(block_size_ptr, BLOCK_N)


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_D": [128, 64, 32, 16],
            "BLOCK_E": [16, 32, 64, 128],
        }
    ),
    key=["B", "N", "H", "D", "E"],
)
@triton.jit
def _lavd_state_reduce(
    STATES,  # B M H D E
    LOG_PI,  # B N H D
    LOG_RHO,  # B N H E
    INITIAL_STATE,  # H D E if USE_STATIC_INITIAL_STATE else B H D E
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    USE_LDK: tl.constexpr,
    USE_LDV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_STATIC_INITIAL_STATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)
    off_b = off_bh // H
    off_h = off_bh % H

    # compute offset
    offset_d = off_d * BLOCK_D
    offset_e = off_e * BLOCK_E
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    # start from index 1
    offset_state = (
        off_b * M * H * D * E + H * D * E + off_h * D * E + offset_d * E + offset_e
    )
    NUM_BLOCK = tl.cdiv(N, BLOCK_N)
    offset_log_pi = off_b * N * H * D + off_h * D + offset_d
    offset_log_rho = off_b * N * H * E + off_h * E + offset_e
    # !!! important
    if NUM_BLOCK == 1:
        if N % BLOCK_N > 0:
            offset_block = N % BLOCK_N - 1
        else:
            offset_block = BLOCK_N - 1
    else:
        offset_block = BLOCK_N - 1
    offset_log_pi = off_b * N * H * D + offset_block * H * D + off_h * D + offset_d
    offset_log_rho = off_b * N * H * E + offset_block * H * E + off_h * E + offset_e

    if USE_INITIAL_STATE:
        if USE_STATIC_INITIAL_STATE:
            offset_initial_state = off_h * D * E + offset_d * E + offset_e
        else:
            offset_initial_state = off_bh * D * E + offset_d * E + offset_e

    # compute block ptr
    array_nd = tl.full([BLOCK_D], value=0, dtype=tl.int64)
    array_ne = tl.full([BLOCK_E], value=0, dtype=tl.int64)
    mask_d = (offset_d + array_d) < D
    mask_e = (offset_e + array_e) < E
    mask_state = mask_d[:, None] & mask_e[None, :]
    state_block_ptr = STATES + offset_state + array_d[:, None] * E + array_e[None, :]
    state_save_block_ptr = state_block_ptr - H * D * E
    if USE_LDK:
        log_pi_block_ptr = LOG_PI + offset_log_pi + array_d
    if USE_LDV:
        log_rho_block_ptr = LOG_RHO + offset_log_rho + array_e
    if USE_INITIAL_STATE:
        initial_state_block_ptr = (
            INITIAL_STATE
            + offset_initial_state
            + array_d[:, None] * E
            + array_e[None, :]
        )
        state = tl.load(initial_state_block_ptr, mask=mask_state, other=0).to(
            tl.float32
        )
    else:
        state = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)

    tl.store(
        state_save_block_ptr,
        state.to(state_save_block_ptr.dtype.element_ty),
        mask=mask_state,
    )

    for i in range(NUM_BLOCK):
        array_nd < N
        array_ne < N
        state_ = tl.load(state_block_ptr, mask=mask_state, other=0).to(tl.float32)
        if USE_LDK:
            log_pi = tl.load(log_pi_block_ptr, mask=mask_d, other=0).to(tl.float32)
            pi = tl.exp(log_pi)
            state *= pi[:, None]
            if i < NUM_BLOCK - 2:
                log_pi_block_ptr += BLOCK_N * H * D
            else:
                if N % BLOCK_N > 0:
                    log_pi_block_ptr += (N % BLOCK_N) * H * D
                else:
                    log_pi_block_ptr += BLOCK_N * H * D

        if USE_LDV:
            log_rho = tl.load(log_rho_block_ptr, mask=mask_e, other=0).to(tl.float32)
            rho = tl.exp(log_rho)
            state *= rho[None, :]
            if i < NUM_BLOCK - 2:
                log_rho_block_ptr += BLOCK_N * H * E
            else:
                if N % BLOCK_N > 0:
                    log_rho_block_ptr += (N % BLOCK_N) * H * E
                else:
                    log_rho_block_ptr += BLOCK_N * H * E

        state += state_

        array_nd += BLOCK_N
        array_ne += BLOCK_N
        state_block_ptr += H * D * E
        state_save_block_ptr += H * D * E

        tl.store(
            state_save_block_ptr,
            state.to(state_save_block_ptr.dtype.element_ty),
            mask=mask_state,
        )


class LavdChunkParallelFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        ldk=None,
        ldv=None,
        use_ldk=True,
        use_ldv=False,
        initial_state=None,
    ):
        # Get shapes and device
        dtype = q.dtype
        b, n, h, d = q.shape
        e = v.shape[-1]
        device = q.device

        m = (n + MIN_CHUNK_SIZE - 1) // MIN_CHUNK_SIZE + 1
        has_ldk = ldk is not None
        has_ldv = ldv is not None

        static_state = initial_state is not None and len(initial_state.shape) == 3
        o = torch.empty((b, n, h, e), dtype=dtype, device=device)
        states = torch.empty((b, m, h, d, e), dtype=dtype, device=device)
        if use_ldk:
            log_pi = torch.empty((b, n, h, d), dtype=dtype, device=device)
        else:
            log_pi = None
        if use_ldv:
            log_rho = torch.empty((b, n, h, e), dtype=dtype, device=device)
        else:
            log_rho = None
        block_d = min(triton.next_power_of_2(d), MAX_BLOCK_SIZE)
        block_size = torch.empty((1,), dtype=torch.int32, device=device)
        # block_e = min(triton.next_power_of_two(e), MAX_BLOCK_SIZE)

        ##### Use three pass to compute the output and state
        # 1. Compute intra and local state
        # 2. Compute global state
        # 3. Compute inter

        # 1. Compute intra and local state
        def grid(meta):
            return (
                b * h,
                triton.cdiv(n, meta["BLOCK_N"]),
                triton.cdiv(e, meta["BLOCK_E"]),
            )

        _lavd_intra_fwd[grid](
            Q=q,
            K=k,
            V=v,
            LDK=ldk,
            LDV=ldv,
            O=o,
            STATES=states,
            LOG_PI=log_pi,
            LOG_RHO=log_rho,
            BLOCK_SIZE=block_size,
            B=b,
            N=n,
            H=h,
            D=d,
            E=e,
            M=m,
            USE_LDK=use_ldk,
            USE_LDV=use_ldv,
            HAS_LDK=has_ldk,
            HAS_LDV=has_ldv,
            BLOCK_D=block_d,
        )

        # 2. Compute global state
        block_n = block_size.item()

        def grid(meta):
            # meta["BLOCK_D"] = min(meta["BLOCK_D"], d)
            # meta["BLOCK_E"] = min(meta["BLOCK_E"], e)
            return (
                b * h,
                triton.cdiv(d, meta["BLOCK_D"]),
                triton.cdiv(e, meta["BLOCK_E"]),
            )

        use_initial_state = initial_state is not None
        use_static_initial_state = (
            initial_state is not None and len(initial_state.shape) == 3
        )

        _lavd_state_reduce[grid](
            STATES=states,
            LOG_PI=log_pi,
            LOG_RHO=log_rho,
            INITIAL_STATE=initial_state,
            B=b,
            N=n,
            H=h,
            D=d,
            E=e,
            M=m,
            USE_LDK=use_ldk,
            USE_LDV=use_ldv,
            USE_INITIAL_STATE=use_initial_state,
            USE_STATIC_INITIAL_STATE=use_static_initial_state,
            BLOCK_N=block_n,
        )

        # # 3. Compute inter
        # # c = block_size.item()
        # # def grid(meta):
        # #     return (b * h, triton.cdiv(n, c))
        # c = block_size.item()
        # m = (n + c - 1) // c
        # pi = torch.exp(log_pi)
        # rho = torch.exp(log_rho)
        # # update
        # q_ = q * pi
        # q_ = rearrange(q_, "b (m c) h d -> b m c h d", c=c)
        # rho = rearrange(rho, "b (m c) h e -> b m c h e", c=c)
        # # print(q.shape, state.shape, c, n, m, q_.shape, state[:, :m].shape, rho.shape)
        # print(q_.shape, state[:, :m].contiguous().shape, rho.shape)
        # o_inter = (
        #     torch.einsum("b m c h d, b m h d e -> b m c h e", q_, state[:, :m].contiguous())
        #     * rho
        # )
        # o_inter = rearrange(o_inter, "b m c h e -> b (m c) h e")
        # o += o_inter

        # # Save for backward
        # state = state[:, :m].contiguous()
        # ctx.save_for_backward(q, ldk, ldv, k, v, state[:, :m].contiguous())
        # ctx.static_state = static_state
        # ctx.chunk_size = c

        state = None

        return o, state, states[:, :m].contiguous(), log_pi, log_rho

    @staticmethod
    @contiguous
    def backward(ctx, do, dstate):
        q, ldk, ldv, k, v, state = ctx.saved_tensors
        static_state = ctx.static_state
        ctx.chunk_size

        # Get shapes and convert to float32
        b, n, h, d = q.shape
        ldv.shape[-1]
        dtype = q.dtype
        q.device

        # Convert to float32
        do = do.float()
        if dstate is not None:
            dstate = dstate.float()

        # Allocate gradients
        dq = torch.empty_like(q, dtype=torch.float32)
        dldk = torch.empty_like(ldk, dtype=torch.float32)
        dldv = torch.empty_like(ldv, dtype=torch.float32)
        dk = torch.empty_like(q, dtype=torch.float32) if k is not None else None
        dv = torch.empty_like(ldv, dtype=torch.float32) if v is not None else None

        # Launch kernel
        def grid(meta):
            return (triton.cdiv(b, meta["BLOCK_B"]), triton.cdiv(n, meta["BLOCK_N"]))

        # Handle static state gradient
        if static_state and dstate is not None:
            dstate = dstate.sum(dim=0)

        return (
            dq.to(dtype),
            dldk.to(dtype),
            dldv.to(dtype),
            dk.to(dtype) if dk is not None else None,
            dv.to(dtype) if dv is not None else None,
            dstate.to(dtype) if dstate is not None else None,
            None,  # chunk_size
        )


def lavd_chunk_parallel_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements Lightning Attention with Vector Decay with chunking parallel using Triton.

    Args:
        q: Query tensor, shape (B, N, H, D)
        k: Key tensor, shape (B, N, H, D)
        v: Value tensor, shape (B, N, H, E)
        ldk: Log Decay vector for key, shape (B, N, H, D), if not provided uses log(1 - exp(k))
        ldv: Log Decay vector for value, shape (B, N, H, E), if not provided uses log(1 - exp(v))
        use_ldk: Whether to use log decay for key
        use_ldv: Whether to use log decay for value
        initial_state: Initial state tensor, shape (B, H, D, E) or (H, D, E)

    Returns:
        Output tensor, shape (B, N, H, E)
        State tensor, shape (B, H, D, E)
    """
    if ldk is not None:
        use_ldk = True
    if ldv is not None:
        use_ldv = True
    assert use_ldk or use_ldv, "At least one of ldk or ldv must be used"
    return LavdChunkParallelFunction.apply(
        q, k, v, ldk, ldv, use_ldk, use_ldv, initial_state
    )


if __name__ == "__main__":
    # Test the implementation
    b, n, h, d = 2, 128, 12, 128
    e = 64
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
    ldk = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    ldv = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
    initial_state = torch.randn(
        (b, h, d, e), dtype=dtype, device=device
    ).requires_grad_()

    o, state = lavd_chunk_parallel_triton(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        initial_state=initial_state,
    )
    # (o.sum() + state.sum()).backward()
    print(o.shape)
