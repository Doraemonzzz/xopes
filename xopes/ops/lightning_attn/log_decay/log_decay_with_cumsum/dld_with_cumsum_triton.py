from typing import Optional

import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
            "BLOCK_N": [32, 64, 128],
        }
    ),
    key=["B", "N", "H", "D", "E", "USE_FINAL_STATE"],
)
@triton.jit
def _compute_dld_cumsum_kernel(
    DLD_Q,  # B N H F
    DLD_K,  # B N H F
    DLD,  # B N H F or B N H
    FINAL_STATE,  # B H D E
    DFINAL_STATE,  # B H D E
    DLD_STATE,  # B H or B H F
    CU_SEQLENS,  # M
    SUM_OPTION: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    F: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_F: tl.constexpr,
    NUM_BLOCK_D: tl.constexpr,
    NUM_BLOCK_E: tl.constexpr,
):
    NUM_BLOCK_N = triton.cdiv(N, BLOCK_N)
    # Calculate program ID and offsets
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)

    offset_b = off_b * N * H * F
    offset_h = off_h * F
    if SUM_OPTION == -1:
        offset_b_dld = off_b * N * H
        offset_h_dld = off_h

    # Calculate pointers for DLD_Q and DLD_K
    array_n = N - 1 - tl.arange(0, BLOCK_N)
    array_f = tl.arange(0, BLOCK_F)
    mask_f = array_f < F

    # If using final_state, calculate additional term
    if USE_FINAL_STATE:
        if SUM_OPTION == -1:
            dld_state = tl.load(DLD_STATE + off_b * H + offset_h_dld).to(tl.float32)
        else:
            dld_state = tl.load(
                DLD_STATE + off_b * H * F + offset_h + tl.arange(0, BLOCK_F),
                mask=mask_f,
            ).to(tl.float32)

    if SUM_OPTION == -1:
        dld_cumsum = tl.zeros((1,), dtype=tl.float32)
    else:
        dld_cumsum = tl.zeros((BLOCK_F,), dtype=tl.float32)

    for i in range(NUM_BLOCK_N):
        dld_q_block_ptr = (
            DLD_Q + offset_b + array_n[:, None] * H * F + offset_h + array_f[None, :]
        )
        dld_k_block_ptr = (
            DLD_K + offset_b + array_n[:, None] * H * F + offset_h + array_f[None, :]
        )
        if SUM_OPTION == -1:
            dld_block_ptr = DLD + offset_b_dld + array_n[:, None] * H + offset_h_dld
        else:
            dld_block_ptr = (
                DLD + offset_b + array_n[:, None] * H * F + offset_h + array_f[None, :]
            )
        mask_n = array_n >= 0
        mask = mask_n[:, None] & mask_f[None, :]

        # Load values from DLD_Q and DLD_K
        dld_q = tl.load(dld_q_block_ptr, mask=mask, other=0.0).to(tl.float32)
        dld_k = tl.load(dld_k_block_ptr, mask=mask, other=0.0).to(tl.float32)
        dld = dld_q - dld_k

        if SUM_OPTION == -1:
            # BLOCK_N, BLOCK_F -> BLOCK_N, 1
            dld_ = tl.sum(dld, axis=-1, keep_dims=True)
            # BLOCK_N, 1 -> BLOCK_N, 1
            dld__ = tl.cumsum(dld_, axis=0) + dld_cumsum
            # BLOCK_N, 1 -> 1
            dld_cumsum += tl.sum(dld_, axis=0)

            if USE_FINAL_STATE:
                dld__ += dld_state

            # Store result
            tl.store(
                dld_block_ptr,
                dld__.to(dld_block_ptr.dtype.element_ty),
                mask=mask_n[:, None],
            )
        else:
            dld_ = tl.cumsum(dld, axis=0) + dld_cumsum
            dld_cumsum += tl.sum(dld, axis=0)

            if USE_FINAL_STATE:
                dld_ += dld_state

            # Store result
            tl.store(dld_block_ptr, dld_.to(dld_block_ptr.dtype.element_ty), mask=mask)

        array_n -= BLOCK_N


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
        }
    ),
    key=["B", "N", "H", "D", "E", "USE_FINAL_STATE"],
)
@triton.jit
def _compute_dld_state(
    FINAL_STATE,  # B H D E
    DFINAL_STATE,  # B H D E
    DLD_STATE,  # B H or B H D or B H E
    CU_SEQLENS,  # M
    SUM_OPTION: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NUM_BLOCK_D: tl.constexpr,
    NUM_BLOCK_E: tl.constexpr,
):
    # Calculate program ID and offsets
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)

    # Calculate the sum of products of final_state and dfinal_state
    if SUM_OPTION == -1:
        dld_state = tl.zeros((1,), dtype=tl.float32)
        dld_state_block_ptr = DLD_STATE + off_b * H + off_h + tl.arange(0, 1)

        for i in range(NUM_BLOCK_D):
            offset_d = i * BLOCK_D
            array_d = offset_d + tl.arange(0, BLOCK_D)
            mask_d = array_d < D

            for j in range(NUM_BLOCK_E):
                offset_e = j * BLOCK_E
                array_e = offset_e + tl.arange(0, BLOCK_E)
                mask_e = array_e < E
                mask = mask_d[:, None] & mask_e[None, :]

                final_state_block_ptr = (
                    FINAL_STATE
                    + off_b * H * D * E
                    + off_h * D * E
                    + array_d[:, None] * E
                    + array_e[None, :]
                )
                dfinal_state_block_ptr = (
                    DFINAL_STATE
                    + off_b * H * D * E
                    + off_h * D * E
                    + array_d[:, None] * E
                    + array_e[None, :]
                )

                # BLOCK_D, BLOCK_E
                final_state_block = tl.load(
                    final_state_block_ptr, mask=mask, other=0.0
                ).to(tl.float32)
                dfinal_state_block = tl.load(
                    dfinal_state_block_ptr, mask=mask, other=0.0
                ).to(tl.float32)

                dld_state += tl.sum(final_state_block * dfinal_state_block)

        tl.store(dld_state_block_ptr, dld_state.to(DLD_STATE.dtype.element_ty))
    elif SUM_OPTION == 0:
        for i in range(NUM_BLOCK_D):
            offset_d = i * BLOCK_D
            array_d = offset_d + tl.arange(0, BLOCK_D)
            mask_d = array_d < D
            dld_state = tl.zeros((BLOCK_D,), dtype=tl.float32)
            dld_state_block_ptr = DLD_STATE + off_b * H * D + off_h * D + array_d

            for j in range(NUM_BLOCK_E):
                offset_e = j * BLOCK_E
                array_e = offset_e + tl.arange(0, BLOCK_E)
                mask_e = array_e < E
                mask = mask_d[:, None] & mask_e[None, :]

                final_state_block_ptr = (
                    FINAL_STATE
                    + off_b * H * D * E
                    + off_h * D * E
                    + array_d[:, None] * E
                    + array_e[None, :]
                )
                dfinal_state_block_ptr = (
                    DFINAL_STATE
                    + off_b * H * D * E
                    + off_h * D * E
                    + array_d[:, None] * E
                    + array_e[None, :]
                )

                # BLOCK_D, BLOCK_E
                final_state_block = tl.load(
                    final_state_block_ptr, mask=mask, other=0.0
                ).to(tl.float32)
                dfinal_state_block = tl.load(
                    dfinal_state_block_ptr, mask=mask, other=0.0
                ).to(tl.float32)

                dld_state += tl.sum(final_state_block * dfinal_state_block, axis=-1)

            tl.store(
                dld_state_block_ptr,
                dld_state.to(DLD_STATE.dtype.element_ty),
                mask=mask_d,
            )
    else:
        for j in range(NUM_BLOCK_E):
            offset_e = j * BLOCK_E
            array_e = offset_e + tl.arange(0, BLOCK_E)
            mask_e = array_e < E
            dld_state = tl.zeros((BLOCK_E,), dtype=tl.float32)
            dld_state_block_ptr = DLD_STATE + off_b * H * E + off_h * E + array_e

            for i in range(NUM_BLOCK_D):
                offset_d = i * BLOCK_D
                array_d = offset_d + tl.arange(0, BLOCK_D)
                mask_d = array_d < D
                mask = mask_d[:, None] & mask_e[None, :]

                final_state_block_ptr = (
                    FINAL_STATE
                    + off_b * H * D * E
                    + off_h * D * E
                    + array_d[:, None] * E
                    + array_e[None, :]
                )
                dfinal_state_block_ptr = (
                    DFINAL_STATE
                    + off_b * H * D * E
                    + off_h * D * E
                    + array_d[:, None] * E
                    + array_e[None, :]
                )

                # BLOCK_D, BLOCK_E
                final_state_block = tl.load(
                    final_state_block_ptr, mask=mask, other=0.0
                ).to(tl.float32)
                dfinal_state_block = tl.load(
                    dfinal_state_block_ptr, mask=mask, other=0.0
                ).to(tl.float32)

                dld_state += tl.sum(final_state_block * dfinal_state_block, axis=0)

            tl.store(
                dld_state_block_ptr,
                dld_state.to(DLD_STATE.dtype.element_ty),
                mask=mask_e,
            )


@contiguous
def compute_dld_with_cumsum_triton(
    dld_q: torch.Tensor,  # B N H F
    dld_k: torch.Tensor,  # B N H F
    final_state: Optional[torch.Tensor] = None,  # B H D E
    dfinal_state: Optional[torch.Tensor] = None,  # B H D E
    cu_seqlens: Optional[torch.Tensor] = None,  # M
    sum_option: Optional[int] = -1,
):
    """
    Compute the derivative of the log decay using Triton with cumsum.

    Args:
        dld_q: The derivative of the log decay with respect to the query of shape (B, N, H, F), F could be D or E or NUM_FEATURES.
        dld_k: The derivative of the log decay with respect to the key of shape (B, N, H, F), F could be D or E or NUM_FEATURES.
        final_state: The final state of the recurrence of shape (B, H, D, E).
        dfinal_state: The derivative of the final state of the recurrence of shape (B, H, D, E).
        cu_seqlens: The cumulative sequence lengths of the query of shape (M,).
        sum_option: The option to sum the derivative of the log decay over the dimension,
            -1: for dld_q and dld_k, sum over the last dimension,
            0: for final_state and dfinal_state, sum over the e dimension,
            1: for dfinal_state and dld_k, sum over the d dimension.

    Returns:
        dld: The derivative of the log decay.
    """
    b, n, h, f = dld_q.shape

    # Create output tensor
    if sum_option == -1:
        dld = torch.empty(b, n, h, device=dld_q.device, dtype=dld_q.dtype)
    else:
        dld = torch.empty_like(dld_q, dtype=dld_q.dtype)

    # Determine if using final_state
    use_final_state = final_state is not None and dfinal_state is not None

    if use_final_state:
        d, e = final_state.shape[-2], final_state.shape[-1]
        if sum_option == -1:
            dld_state = torch.empty(
                (
                    b,
                    h,
                ),
                dtype=dld_q.dtype,
                device=dld_q.device,
            )
        elif sum_option == 0:
            dld_state = torch.empty(
                (
                    b,
                    h,
                    d,
                ),
                dtype=dld_q.dtype,
                device=dld_q.device,
            )
        elif sum_option == 1:
            dld_state = torch.empty(
                (
                    b,
                    h,
                    e,
                ),
                dtype=dld_q.dtype,
                device=dld_q.device,
            )
        else:
            raise ValueError(f"Invalid sum_option: {sum_option}")
    else:
        d = f
        e = f
        dld_state = None

    # Determine if using cu_seqlens
    use_cu_seqlens = cu_seqlens is not None

    BLOCK_D = min(128, triton.next_power_of_2(d))
    NUM_BLOCK_D = triton.cdiv(d, BLOCK_D)
    BLOCK_E = min(128, triton.next_power_of_2(e))
    NUM_BLOCK_E = triton.cdiv(e, BLOCK_E)
    BLOCK_F = triton.next_power_of_2(f)

    if use_final_state:
        grid = (b, h)

        _compute_dld_state[grid](
            FINAL_STATE=final_state,
            DFINAL_STATE=dfinal_state,
            DLD_STATE=dld_state,
            CU_SEQLENS=cu_seqlens,
            SUM_OPTION=sum_option,
            B=b,
            N=n,
            H=h,
            D=d,
            E=e,
            USE_CU_SEQLENS=use_cu_seqlens,
            BLOCK_D=BLOCK_D,
            BLOCK_E=BLOCK_E,
            NUM_BLOCK_D=NUM_BLOCK_D,
            NUM_BLOCK_E=NUM_BLOCK_E,
        )

    def grid(meta):
        return (
            b,
            h,
        )

    _compute_dld_cumsum_kernel[grid](
        DLD_Q=dld_q,
        DLD_K=dld_k,
        DLD=dld,
        FINAL_STATE=final_state,
        DFINAL_STATE=dfinal_state,
        DLD_STATE=dld_state,
        CU_SEQLENS=cu_seqlens,
        SUM_OPTION=sum_option,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        F=f,
        USE_FINAL_STATE=use_final_state,
        USE_CU_SEQLENS=use_cu_seqlens,
        BLOCK_D=BLOCK_D,
        BLOCK_E=BLOCK_E,
        BLOCK_F=BLOCK_F,
        NUM_BLOCK_D=NUM_BLOCK_D,
        NUM_BLOCK_E=NUM_BLOCK_E,
    )

    return dld


if __name__ == "__main__":
    device = torch.device("cuda")
    dld_q = torch.randn(1, 1024, 128, device=device)
    dld_k = torch.randn(1, 1024, 128, device=device)
    final_state = torch.randn(1, 128, 128, 128, device=device)
    dfinal_state = torch.randn(1, 128, 128, 128, device=device)
    dld = compute_dld_with_cumsum_triton(dld_q, dld_k, final_state, dfinal_state)
    print(dld.sum())
