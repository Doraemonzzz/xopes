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
def _compute_dld_kernel(
    DLD_Q,  # B N H
    DLD_K,  # B N H
    DLD,  # B N H (output)
    FINAL_STATE,  # B H D E
    DFINAL_STATE,  # B H D E
    DLD_STATE,  # B H
    CU_SEQLENS,  # M
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NUM_BLOCK_D: tl.constexpr,
    NUM_BLOCK_E: tl.constexpr,
):
    # Calculate program ID and offsets
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_n = tl.program_id(2)

    offset_b = off_b * N * H
    offset_n = off_n * BLOCK_N * H
    offset_h = off_h

    # Calculate pointers for DLD_Q and DLD_K
    array = tl.arange(0, BLOCK_N)
    dld_q_block_ptr = DLD_Q + offset_b + offset_n + offset_h + array * H
    dld_k_block_ptr = DLD_K + offset_b + offset_n + offset_h + array * H
    dld_block_ptr = DLD + offset_b + offset_n + offset_h + array * H
    mask_n = (off_n * BLOCK_N + array) < N

    # Load values from DLD_Q and DLD_K
    dld_q = tl.load(dld_q_block_ptr, mask=mask_n, other=0.0).to(tl.float32)
    dld_k = tl.load(dld_k_block_ptr, mask=mask_n, other=0.0).to(tl.float32)
    dld = dld_q - dld_k

    # If using final_state, calculate additional term
    if USE_FINAL_STATE:
        # dld_state = tl.load(DLD_STATE + off_b * H + off_h, other=0.0).to(tl.float32)
        dld_state = tl.load(DLD_STATE + off_b * H + off_h).to(tl.float32)
        dld += dld_state

    # Store result
    tl.store(dld_block_ptr, dld.to(dld_block_ptr.dtype.element_ty), mask=mask_n)


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
    DLD_STATE,  # B H
    CU_SEQLENS,  # M
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
    dld_state = 0.0

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

            final_state_block = tl.load(final_state_block_ptr, mask=mask, other=0.0).to(
                tl.float32
            )
            dfinal_state_block = tl.load(
                dfinal_state_block_ptr, mask=mask, other=0.0
            ).to(tl.float32)

            dld_state += tl.sum(final_state_block * dfinal_state_block)

    tl.store(DLD_STATE + off_b * H + off_h, dld_state.to(DLD_STATE.dtype.element_ty))


@contiguous
def compute_dld_triton(
    dld_q: torch.Tensor,  # B N H
    dld_k: torch.Tensor,  # B N H
    final_state: Optional[torch.Tensor] = None,  # B H D E
    dfinal_state: Optional[torch.Tensor] = None,  # B H D E
    cu_seqlens: Optional[torch.Tensor] = None,  # M
):
    """
    Compute the derivative of the log decay using Triton.

    Args:
        dld_q: The derivative of the log decay with respect to the query of shape (B, N, H).
        dld_k: The derivative of the log decay with respect to the key of shape (B, N, H).
        final_state: The final state of the recurrence of shape (B, H, D, E).
        dfinal_state: The derivative of the final state of the recurrence of shape (B, H, D, E).
        cu_seqlens: The cumulative sequence lengths of the query of shape (M,).

    Returns:
        dld: The derivative of the log decay.
    """
    b, n, h = dld_q.shape

    # Create output tensor
    dld = torch.empty_like(dld_q)

    # Determine if using final_state
    use_final_state = final_state is not None and dfinal_state is not None

    if use_final_state:
        d, e = final_state.shape[-2], final_state.shape[-1]
        dld_state = torch.empty(
            (
                b,
                h,
            ),
            dtype=dld_q.dtype,
            device=dld_q.device,
        )
    else:
        d, e = 1, 1
        dld_state = None

    # Determine if using cu_seqlens
    use_cu_seqlens = cu_seqlens is not None

    BLOCK_D = min(128, triton.next_power_of_2(d))
    NUM_BLOCK_D = triton.cdiv(d, BLOCK_D)
    BLOCK_E = min(128, triton.next_power_of_2(e))
    NUM_BLOCK_E = triton.cdiv(e, BLOCK_E)

    if use_final_state:
        grid = (b, h)

        _compute_dld_state[grid](
            FINAL_STATE=final_state,
            DFINAL_STATE=dfinal_state,
            DLD_STATE=dld_state,
            CU_SEQLENS=cu_seqlens,
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
        return (b, h, triton.cdiv(n, meta["BLOCK_N"]))

    _compute_dld_kernel[grid](
        DLD_Q=dld_q,
        DLD_K=dld_k,
        DLD=dld,
        FINAL_STATE=final_state,
        DFINAL_STATE=dfinal_state,
        DLD_STATE=dld_state,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        USE_FINAL_STATE=use_final_state,
        USE_CU_SEQLENS=use_cu_seqlens,
        BLOCK_D=BLOCK_D,
        BLOCK_E=BLOCK_E,
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
    dld = compute_dld_triton(dld_q, dld_k, final_state, dfinal_state)
    print(dld.sum())
