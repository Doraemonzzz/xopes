from typing import Optional

import torch
import triton
import triton.language as tl

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
        }
    ),
    key=["B", "D", "E"],
)
@triton.jit
def _oplr_ddd_fwd(
    XK,  # B N D
    XV,  # B N E
    LOG_DECAY,  # B N D or None
    O,  # B N D E
    HAS_LOG_DECAY: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_d = tl.program_id(1)
    # compute offset
    offset_xk = off_b * N * D + off_d
    offset_xv = off_b * N * E
    offset_o = off_b * N * D * E + off_d * E
    # compute block ptr
    array = tl.arange(0, BLOCK_E)
    xk_block_ptr = XK + offset_xk
    xv_block_ptr = XV + offset_xv + array
    o_block_ptr = O + offset_o + array
    if HAS_LOG_DECAY:
        log_decay_block_ptr = LOG_DECAY + offset_xk
    mask = array < E

    # compute
    o = tl.zeros([BLOCK_E], dtype=tl.float32)
    for i in range(N):
        # load
        xk = tl.load(xk_block_ptr)
        xv = tl.load(xv_block_ptr, mask=mask, other=0)
        if HAS_LOG_DECAY:
            log_decay = tl.load(log_decay_block_ptr).to(tl.float32)
            decay = tl.exp(log_decay)
            log_decay_block_ptr += D
        else:
            decay = 1 - xk

        o = decay * o + xk * xv

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask)

        # update
        xk_block_ptr += D
        xv_block_ptr += E
        o_block_ptr += D * E


@triton.autotune(
    generate_configs(
        {
            "num_warps": [1, 2, 4, 8, 16, 32],
        }
    ),
    key=["D", "E"],
)
@triton.jit
def _oplr_ddd_bwd(
    XK,  # B N D
    XV,  # B N E
    LOG_DECAY,  # B N D or None
    DO,  # B N D E
    DXK,  # B N D
    DXV,  # B N E
    DBETA,  # B N D
    DLOG_DECAY,  # B N D or None
    HAS_LOG_DECAY: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    off_b = tl.program_id(0)
    # compute offset
    offset_xk = off_b * N * D
    offset_xv = off_b * N * E
    offset_do = off_b * N * D * E + N * D * E
    # compute block ptr
    array_d = tl.arange(0, BLOCK_D) + N * D
    array_e = tl.arange(0, BLOCK_E) + N * E
    xk_block_ptr = XK + offset_xk + array_d
    xv_block_ptr = XV + offset_xv + array_e
    do_block_ptr = (
        DO
        + offset_do
        + tl.arange(0, BLOCK_D)[:, None] * E
        + tl.arange(0, BLOCK_E)[None, :]
    )
    dxk_block_ptr = DXK + offset_xk + array_d
    dxv_block_ptr = DXV + offset_xv + array_e
    dbeta_block_ptr = DBETA + offset_xk + array_d
    if HAS_LOG_DECAY:
        log_decay_block_ptr = LOG_DECAY + offset_xk + array_d

    dkv = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)
    for i in range(N):
        array_d -= D
        array_e -= E
        xk_block_ptr -= D
        xv_block_ptr -= E
        do_block_ptr -= D * E
        dxk_block_ptr -= D
        dxv_block_ptr -= E
        dbeta_block_ptr -= D
        mask_d = array_d >= 0 and array_d < D
        mask_e = array_e >= 0 and array_e < E
        # load
        xk = tl.load(xk_block_ptr, mask=mask_d, other=0).to(tl.float32)
        xv = tl.load(xv_block_ptr, mask=mask_e, other=0)
        do = tl.load(do_block_ptr, mask=mask_d[:, None] & mask_e[None, :], other=0)

        if i == 0:
            decay = tl.zeros([BLOCK_D], dtype=tl.float32)
        else:
            if HAS_LOG_DECAY:
                log_decay_block_ptr -= D
                log_decay = tl.load(log_decay_block_ptr, mask=mask_d, other=0).to(
                    tl.float32
                )
                decay = tl.exp(log_decay)
            else:
                decay = 1 - xk

        dkv = decay[:, None] * dkv + do
        dxk = tl.sum(dkv * xv[None, :], axis=1)
        dxv = tl.sum(dkv * xk[:, None], axis=0)
        dbeta = xk * tl.sum(do * xv[None, :], axis=1)

        tl.store(dxk_block_ptr, dxk.to(dxk_block_ptr.dtype.element_ty), mask=mask_d)
        tl.store(dxv_block_ptr, dxv.to(dxv_block_ptr.dtype.element_ty), mask=mask_e)
        tl.store(
            dbeta_block_ptr, dbeta.to(dbeta_block_ptr.dtype.element_ty), mask=mask_d
        )


class OplrDddTriton(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, xk, xv, log_decay=None):
        if log_decay is None:
            assert torch.all(xk <= 1), "xk must be all negative when decay is None"

        # Ensure inputs are contiguous and shapes are correct
        b, n, d = xk.shape
        e = xv.shape[-1]

        # Allocate output memory
        o = torch.empty((b, n, d, e), dtype=xk.dtype, device=xk.device)

        has_log_decay = log_decay is not None
        MAX_BLOCK_SIZE = 65536
        BLOCK_E = min(triton.next_power_of_2(e), MAX_BLOCK_SIZE)

        # Launch kernel
        grid = (b, d)
        _oplr_ddd_fwd[grid](
            XK=xk,
            XV=xv,
            LOG_DECAY=log_decay,
            O=o,
            HAS_LOG_DECAY=has_log_decay,
            B=b,
            N=n,
            D=d,
            E=e,
            BLOCK_E=BLOCK_E,
        )

        ctx.save_for_backward(xk, xv, log_decay)

        return o

    @staticmethod
    @contiguous
    def backward(ctx, do):
        xk, xv, log_decay = ctx.saved_tensors

        # Ensure inputs are contiguous and shapes are correct
        b, n, d = xk.shape
        e = xv.shape[-1]

        # Allocate gradient tensors
        dxk = torch.empty_like(xk)
        dxv = torch.empty_like(xv)
        dbeta = torch.empty_like(xk)

        has_log_decay = log_decay is not None
        if has_log_decay:
            dlog_decay = torch.empty_like(log_decay)
        else:
            dlog_decay = None

        b, n, d = xk.shape
        e = xv.shape[-1]

        MAX_BLOCK_SIZE = 65536
        BLOCK_D = min(triton.next_power_of_2(d), MAX_BLOCK_SIZE)
        BLOCK_E = min(triton.next_power_of_2(e), MAX_BLOCK_SIZE)
        print("aaa", do.shape)
        grid = (b,)
        _oplr_ddd_bwd[grid](
            XK=xk,
            XV=xv,
            LOG_DECAY=log_decay,
            DO=do,
            DXK=dxk,
            DXV=dxv,
            DLOG_DECAY=dlog_decay,
            DBETA=dbeta,
            HAS_LOG_DECAY=has_log_decay,
            B=b,
            N=n,
            D=d,
            E=e,
            BLOCK_D=BLOCK_D,
            BLOCK_E=BLOCK_E,
        )

        dbeta -= xk * dxk
        dlog_decay_ = torch.flip(torch.cumsum(dbeta, dim=1), dims=[1])
        if has_log_decay:
            dlog_decay = dlog_decay_
        else:
            dlog_decay = None
            dxk -= (1 - xk) * dlog_decay_

        return dxk, dxv, dlog_decay


def oplr_ddd_triton(
    xk: torch.Tensor,
    xv: torch.Tensor,
    log_decay: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies Out Product Linear Recurrence with data-dependent decay using Triton.

    Args:
        xk: Input tensor (B, N, D)
        xv: Expansion vector (B, N, E)
        log_decay: Optional decay tensor (B, N, D)

    Returns:
        Output tensor (B, N, D, E)
    """
    return OplrDddTriton.apply(xk, xv, log_decay)


if __name__ == "__main__":
    import torch.nn.functional as F

    b, n, d, e = 2, 512, 128, 128
    dtype = torch.bfloat16
    xv = torch.randn((b, n, e), dtype=dtype).cuda().requires_grad_(True)
    xk = torch.randn((b, n, d), dtype=dtype).cuda().requires_grad_(True)
    log_decay = F.logsigmoid(torch.randn((b, n, d), dtype=dtype).cuda()).requires_grad_(
        True
    )
    o = oplr_ddd_triton(xk, xv, log_decay)
    print(o.shape)

    o.sum().backward()
