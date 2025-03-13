from typing import Optional

import torch
import triton
import triton.language as tl
from einops import repeat

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs(
        {
            "num_warps": [2, 4, 8, 16, 32],
            "num_stages": [2, 3, 4],
            "BLOCK_B": [16, 32, 64],
            "BLOCK_N": [128, 256, 512],
        }
    ),
    key=["b", "n"],
)
@triton.jit
def _lcse_parallel_fwd(
    X,  # B N
    O,  # B N
    STATE,  # B 1
    FINAL_STATE,  # B 1
    X_MIN,  # B 1
    B: tl.constexpr,
    N: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    NUM_BLOCK_N = triton.cdiv(N, BLOCK_N)
    # compute offset
    offset_b = off_b * BLOCK_B
    offset = offset_b * N
    offset_state = offset_b
    array_b = tl.arange(0, BLOCK_B)
    array_n = tl.arange(0, BLOCK_N)
    mask_b = (offset_b + array_b) < B

    x_block_ptr = X + offset + array_b[:, None] * N + array_n[None, :]
    o_block_ptr = O + offset + array_b[:, None] * N + array_n[None, :]
    final_state_block_ptr = FINAL_STATE + offset_state + array_b[:, None]
    x_min_block_ptr = X_MIN + offset_state + array_b[:, None]
    if USE_INITIAL_STATE:
        state_block_ptr = STATE + offset_state + array_b[:, None]
        state = tl.load(state_block_ptr, mask=mask_b[:, None], other=-float("inf")).to(
            tl.float32
        )
        if SCALE != -1:
            state = tl.clamp(state, min=-SCALE, max=SCALE)
        x_max = tl.max(state, axis=-1, keep_dims=True)  # B 1
        state = state - x_max  # !!! important
        x_min = tl.min(state, axis=-1, keep_dims=True)  # B 1
    else:
        state = tl.full([BLOCK_B, 1], -float("inf"), dtype=tl.float32)
        x_max = tl.full([BLOCK_B, 1], -float("inf"), dtype=tl.float32)
        x_min = tl.full([BLOCK_B, 1], float("inf"), dtype=tl.float32)
    o = tl.zeros([BLOCK_B, BLOCK_N], dtype=tl.float32)

    for i in range(NUM_BLOCK_N):
        mask_n = array_n < N
        mask = mask_b[:, None] & mask_n[None, :]
        x = tl.load(x_block_ptr, mask=mask, other=-float("inf")).to(tl.float32)
        if SCALE != -1:
            x = tl.clamp(x, min=-SCALE, max=SCALE)
        x_min_ = tl.min(tl.where(mask, x, 0), axis=-1, keep_dims=True)
        x_min = tl.minimum(x_min, x_min_)
        x_max_ = tl.max(x, axis=-1, keep_dims=True)
        x_max_ = tl.maximum(x_max, x_max_)  # B 1

        # B N
        o = tl.exp(state + x_max - x_max_) + tl.cumsum(tl.exp(x - x_max_), axis=-1)
        o = tl.log(o) + x_max_
        state = tl.log(
            tl.exp(state + x_max - x_max_)
            + tl.sum(tl.exp(x - x_max_), axis=-1, keep_dims=True)
        )
        x_max = x_max_

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask)

        x_block_ptr += BLOCK_N
        o_block_ptr += BLOCK_N
        array_n += BLOCK_N

    final_state = state + x_max
    tl.store(
        final_state_block_ptr,
        final_state.to(final_state_block_ptr.dtype.element_ty),
        mask=mask_b[:, None],
    )
    tl.store(
        x_min_block_ptr,
        x_min.to(x_min_block_ptr.dtype.element_ty),
        mask=mask_b[:, None],
    )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [2, 4, 8, 16, 32],
            "num_stages": [2, 3, 4],
            "BLOCK_B": [16, 32, 64],
            "BLOCK_N": [128, 256, 512],
        }
    ),
    key=["b", "n"],
)
@triton.jit
def _lcse_parallel_bwd(
    X,  # B N
    O,  # B N
    DX,  # B N
    DO,  # B N
    STATE,  # B 1
    DSTATE,  # B 1
    DINITIAL_STATE,  # B 1
    X_MIN,  # B 1
    B: tl.constexpr,
    N: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_DFINAL_STATE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    NUM_BLOCK_N = triton.cdiv(N, BLOCK_N)
    # compute offset
    offset_b = off_b * BLOCK_B
    offset = offset_b * N
    offset_state = offset_b
    array_b = tl.arange(0, BLOCK_B)
    array_n = tl.arange(0, BLOCK_N) + N
    mask_b = (offset_b + array_b) < B

    x_block_ptr = X + offset + array_b[:, None] * N + array_n[None, :]
    o_block_ptr = O + offset + array_b[:, None] * N + array_n[None, :]
    dx_block_ptr = DX + offset + array_b[:, None] * N + array_n[None, :]
    do_block_ptr = DO + offset + array_b[:, None] * N + array_n[None, :]
    x_min_block_ptr = X_MIN + offset_state + array_b[:, None]

    if USE_DFINAL_STATE:
        dstate_block_ptr = DSTATE + offset_state + array_b[:, None]
        dstate = tl.load(dstate_block_ptr, mask=mask_b[:, None], other=0).to(tl.float32)
    else:
        dstate = tl.zeros([BLOCK_B, 1], dtype=tl.float32)

    x_min = tl.load(x_min_block_ptr, mask=mask_b[:, None], other=0).to(tl.float32)
    dx_cumsum = tl.zeros([BLOCK_B, 1], dtype=tl.float32)
    dx = tl.zeros([BLOCK_B, BLOCK_N], dtype=tl.float32)

    for i in range(NUM_BLOCK_N):
        array_n -= BLOCK_N
        x_block_ptr -= BLOCK_N
        o_block_ptr -= BLOCK_N
        dx_block_ptr -= BLOCK_N
        do_block_ptr -= BLOCK_N

        mask_n = array_n >= 0
        mask = mask_b[:, None] & mask_n[None, :]

        x = tl.load(x_block_ptr, mask=mask, other=-float("inf")).to(tl.float32)
        if SCALE != -1:
            flag = (x >= -SCALE) and (x <= SCALE)
            x = tl.clamp(x, min=-SCALE, max=SCALE)
        o = tl.load(o_block_ptr, mask=mask, other=0).to(tl.float32)
        do = tl.load(do_block_ptr, mask=mask, other=0).to(tl.float32)
        if i == 0:
            array = tl.arange(0, BLOCK_N)
            do_ = tl.where(array == BLOCK_N - 1, dstate, 0)
            do += do_

        # B N
        do_ = do * tl.exp(x_min - o)
        dz = tl.cumsum(do_, axis=-1)
        dx = dx_cumsum + dz
        dx_cumsum += tl.sum(do_, axis=-1, keep_dims=True)

        # https://github.com/pytorch/pytorch/blob/53fe804322640653d2dddaed394838b868ce9a26/torch/autograd/_functions/pointwise.py#L95
        dx_ = dx * tl.exp(x - x_min)
        if SCALE != -1:
            dx_ = tl.where(flag, dx_, 0)
        tl.store(dx_block_ptr, dx_.to(dx_block_ptr.dtype.element_ty), mask=mask)

    if USE_INITIAL_STATE:
        state_block_ptr = STATE + offset_state + array_b[:, None]
        dinitial_state_block_ptr = DINITIAL_STATE + offset_state + array_b[:, None]
        state = tl.load(state_block_ptr, mask=mask_b[:, None], other=-float("inf")).to(
            tl.float32
        )
        if SCALE != -1:
            flag = (state >= -SCALE) and (state <= SCALE)
            state = tl.clamp(state, min=-SCALE, max=SCALE)
        dstate = dx_cumsum * tl.exp(state - x_min)
        if SCALE != -1:
            dstate = tl.where(flag, dstate, 0)
        tl.store(
            dinitial_state_block_ptr,
            dstate.to(dinitial_state_block_ptr.dtype.element_ty),
            mask=mask_b[:, None],
        )


class Lcseparallel(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, initial_state=None, scale=-1):
        b, n = x.shape
        MAX_BLOCK_B = triton.next_power_of_2(b)
        MAX_BLOCK_N = triton.next_power_of_2(n)

        def grid(meta):
            BLOCK_B = min(meta["BLOCK_B"], MAX_BLOCK_B)
            BLOCK_N = min(meta["BLOCK_N"], MAX_BLOCK_N)
            return (triton.cdiv(b, BLOCK_B), triton.cdiv(n, BLOCK_N))

        o = torch.empty_like(x)
        use_initial_state = initial_state is not None
        final_state = torch.empty(b, 1, dtype=x.dtype, device=x.device)
        x_min = torch.empty(b, 1, dtype=x.dtype, device=x.device)

        _lcse_parallel_fwd[grid](
            X=x,
            O=o,
            STATE=initial_state,
            FINAL_STATE=final_state,
            B=b,
            N=n,
            USE_INITIAL_STATE=use_initial_state,
            X_MIN=x_min,
            SCALE=scale,
        )

        ctx.save_for_backward(x, o, initial_state, x_min)
        ctx.scale = scale

        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dfinal_state):
        x, o, initial_state, x_min = ctx.saved_tensors
        scale = ctx.scale
        b, n = x.shape
        MAX_BLOCK_B = triton.next_power_of_2(b)
        MAX_BLOCK_N = triton.next_power_of_2(n)

        use_initial_state = initial_state is not None
        use_dfinal_state = dfinal_state is not None
        if use_initial_state:
            dinitial_state = torch.empty(b, 1, device=x.device, dtype=x.dtype)
        else:
            dinitial_state = None

        dx = torch.empty_like(x)

        def grid(meta):
            BLOCK_B = min(meta["BLOCK_B"], MAX_BLOCK_B)
            BLOCK_N = min(meta["BLOCK_N"], MAX_BLOCK_N)
            return (triton.cdiv(b, BLOCK_B), triton.cdiv(n, BLOCK_N))

        _lcse_parallel_bwd[grid](
            X=x,
            O=o,
            DX=dx,
            DO=do,
            STATE=initial_state,
            DSTATE=dfinal_state,
            DINITIAL_STATE=dinitial_state,
            X_MIN=x_min,
            B=b,
            N=n,
            USE_INITIAL_STATE=use_initial_state,
            USE_DFINAL_STATE=use_dfinal_state,
            SCALE=scale,
        )

        return dx, dinitial_state, None


def lcse_parallel_triton(
    x: torch.Tensor,
    dim: int = -1,
    initial_state: Optional[torch.Tensor] = None,
    scale: float = -1,
):
    """
    Apply logcumsumexp on the dim dimension of x.

    Args:
        x: Input tensor of shape (...)
        dim: Dimension to apply the operation on
        initial_state: Initial state, the same shape as x, except the dim dimension, which is 1
        scale: Clamp the input tensor to [-scale, scale]

    Returns:
        output: Tensor of shape (...)
    """
    if dim != -1:
        x = x.transpose(dim, -1)
        if initial_state is not None and len(initial_state.shape) > 1:
            initial_state = initial_state.transpose(dim, -1)

    # reshape input data into 2D tensor
    shape = list(x.shape)
    x = x.reshape(-1, x.shape[-1]).contiguous()
    if initial_state is not None and len(initial_state.shape) == 1:
        initial_state = repeat(initial_state, "n -> b n", b=x.shape[0])
    if initial_state is not None:
        initial_state = initial_state.reshape(-1, initial_state.shape[-1]).contiguous()

    o, state = Lcseparallel.apply(x, initial_state, scale)

    o = o.reshape(shape)
    state = state.reshape(shape[:-1] + [1])

    if dim != -1:
        o = o.transpose(dim, -1)
        state = state.transpose(dim, -1)

    return o, state


if __name__ == "__main__":
    x = torch.randn(10, 10).cuda().requires_grad_()
    initial_state = torch.randn(1).cuda().requires_grad_()
    o, state = lcse_parallel_triton(x, initial_state=initial_state)
    print(o.mean())
    print(state.mean())
    res = o.sum() + state.sum()
    res.backward()
    print(x.grad.mean())
