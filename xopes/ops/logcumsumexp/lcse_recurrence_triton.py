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
            "BLOCK": [16, 32, 64],
        }
    ),
    key=["b", "n"],
)
@triton.jit
def _lcse_recurrence_fwd(
    X,  # B N
    O,  # B N
    STATE,  # B 1
    FINAL_STATE,  # B 1
    X_MIN,  # B 1
    B: tl.constexpr,
    N: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(0)
    # compute offset
    offset_b = off_b * BLOCK
    offset = offset_b * N
    offset_state = offset_b
    array = tl.arange(0, BLOCK)
    mask = (offset_b + array) < B

    x_block_ptr = X + offset + array * N
    o_block_ptr = O + offset + array * N
    final_state_block_ptr = FINAL_STATE + offset_state + array
    x_min_block_ptr = X_MIN + offset_state + array
    if USE_INITIAL_STATE:
        state_block_ptr = STATE + offset_state + array
        state = tl.load(state_block_ptr, mask=mask, other=-float("inf")).to(tl.float32)
        if SCALE != -1:
            state = tl.clamp(state, min=-SCALE, max=SCALE)
        x_max = state
        state = state - x_max  # !!! important
        x_min = state
    else:
        state = tl.full([BLOCK], -float("inf"), dtype=tl.float32)
        x_max = tl.full([BLOCK], -float("inf"), dtype=tl.float32)
        x_min = tl.full([BLOCK], float("inf"), dtype=tl.float32)

    o = tl.zeros([BLOCK], dtype=tl.float32)

    for i in range(N):
        x = tl.load(x_block_ptr, mask=mask, other=-float("inf")).to(tl.float32)
        if SCALE != -1:
            x = tl.clamp(x, min=-SCALE, max=SCALE)
        # x_min_ = tl.min(tl.where(mask, x, 0), keep_dims=True)
        # x_min = tl.minimum(x_min, x_min_)
        # x_max_ = tl.max(x, keep_dims=True)
        # x_max_ = tl.maximum(x_max, x_max_)

        x_min_ = tl.where(mask, x, 0)
        x_min = tl.minimum(x_min, x_min_)
        x_max_ = tl.maximum(x_max, x)

        state = tl.log(tl.exp(state + x_max - x_max_) + tl.exp(x - x_max_))
        o = state + x_max_
        x_max = x_max_

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), mask=mask)

        x_block_ptr += 1
        o_block_ptr += 1

    tl.store(
        final_state_block_ptr, o.to(final_state_block_ptr.dtype.element_ty), mask=mask
    )
    tl.store(x_min_block_ptr, x_min.to(x_min_block_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    generate_configs(
        {
            "num_warps": [2, 4, 8, 16, 32],
            "num_stages": [2, 3, 4],
            "BLOCK": [16, 32, 64],
        }
    ),
    key=["b", "n"],
)
@triton.jit
def _lcse_recurrence_bwd(
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
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(0)
    # compute offset
    offset_b = off_b * BLOCK
    offset = offset_b * N + N
    offset_state = offset_b
    array = tl.arange(0, BLOCK)
    mask = (offset_b + array) < B

    x_block_ptr = X + offset + array * N
    o_block_ptr = O + offset + array * N
    dx_block_ptr = DX + offset + array * N
    do_block_ptr = DO + offset + array * N
    x_min_block_ptr = X_MIN + offset_state + array

    if USE_DFINAL_STATE:
        dstate_block_ptr = DSTATE + offset_state + array
        dstate = tl.load(dstate_block_ptr, mask=mask, other=0).to(tl.float32)
    else:
        dstate = tl.zeros([BLOCK], dtype=tl.float32)

    x_min = tl.load(x_min_block_ptr, mask=mask, other=0).to(tl.float32)
    dx = tl.zeros([BLOCK], dtype=tl.float32)

    for i in range(N):
        x_block_ptr -= 1
        o_block_ptr -= 1
        dx_block_ptr -= 1
        do_block_ptr -= 1

        x = tl.load(x_block_ptr, mask=mask, other=-float("inf")).to(tl.float32)
        if SCALE != -1:
            flag = (x >= -SCALE) and (x <= SCALE)
            x = tl.clamp(x, min=-SCALE, max=SCALE)
        o = tl.load(o_block_ptr, mask=mask, other=0).to(tl.float32)
        do = tl.load(do_block_ptr, mask=mask, other=0).to(tl.float32)
        if i == 0 and USE_DFINAL_STATE:
            do += dstate

        dz = do * tl.exp(x_min - o)
        dx += dz

        # https://github.com/pytorch/pytorch/blob/53fe804322640653d2dddaed394838b868ce9a26/torch/autograd/_functions/pointwise.py#L95
        dx_ = dx * tl.exp(x - x_min)
        if SCALE != -1:
            dx_ = tl.where(flag, dx_, 0)
        tl.store(dx_block_ptr, dx_.to(dx_block_ptr.dtype.element_ty), mask=mask)

    if USE_INITIAL_STATE:
        state_block_ptr = STATE + offset_state + array
        dinitial_state_block_ptr = DINITIAL_STATE + offset_state + array
        state = tl.load(state_block_ptr, mask=mask, other=-float("inf")).to(tl.float32)
        if SCALE != -1:
            flag = (state >= -SCALE) and (state <= SCALE)
            state = tl.clamp(state, min=-SCALE, max=SCALE)
        dstate = dx * tl.exp(state - x_min)
        if SCALE != -1:
            dstate = tl.where(flag, dstate, 0)
        tl.store(
            dinitial_state_block_ptr,
            dstate.to(dinitial_state_block_ptr.dtype.element_ty),
            mask=mask,
        )


class LcseRecurrence(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, initial_state=None, scale=-1):
        b, n = x.shape
        MAX_BLOCK = triton.next_power_of_2(b)

        def grid(meta):
            BLOCK = min(meta["BLOCK"], MAX_BLOCK)
            return (triton.cdiv(b, BLOCK),)

        o = torch.empty_like(x)
        use_initial_state = initial_state is not None
        final_state = torch.empty(b, 1, dtype=x.dtype, device=x.device)
        x_min = torch.empty(b, 1, dtype=x.dtype, device=x.device)

        _lcse_recurrence_fwd[grid](
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
        MAX_BLOCK = triton.next_power_of_2(b)

        use_initial_state = initial_state is not None
        use_dfinal_state = dfinal_state is not None
        if use_initial_state:
            dinitial_state = torch.empty(b, 1, device=x.device, dtype=x.dtype)
        else:
            dinitial_state = None

        dx = torch.empty_like(x)

        def grid(meta):
            BLOCK = min(meta["BLOCK"], MAX_BLOCK)
            return (triton.cdiv(b, BLOCK),)

        _lcse_recurrence_bwd[grid](
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


def lcse_recurrence_triton(
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

    o, state = LcseRecurrence.apply(x, initial_state, scale)

    o = o.reshape(shape)
    state = state.reshape(shape[:-1] + [1])

    if dim != -1:
        o = o.transpose(dim, -1)
        state = state.transpose(dim, -1)

    return o, state


if __name__ == "__main__":
    x = torch.randn(10, 10).cuda().requires_grad_()
    initial_state = torch.randn(1).cuda().requires_grad_()
    o, state = lcse_recurrence_triton(x, initial_state=initial_state)
    print(o.mean())
    print(state.mean())
    res = o.sum() + state.sum()
    res.backward()
    print(x.grad.mean())
