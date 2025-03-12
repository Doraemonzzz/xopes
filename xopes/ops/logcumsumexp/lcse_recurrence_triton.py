from typing import Optional

import torch
import triton
import triton.language as tl
from einops import repeat

from xopes.utils import contiguous, generate_configs


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8, 16, 32], "num_stages": [2, 3, 4]}),
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
):
    off_b = tl.program_id(0)
    # compute offset
    offset = off_b * N
    offset_state = off_b

    x_block_ptr = X + offset + tl.arange(0, 1)
    o_block_ptr = O + offset + tl.arange(0, 1)
    final_state_block_ptr = FINAL_STATE + offset_state + tl.arange(0, 1)
    x_min_block_ptr = X_MIN + offset_state + tl.arange(0, 1)
    if USE_INITIAL_STATE:
        state_block_ptr = STATE + offset_state + tl.arange(0, 1)
        state = tl.load(state_block_ptr).to(tl.float32)
        m = state
        state = state - m  # !!! important
        x_min = state
    else:
        state = tl.full([1], -float("inf"), dtype=tl.float32)
        m = state
        x_min = tl.full([1], float("inf"), dtype=tl.float32)
    o = tl.zeros([1], dtype=tl.float32)

    for i in range(N):
        x = tl.load(x_block_ptr).to(tl.float32)
        x_min = tl.minimum(x_min, x)
        m_ = tl.maximum(x, m)

        state = tl.log(tl.exp(state + m - m_) + tl.exp(x - m_))
        o = state + m_
        m = m_

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty))

        x_block_ptr += 1
        o_block_ptr += 1

    tl.store(final_state_block_ptr, o.to(final_state_block_ptr.dtype.element_ty))
    tl.store(x_min_block_ptr, x_min.to(x_min_block_ptr.dtype.element_ty))


@triton.autotune(
    generate_configs({"num_warps": [2, 4, 8, 16, 32], "num_stages": [2, 3, 4]}),
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
):
    off_b = tl.program_id(0)
    # compute offset
    offset = off_b * N + N
    offset_state = off_b

    x_block_ptr = X + offset + tl.arange(0, 1)
    o_block_ptr = O + offset + tl.arange(0, 1)
    dx_block_ptr = DX + offset + tl.arange(0, 1)
    do_block_ptr = DO + offset + tl.arange(0, 1)
    x_min_block_ptr = X_MIN + offset_state + tl.arange(0, 1)

    if USE_DFINAL_STATE:
        dstate_block_ptr = DSTATE + offset_state + tl.arange(0, 1)
        dstate = tl.load(dstate_block_ptr).to(tl.float32)
    else:
        dstate = tl.zeros([1], dtype=tl.float32)

    x_min = tl.load(x_min_block_ptr).to(tl.float32)
    dx = tl.zeros([1], dtype=tl.float32)

    for i in range(N):
        x_block_ptr -= 1
        o_block_ptr -= 1
        dx_block_ptr -= 1
        do_block_ptr -= 1

        x = tl.load(x_block_ptr).to(tl.float32)
        o = tl.load(o_block_ptr).to(tl.float32)
        do = tl.load(do_block_ptr).to(tl.float32)
        if i == 0:
            do += dstate

        dz = do * tl.exp(x_min - o)
        dx += dz

        dx_ = dx * tl.exp(x - x_min)
        tl.store(dx_block_ptr, dx_.to(dx_block_ptr.dtype.element_ty))

    if USE_INITIAL_STATE:
        state_block_ptr = STATE + offset_state + tl.arange(0, 1)
        dinitial_state_block_ptr = DINITIAL_STATE + offset_state + tl.arange(0, 1)
        state = tl.load(state_block_ptr).to(tl.float32)
        dstate = dx * tl.exp(state - x_min)
        tl.store(
            dinitial_state_block_ptr,
            dstate.to(dinitial_state_block_ptr.dtype.element_ty),
        )


class LcseRecurrence(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, x, initial_state=None):
        b, n = x.shape

        def grid(meta):
            return (b, n)

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
        )

        ctx.save_for_backward(x, o, initial_state, x_min)

        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dfinal_state):
        x, o, initial_state, x_min = ctx.saved_tensors
        b, n = x.shape

        use_initial_state = initial_state is not None
        use_dfinal_state = dfinal_state is not None
        if use_initial_state:
            dinitial_state = torch.empty(b, 1, device=x.device, dtype=x.dtype)
        else:
            dinitial_state = None

        dx = torch.empty_like(x)

        def grid(meta):
            return (b, n)

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
        )

        return dx, dinitial_state


def lcse_recurrence_triton(
    x: torch.Tensor,
    dim: int = -1,
    initial_state: Optional[torch.Tensor] = None,
):
    """
    Apply logcumsumexp on the dim dimension of x.

    Args:
        x: Input tensor of shape (...)
        dim: Dimension to apply the operation on
        initial_state: Initial state, the same shape as x, except the dim dimension, which is 1

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

    o, state = LcseRecurrence.apply(x, initial_state)

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
