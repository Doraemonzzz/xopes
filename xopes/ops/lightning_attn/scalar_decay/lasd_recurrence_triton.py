from typing import Optional

import torch
import triton
import triton.language as tl
from einops import repeat

from xopes.utils import contiguous, generate_configs


@triton.jit
def _apply_activation(X, ACT: tl.constexpr):
    if ACT != "none":
        if ACT == "relu":
            X = tl.where(X >= 0, X, 0)
        elif ACT == "sigmoid":
            X = tl.sigmoid(X)
        elif ACT == "silu":
            X = X * tl.sigmoid(X)
        elif ACT == "softmax":
            X_max = tl.max(X, axis=-1)
            X_minus_max = X - X_max
            # softmax
            numerator = tl.exp(X_minus_max)
            denominator = tl.sum(numerator, axis=-1, keep_dims=True)
            X = numerator / denominator
    return X


@triton.autotune(
    generate_configs(
        {
            "num_warps": [4, 8, 16, 32],
        }
    ),
    key=["B", "H", "D", "E"],
)
@triton.jit
def _lasd_recurrence_fwd(
    Q,  # B N H D
    K,  # B N H D
    V,  # B N H E
    STATE,  # B H D E
    CU_SEQLENS,  # M
    O,  # B N H E
    LOG_DECAY,  # H
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    q_act: tl.constexpr,
    k_act: tl.constexpr,
    v_act: tl.constexpr,
    q_norm: tl.constexpr,
    k_norm: tl.constexpr,
    v_norm: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    # compute offset
    if not USE_CU_SEQLENS:
        offset_qk = off_b * N * H * D + off_h * D
        offset_vo = off_b * N * H * E + off_h * E
        offset_state = off_b * H * D * E + off_h * D * E
    else:
        start = tl.load(CU_SEQLENS + off_b)
        end = tl.load(CU_SEQLENS + off_b + 1)
        N = end - start
        offset_qk = start * H * D + off_h * D
        offset_vo = start * H * E + off_h * E
        offset_state = off_b * H * D * E + off_h * D * E

    # compute block ptr
    array_d = tl.arange(0, D)
    array_e = tl.arange(0, E)
    q_block_ptr = Q + offset_qk + array_d
    k_block_ptr = K + offset_qk + array_d
    v_block_ptr = V + offset_vo + array_e
    o_block_ptr = O + offset_vo + array_e
    state_block_ptr = STATE + offset_state + array_d[:, None] * E + array_e[None, :]
    log_decay_block_ptr = LOG_DECAY + off_h

    # compute
    state = tl.load(state_block_ptr).to(tl.float32)  # D E
    log_decay = tl.load(log_decay_block_ptr).to(tl.float32)
    decay = tl.exp(-log_decay)
    for i in range(N):
        # load
        q = tl.load(q_block_ptr)
        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)
        q = _apply_activation(q, q_act)
        k = _apply_activation(k, k_act)
        v = _apply_activation(v, v_act)
        state = decay * state + k[:, None] * v[None, :]
        o = tl.sum(q[:, None] * state, axis=0)

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty))

        # update
        q_block_ptr += D
        k_block_ptr += D
        v_block_ptr += E
        o_block_ptr += E

    tl.store(state_block_ptr, state.to(state_block_ptr.dtype.element_ty))


def lasd_recurrence_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    q_act: str = "none",
    k_act: str = "none",
    v_act: str = "none",
    q_norm: bool = False,
    k_norm: bool = False,
    v_norm: bool = False,
):
    b, n, h, d = q.shape
    e = v.shape[-1]

    use_cu_seqlens = cu_seqlens is not None
    if use_cu_seqlens:
        b = len(cu_seqlens) - 1

    if initial_state is None:
        state = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)
    else:
        state = initial_state
        if len(state.shape) == 3:
            state = repeat(state, "h d e -> b h d e", b=b)

    def grid(meta):
        return (b, h)

    if initial_state is None:
        state = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)
    else:
        state = initial_state

    if use_cu_seqlens:
        o = torch.empty((1, cu_seqlens[-1], h, e), dtype=q.dtype, device=q.device)
    else:
        o = torch.empty((b, n, h, e), dtype=q.dtype, device=q.device)

    _lasd_recurrence_fwd[grid](
        Q=q,
        K=k,
        V=v,
        STATE=state,
        CU_SEQLENS=cu_seqlens,
        O=o,
        LOG_DECAY=ld,
        B=b,
        N=n,
        H=h,
        D=d,
        E=e,
        q_act=q_act,
        k_act=k_act,
        v_act=v_act,
        q_norm=q_norm,
        k_norm=k_norm,
        v_norm=v_norm,
        USE_CU_SEQLENS=use_cu_seqlens,
    )

    return o, state


class LasdRecurrenceFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        ld,
        initial_state=None,
        cu_seqlens=None,
        q_act="none",
        k_act="none",
        v_act="none",
        q_norm=False,
        k_norm=False,
        v_norm=False,
    ):
        # Save non-tensor inputs for backward
        ctx.q_act = q_act
        ctx.k_act = k_act
        ctx.v_act = v_act
        ctx.q_norm = q_norm
        ctx.k_norm = k_norm
        ctx.v_norm = v_norm

        # Forward computation
        output, final_state = lasd_recurrence_fwd(
            q=q,
            k=k,
            v=v,
            ld=ld,
            initial_state=initial_state,
            q_act=q_act,
            k_act=k_act,
            v_act=v_act,
            q_norm=q_norm,
            k_norm=k_norm,
            v_norm=v_norm,
        )

        # Save tensors needed for backward
        ctx.save_for_backward(q, k, v, ld, final_state)

        return output, final_state

    @staticmethod
    def backward(ctx, grad_output, grad_state):
        pass


def lasd_recurrence_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    q_act: str = "none",
    k_act: str = "none",
    v_act: str = "none",
    q_norm: bool = False,
    k_norm: bool = False,
    v_norm: bool = False,
):
    """
    Apply Lightning Attention with Scalar Decay in Triton.

    Args:
        q: Query tensor of shape (B, N, H, D)
        k: Key tensor of shape (B, N, H, D)
        v: Value tensor of shape (B, N, H, E)
        ld: Logarithmic decay tensor of shape (H,)
        initial_state: Initial state tensor of shape (B, H, D, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training
        q_act: Activation function for query
        k_act: Activation function for key
        v_act: Activation function for value
        q_norm: Normalize query
        k_norm: Normalize key
        v_norm: Normalize value

    Returns:
        output: Tensor of shape (B, N, H, E)
        state: Tensor of shape (B, H, D, E)
    """
    return LasdRecurrenceFunction.apply(
        q,
        k,
        v,
        ld,
        initial_state,
        cu_seqlens,
        q_act,
        k_act,
        v_act,
        q_norm,
        k_norm,
        v_norm,
    )


if __name__ == "__main__":
    import torch.nn.functional as F

    b, n, h, d = 2, 16, 12, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    q = torch.randn(b, n, h, d, device=device, dtype=dtype)
    k = torch.randn(b, n, h, d, device=device, dtype=dtype)
    v = torch.randn(b, n, h, d, device=device, dtype=dtype)
    ld = F.logsigmoid(torch.randn(h, device=device))
    output, state = lasd_recurrence_triton(q, k, v, ld)
