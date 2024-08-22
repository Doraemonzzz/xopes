import torch
import triton
import triton.language as tl

from xopes.utils import contiguous

# HEAD_DIM = 64


def _get_fw_configs():
    return None


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_D': 32}, num_warps=1),
#         triton.Config({'BLOCK_D': 32}, num_warps=2),
#         triton.Config({'BLOCK_D': 32}, num_warps=4),
#         triton.Config({'BLOCK_D': 32}, num_warps=8),
#         triton.Config({'BLOCK_D': 64}, num_warps=1),
#         triton.Config({'BLOCK_D': 64}, num_warps=2),
#         triton.Config({'BLOCK_D': 64}, num_warps=4),
#         triton.Config({'BLOCK_D': 64}, num_warps=8),
#         triton.Config({'BLOCK_D': 128}, num_warps=1),
#         triton.Config({'BLOCK_D': 128}, num_warps=2),
#         triton.Config({'BLOCK_D': 128}, num_warps=4),
#         triton.Config({'BLOCK_D': 128}, num_warps=8),
#     ],
#     key=['d']
# )
@triton.jit
def _grpe_recurrence_fwd(
    Q,
    K,
    V,
    M,
    O,
    S_INITIAL_STATE,
    S_FINAL_STATE,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    OUTPUT_FINAL_STATE: tl.constexpr,  # whether to output final state
):
    off_bh = tl.program_id(0)
    # compute offset
    off_qk = off_bh * n * d
    off_ov = off_bh * n * e
    off_m = off_bh * n * d * d
    off_s = off_bh * d * e

    # compute block ptr
    q_trans_block_ptr = Q + off_qk + tl.arange(0, d)[:, None]
    k_trans_block_ptr = K + off_qk + tl.arange(0, d)[:, None]
    v_block_ptr = V + off_ov + tl.arange(0, e)[None, :]
    m_block_ptr = M + off_m + tl.arange(0, d)[:, None] * d + tl.arange(0, d)[None, :]
    o_block_ptr = O + off_ov + tl.arange(0, e)[None, :]

    if USE_INITIAL_STATE:
        s_block_ptr = (
            S_INITIAL_STATE
            + off_s
            + tl.arange(0, d)[:, None]
            + tl.arange(0, e)[None, :]
        )

        s = tl.load(s_block_ptr).to(tl.float32)
    else:
        s = tl.zeros([d, e], dtype=tl.float32)

    for i in range(n):
        q_trans = tl.load(q_trans_block_ptr).to(tl.float32)
        k_trans = tl.load(k_trans_block_ptr).to(tl.float32)
        v = tl.load(v_block_ptr).to(tl.float32)
        m = tl.load(m_block_ptr).to(tl.float32)

        s = tl.dot(m, s) + k_trans.to(v.dtype) * v
        o = q_trans * s
        # d e -> 1 e
        o = tl.sum(o, axis=0)[None, :]

        tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty))

        q_trans_block_ptr += d
        k_trans_block_ptr += d
        v_block_ptr += e
        m_block_ptr += d * d
        o_block_ptr += e

    if OUTPUT_FINAL_STATE:
        s_final_block_ptr = (
            S_FINAL_STATE
            + off_s
            + tl.arange(0, d)[:, None] * d
            + tl.arange(0, e)[None, :]
        )

        tl.store(
            s_final_block_ptr,
            s.to(s_final_block_ptr.dtype.element_ty),
        )


class GrpeRecurrenceFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx, q, k, v, alpha, beta, gamma, initial_state=None, output_final_state=None
    ):
        b, h, n, d = q.shape
        e = v.shape[-1]

        # m = exp(alpha + beta * gamma * gamma ^ T)
        identity = torch.eye(d, device=torch.cuda.current_device())
        order_one_term = alpha.unsqueeze(-1) * identity
        order_two_term = (
            beta.unsqueeze(-1).unsqueeze(-1) * gamma.unsqueeze(-1) * gamma.unsqueeze(-2)
        )
        log_m = order_one_term + order_two_term
        m = torch.matrix_exp(log_m)

        o = torch.empty((b, h, n, e), dtype=q.dtype, device=torch.cuda.current_device())

        if initial_state is not None:
            s_initial_state = initial_state
        else:
            s_initial_state = None

        if output_final_state:
            s_final_state = torch.empty(
                (b, h, d, e), dtype=torch.float32, device=torch.cuda.current_device()
            )
        else:
            s_final_state = None

        USE_INITIAL_STATE = initial_state is not None
        OUTPUT_FINAL_STATE = output_final_state

        grid = (b * h,)

        _grpe_recurrence_fwd[grid](
            q,
            k,
            v,
            m,
            o,
            s_initial_state,
            s_final_state,
            b,
            h,
            n,
            d,
            e,
            USE_INITIAL_STATE,
            OUTPUT_FINAL_STATE,
        )

        if OUTPUT_FINAL_STATE:
            final_state = s_final_state
        else:
            final_state = None

        ctx.save_for_backward(q, k, v, alpha, beta, gamma)

        return o, final_state


def grpe_recurrence_triton(
    q, k, v, alpha, beta, gamma, initial_state=None, output_final_state=False
):
    o, final_state = GrpeRecurrenceFunction.apply(
        q, k, v, alpha, beta, gamma, initial_state, output_final_state
    )
    return o, final_state


if __name__ == "__main__":
    import torch.nn.functional as F

    b, h, n, d, e = 2, 8, 128, 64, 32
    dtype = torch.float32
    device = torch.cuda.current_device()
    q = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    k = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    v = (torch.randn((b, h, n, e), dtype=dtype, device=device)).requires_grad_()
    lower_bound = 0.95
    alpha = torch.log(
        lower_bound
        + (1 - lower_bound)
        * F.sigmoid(torch.randn((b, h, n, d), dtype=dtype, device=device))
    ).requires_grad_()
    beta = torch.log(
        lower_bound
        + (1 - lower_bound)
        * F.sigmoid(torch.randn((b, h, n), dtype=dtype, device=device))
    ).requires_grad_()
    gamma = F.normalize(
        torch.randn((b, h, n, d), dtype=dtype, device=device)
    ).requires_grad_()
    initial_state = None
    output_final_state = False

    o_recurrence_torch, final_state_recurrence_torch = grpe_recurrence_triton(
        q, k, v, alpha, beta, gamma, initial_state, output_final_state
    )
