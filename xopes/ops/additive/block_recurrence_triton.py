import torch

from xopes.ops.multiplicative import fused_chunk_gla

# from xopes.utils import contiguous, max_power_of_2_divisor

# # HEAD_DIM = 64


# def _get_fw_configs():
#     return None


# # @triton.autotune(
# #     configs=[
# #         triton.Config({'BLOCK_D': 32}, num_warps=1),
# #         triton.Config({'BLOCK_D': 32}, num_warps=2),
# #         triton.Config({'BLOCK_D': 32}, num_warps=4),
# #         triton.Config({'BLOCK_D': 32}, num_warps=8),
# #         triton.Config({'BLOCK_D': 64}, num_warps=1),
# #         triton.Config({'BLOCK_D': 64}, num_warps=2),
# #         triton.Config({'BLOCK_D': 64}, num_warps=4),
# #         triton.Config({'BLOCK_D': 64}, num_warps=8),
# #         triton.Config({'BLOCK_D': 128}, num_warps=1),
# #         triton.Config({'BLOCK_D': 128}, num_warps=2),
# #         triton.Config({'BLOCK_D': 128}, num_warps=4),
# #         triton.Config({'BLOCK_D': 128}, num_warps=8),
# #     ],
# #     key=['d']
# # )
# @triton.jit
# def _additive_block_recurrence_fwd(
#     Q,
#     K,
#     V,
#     G,
#     O,
#     S_INITIAL_STATE,
#     DENOM_INITIAL_STATE,
#     M_INITIAL_STATE,
#     S_FINAL_STATE,
#     DENOM_FINAL_STATE,
#     M_FINAL_STATE,
#     b: tl.constexpr,
#     h: tl.constexpr,
#     n: tl.constexpr,
#     d: tl.constexpr,
#     e: tl.constexpr,
#     BLOCK_D: tl.constexpr,
#     BLOCK_E: tl.constexpr,
#     NUM_BLOCK_D: tl.constexpr,
#     NUM_BLOCK_E: tl.constexpr,
#     USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
#     OUTPUT_FINAL_STATE: tl.constexpr,  # whether to output final state
# ):
#     off_bh = tl.program_id(2)
#     off_bh % h
#     off_bh // h
#     off_d, off_e = tl.program_id(0), tl.program_id(1)
#     # compute offset
#     off_qkg = off_bh * n * d
#     off_v = off_bh * n * e
#     off_o = (off_d * b * h + off_bh) * n * e
#     off_d = off_d * BLOCK_D
#     off_e = off_e * BLOCK_E
#     off_s = off_bh * d * e
#     off_denom_m = off_bh * d
#     # mask
#     mask_denom_m = (off_d + tl.arange(0, BLOCK_D) < d)[:, None]

#     # get block ptr
#     q_trans_block_ptr = tl.make_block_ptr(
#         base=Q + off_qkg,
#         shape=(d, n),
#         strides=(1, d),
#         offsets=(
#             off_d,
#             0,
#         ),
#         block_shape=(
#             BLOCK_D,
#             1,
#         ),
#         order=(0, 1),
#     )
#     k_trans_block_ptr = tl.make_block_ptr(
#         base=K + off_qkg,
#         shape=(d, n),
#         strides=(1, d),
#         offsets=(off_d, 0),
#         block_shape=(BLOCK_D, 1),
#         order=(0, 1),
#     )
#     v_block_ptr = tl.make_block_ptr(
#         base=V + off_v,
#         shape=(n, e),
#         strides=(e, 1),
#         offsets=(0, off_e),
#         block_shape=(1, BLOCK_E),
#         order=(1, 0),
#     )
#     g_trans_block_ptr = tl.make_block_ptr(
#         base=G + off_qkg,
#         shape=(d, n),
#         strides=(1, d),
#         offsets=(
#             off_d,
#             0,
#         ),
#         block_shape=(BLOCK_D, 1),
#         order=(0, 1),
#     )
#     o_block_ptr = tl.make_block_ptr(
#         base=O + off_o,
#         shape=(n, e),
#         strides=(e, 1),
#         offsets=(0, off_e),
#         block_shape=(1, BLOCK_E),
#         order=(1, 0),
#     )

#     if USE_INITIAL_STATE:

#         s_block_ptr = tl.make_block_ptr(
#             base=S_INITIAL_STATE + off_s,
#             shape=(d, e),
#             strides=(e, 1),
#             offsets=(
#                 off_d,
#                 off_e,
#             ),
#             block_shape=(
#                 BLOCK_D,
#                 BLOCK_E,
#             ),
#             order=(1, 0),
#         )

#         # !!!! dont use tl.make_block_ptr, since it will cause bug
#         denom_block_ptr = (
#             DENOM_INITIAL_STATE + off_denom_m + off_d + tl.arange(0, BLOCK_D)[:, None]
#         )
#         m_block_ptr = (
#             M_INITIAL_STATE + off_denom_m + off_d + tl.arange(0, BLOCK_D)[:, None]
#         )

#         s = tl.load(s_block_ptr, boundary_check=(0, 1)).to(tl.float32)
#         denom = tl.load(denom_block_ptr, mask=mask_denom_m).to(tl.float32)
#         m = tl.load(m_block_ptr, mask=mask_denom_m).to(tl.float32)
#     else:

#         s = tl.zeros([BLOCK_D, BLOCK_E], dtype=tl.float32)
#         denom = tl.zeros([BLOCK_D, 1], dtype=tl.float32)

#         m = tl.zeros([BLOCK_D, 1], dtype=tl.float32) + (-1e5)

#     for i in range(n):
#         # boundary check on feature dim
#         q_trans = tl.load(q_trans_block_ptr, boundary_check=(0, 1)).to(tl.float32)
#         k_trans = tl.load(k_trans_block_ptr, boundary_check=(0, 1)).to(tl.float32)
#         v = tl.load(v_block_ptr, boundary_check=(0, 1)).to(tl.float32)
#         g_trans = tl.load(g_trans_block_ptr, boundary_check=(0, 1)).to(tl.float32)

#         # update params
#         # d 1
#         m_ = tl.maximum(m, g_trans)
#         g_trans = g_trans - m_
#         lambda_ = tl.exp(m - m_)
#         # compute
#         g_exp_trans = tl.exp(g_trans)
#         k_bar_trans = g_exp_trans * k_trans
#         # d 1, 1 e -> d e
#         s = lambda_ * s + k_bar_trans.to(v.dtype) * v
#         denom = lambda_ * denom + g_exp_trans
#         # d 1, d e -> d e
#         o = (q_trans) * (s / denom)
#         # d e -> 1 e
#         o = tl.sum(o, axis=0)[None, :]

#         m = m_

#         tl.store(o_block_ptr, o.to(o_block_ptr.dtype.element_ty), boundary_check=(0, 1))

#         # update block ptr
#         q_trans_block_ptr = tl.advance(q_trans_block_ptr, (0, 1))
#         k_trans_block_ptr = tl.advance(k_trans_block_ptr, (0, 1))
#         v_block_ptr = tl.advance(v_block_ptr, (1, 0))
#         g_trans_block_ptr = tl.advance(g_trans_block_ptr, (0, 1))
#         o_block_ptr = tl.advance(o_block_ptr, (1, 0))

#     if OUTPUT_FINAL_STATE:
#         s_final_block_ptr = tl.make_block_ptr(
#             base=S_FINAL_STATE + off_s,
#             shape=(d, e),
#             strides=(e, 1),
#             offsets=(
#                 off_d,
#                 off_e,
#             ),
#             block_shape=(
#                 BLOCK_D,
#                 BLOCK_E,
#             ),
#             order=(1, 0),
#         )

#         # !!!! dont use tl.make_block_ptr, since it will cause bug
#         denom_final_block_ptr = (
#             DENOM_FINAL_STATE + off_denom_m + off_d + tl.arange(0, BLOCK_D)[:, None]
#         )
#         m_final_block_ptr = (
#             M_FINAL_STATE + off_denom_m + off_d + tl.arange(0, BLOCK_D)[:, None]
#         )

#         tl.store(
#             s_final_block_ptr,
#             s.to(s_final_block_ptr.dtype.element_ty),
#             boundary_check=(0, 1),
#         )

#         tl.store(
#             denom_final_block_ptr,
#             denom.to(denom_final_block_ptr.dtype.element_ty),
#             mask=mask_denom_m,
#         )

#         tl.store(
#             m_final_block_ptr,
#             m.to(m_final_block_ptr.dtype.element_ty),
#             mask=mask_denom_m,
#         )


# class AdditiveRecurrenceFunction(torch.autograd.Function):
#     @staticmethod
#     @contiguous
#     def forward(ctx, q, k, v, g, initial_state=None, output_final_state=None):
#         b, h, n, d = q.shape
#         e = v.shape[-1]

#         # split over head dim to avoid shared memory not enough
#         head_dim_d = max_power_of_2_divisor(d)
#         head_dim_e = max_power_of_2_divisor(e)

#         BLOCK_D, BLOCK_E = min(d, head_dim_d), min(e, head_dim_e)
#         NUM_BLOCK_D, NUM_BLOCK_E = triton.cdiv(d, BLOCK_D), triton.cdiv(e, BLOCK_E)
#         o = torch.empty(
#             (NUM_BLOCK_D, b, h, n, e), dtype=q.dtype, device=torch.cuda.current_device()
#         )

#         if initial_state is not None:
#             s_initial_state, denom_initial_state, m_initial_state = initial_state
#         else:
#             pass

#         if output_final_state:
#             s_final_state = torch.empty(
#                 (b, h, d, e), dtype=torch.float32, device=torch.cuda.current_device()
#             )
#             denom_final_state = torch.empty(
#                 (b, h, d, 1), dtype=torch.float32, device=torch.cuda.current_device()
#             )
#             m_final_state = torch.empty(
#                 (b, h, d, 1), dtype=torch.float32, device=torch.cuda.current_device()
#             )
#         else:
#             s_final_state = None
#             denom_final_state = None
#             m_final_state = None

#         initial_state is not None
#         OUTPUT_FINAL_STATE = output_final_state

#         # compute log exp sum
#         grid = (
#             b * h,
#             d,
#         )

#         # grid = (
#         #     NUM_BLOCK_D,
#         #     NUM_BLOCK_E,
#         #     b * h,
#         # )

#         # _additive_recurrence_fwd[grid](
#         #     q,
#         #     k,
#         #     v,
#         #     g,
#         #     o,
#         #     s_initial_state,
#         #     denom_initial_state,
#         #     m_initial_state,
#         #     s_final_state,
#         #     denom_final_state,
#         #     m_final_state,
#         #     b,
#         #     h,
#         #     n,
#         #     d,
#         #     e,
#         #     BLOCK_D,
#         #     BLOCK_E,
#         #     NUM_BLOCK_D,
#         #     NUM_BLOCK_E,
#         #     USE_INITIAL_STATE,
#         #     OUTPUT_FINAL_STATE,
#         # )

#         if OUTPUT_FINAL_STATE:
#             final_state = (s_final_state, denom_final_state, m_final_state)
#         else:
#             final_state = None

#         o = o.sum(0)

#         ctx.save_for_backward(q, k, v, g)

#         return o, final_state


# def additive_rule_block_recurrence_triton(
#     q, k, v, g, initial_state=None, output_final_state=False
# ):
#     o, final_state = AdditiveRecurrenceFunction.apply(
#         q, k, v, g, initial_state, output_final_state
#     )
#     return o, final_state


# use gla for now
def additive_rule_block_recurrence_triton(
    q, k, v, g, initial_state=None, output_final_state=False
):
    b, h, n, d = q.shape
    pad = torch.ones(
        (b, h, 1, d), dtype=torch.float32, device=torch.cuda.current_device()
    ) * (-1e5)
    g_log_exp_cumsum = torch.logcumsumexp(
        torch.concat([pad, g], dim=-2), dim=-2
    ).float()
    lambda_ = torch.exp(g_log_exp_cumsum[:, :, :-1] - g_log_exp_cumsum[:, :, 1:])
    # print(lambda_[0, 0, 0,])
    k_ = k * (1 - lambda_)
    lambda_[:, :, 0] = 1

    # assert False
    o, final_state = fused_chunk_gla(
        q, k_, v, torch.log(lambda_), 1, initial_state, output_final_state
    )
    return o, final_state
