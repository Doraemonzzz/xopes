import torch
import triton
import triton.language as tl

from xopes.utils import generate_configs, next_power_of_two


@triton.autotune(
    generate_configs(
        {
            "BLOCK_B": [16],
            "BLOCK_D": [128, 256, 512],
            "BLOCK_V": [128, 256, 512],
            "num_warps": [2, 4, 8],
        }
    ),
    key=["b", "d", "v", "k"],
)
@triton.jit
def _online_multinomial_triton(
    X,
    W,
    Sample,
    Lse,
    Max_value,
    seed,
    load_lse: tl.constexpr,
    load_max_value: tl.constexpr,
    b: tl.constexpr,
    d: tl.constexpr,
    v: tl.constexpr,
    k: tl.constexpr,  # num samples
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    off_b = tl.program_id(0)
    # compute offset
    offset_b = off_b * BLOCK_B
    offset_x = offset_b * d
    offset_sample = offset_b * k
    # mask
    b_mask = (offset_b + tl.arange(0, BLOCK_B)) < b
    k_mask = tl.arange(0, BLOCK_K) < k

    # BLOCK_B, k
    sample_block_ptr = (
        Sample
        + offset_sample
        + tl.arange(0, BLOCK_B)[:, None] * k
        + tl.arange(0, BLOCK_K)[None, :]
    )
    # for random
    # BLOCK_B, 1, k
    rand_block_ptr1 = tl.zeros([BLOCK_B, 1, BLOCK_K], dtype=tl.float32)
    # BLOCK_B, k
    rand_block_ptr2 = tl.zeros([BLOCK_B, BLOCK_K], dtype=tl.float32)

    value = -float("inf")

    if load_lse:
        lse_ptr = Lse + offset_b + tl.arange(0, BLOCK_B)[:, None]
        lse = tl.load(lse_ptr, mask=b_mask, other=value)
    else:
        lse = tl.full([BLOCK_B, 1], value=value, dtype=tl.float32)

    if load_max_value:
        max_valuek_ptr = Max_value + offset_b + tl.arange(0, BLOCK_B)[:, None]
        max_value = tl.load(max_valuek_ptr, mask=b_mask, other=value)
    else:
        max_value = tl.full([BLOCK_B, 1], value=value, dtype=tl.float32)

    sample = tl.zeros([BLOCK_B, BLOCK_K], dtype=tl.int32)

    for i in range(tl.cdiv(v, BLOCK_V)):
        logits = tl.zeros([BLOCK_B, BLOCK_V], dtype=tl.float32)
        v_mask = (i * BLOCK_V + tl.arange(0, BLOCK_V)) < v

        # BLOCK_B, BLOCK_D
        x_block_ptr = (
            X
            + offset_x
            + tl.arange(0, BLOCK_B)[:, None] * d
            + tl.arange(0, BLOCK_D)[None, :]
        )

        # BLOCK_D, BLOCK_V
        w_block_ptr = (
            W
            + tl.arange(0, BLOCK_D)[:, None] * v
            + i * BLOCK_V
            + tl.arange(0, BLOCK_V)[None, :]
        )

        for j in range(tl.cdiv(d, BLOCK_D)):
            d_mask = (j * BLOCK_D + tl.arange(0, BLOCK_D)) < d
            x = tl.load(x_block_ptr, mask=b_mask[:, None] * d_mask[None, :], other=0)
            w = tl.load(w_block_ptr, mask=d_mask[:, None] * v_mask[None, :], other=0)
            logits += tl.dot(x, w)

            x_block_ptr += BLOCK_D
            w_block_ptr += BLOCK_D * v

        logits = tl.where(v_mask[None, :], logits, value)

        # sample by multinomial
        max_value_curr = tl.max(logits, axis=1)[:, None]
        numerator = tl.exp(logits - max_value_curr)
        denominator = tl.sum(numerator, axis=1)[:, None] + 1e-5
        # lse(x) = lse(x - a) + a
        lse_curr = tl.log(denominator) + max_value_curr
        prob_curr = numerator / denominator
        # BLOCK_B, BLOCK_V
        prob_cum_curr = tl.cumsum(prob_curr, axis=1)
        # sample by uniform
        # BLOCK_B, 1, k
        p = tl.rand(seed, rand_block_ptr1)
        # find k, such that p1 + ... + p(k-1) < p <= p1 + ... + pk
        # e.g.
        # prob = [0.1, 0.2, 0.6, 0.1], p = 0.35 => k = 2
        # prob_cum = [0.1, 0.3, 0.9, 1.0]
        # upper = [0, 0, 1, 1]
        # (BLOCK_B, BLOCK_V, k)
        upper = (prob_cum_curr[:, :, None] >= p).to(tl.int32)
        # (BLOCK_B, k)
        sample_curr = i * BLOCK_V + tl.argmax(upper, axis=1)

        # sample by binomial
        # m = max(ma, mb)
        # lse(a, b) = log(exp(lse(a)) + exp(lse(b))) = log(exp(lse(a) - m) + exp(lse(b) - m)) + m
        max_value = tl.where(max_value > max_value_curr, max_value, max_value_curr)
        lse = tl.log(tl.exp(lse - max_value) + tl.exp(lse_curr - max_value)) + max_value
        # BLOCK_B, 1
        prob = tl.exp(lse_curr - lse)
        # x = 1: sample_curr
        # x = 0: sample
        # BLOCK_B, k
        index = tl.rand(seed, rand_block_ptr2) < prob
        sample = tl.where(
            index,
            sample_curr,
            sample,
        )

    tl.store(
        sample_block_ptr,
        sample.to(sample_block_ptr.dtype.element_ty),
        mask=k_mask[None, :],
    )


# @triton.autotune(
#     generate_configs(
#         {
#             "BLOCK_B": [16],
#             "BLOCK_D": [128, 256, 512],
#             "BLOCK_V": [128, 256, 512],
#             "num_warps": [2, 4, 8],
#         }
#     ),
#     key=["b", "d", "v", "k"],
# )
# @triton.jit
# def _online_multinomial_triton(
#     X,
#     W,
#     Sample,
#     Lse,
#     Max_value,
#     seed,
#     load_lse: tl.constexpr,
#     load_max_value: tl.constexpr,
#     b: tl.constexpr,
#     d: tl.constexpr,
#     v: tl.constexpr,
#     k: tl.constexpr,  # num samples
#     BLOCK_K: tl.constexpr,
#     BLOCK_B: tl.constexpr,
#     BLOCK_D: tl.constexpr,
#     BLOCK_V: tl.constexpr,
# ):
#     off_b = tl.program_id(0)
#     off_k = tl.program_id(1)
#     # compute offset
#     offset_b = off_b * BLOCK_B
#     offset_x = offset_b * d
#     offset_sample = offset_b * k
#     # mask
#     b_mask = (offset_b + tl.arange(0, BLOCK_B)) < b

#     # BLOCK_B, BLOCK_D
#     x_block_ptr = (
#         X
#         + offset_x
#         + tl.arange(0, BLOCK_B)[:, None] * d
#         + tl.arange(0, BLOCK_D)[None, :]
#     )
#     # BLOCK_D, BLOCK_V
#     w_block_ptr = (
#         W + tl.arange(0, BLOCK_D)[:, None] * v + tl.arange(0, BLOCK_V)[None, :]
#     )
#     # BLOCK_B, 1
#     sample_block_ptr = (
#         Sample
#         + offset_sample
#         + tl.arange(0, BLOCK_B)[:, None] * k
#     )
#     # for random
#     # BLOCK_B, 1
#     rand_block_ptr1 = tl.zeros([BLOCK_B, 1], dtype=tl.float32)
#     # BLOCK_B
#     rand_block_ptr2 = tl.zeros([BLOCK_B, 1], dtype=tl.float32)

#     value = -float("inf")

#     if load_lse:
#         lse_ptr = Lse + offset_b + tl.arange(0, BLOCK_B)[:, None]
#         lse = tl.load(lse_ptr, mask=b_mask, other=value)
#     else:
#         lse = tl.full([BLOCK_B, 1], value=value, dtype=tl.float32)

#     if load_max_value:
#         max_valuek_ptr = Max_value + offset_b + tl.arange(0, BLOCK_B)[:, None]
#         max_value = tl.load(max_valuek_ptr, mask=b_mask, other=value)
#     else:
#         max_value = tl.full([BLOCK_B, 1], value=value, dtype=tl.float32)

#     sample = tl.zeros([BLOCK_B, 1], dtype=tl.int32)
#     # sample_curr = tl.zeros([BLOCK_B, 1], dtype=tl.int32)

#     for i in range(tl.cdiv(v, BLOCK_V)):
#         logits = tl.zeros([BLOCK_B, BLOCK_V], dtype=tl.float32)
#         v_mask = (i * BLOCK_V + tl.arange(0, BLOCK_V)) < v

#         for j in range(tl.cdiv(d, BLOCK_D)):
#             d_mask = (j * BLOCK_D + tl.arange(0, BLOCK_D)) < d
#             x = tl.load(x_block_ptr, mask=b_mask[:, None] * d_mask[None, :], other=0)
#             w = tl.load(w_block_ptr, mask=d_mask[:, None] * v_mask[None, :], other=0)
#             logits += tl.dot(x, w)

#         logits = tl.where(v_mask[None, :], logits, value)

#         # sample by multinomial
#         max_value_curr = tl.max(logits, axis=1)[:, None]
#         # tl.static_print("aaa", logits, max_value_curr)
#         numerator = tl.exp(logits - max_value_curr)
#         denominator = tl.sum(numerator, axis=1)[:, None]
#         # lse(x) = lse(x - a) + a
#         lse_curr = tl.log(denominator) + max_value_curr
#         prob_curr = numerator / denominator
#         # BLOCK_B, BLOCK_V
#         prob_cum_curr = tl.cumsum(prob_curr, axis=1)
#         # sample by uniform
#         # BLOCK_B, 1
#         p = tl.rand(seed, rand_block_ptr1)
#         # find k, such that p1 + ... + p(k-1) < p <= p1 + ... + pk
#         # e.g.
#         # prob = [0.1, 0.2, 0.6, 0.1], p = 0.35 => k = 2
#         # prob_cum = [0.1, 0.3, 0.9, 1.0]
#         # upper = [0, 0, 1, 1]
#         # (BLOCK_B, BLOCK_V, k)
#         # tl.static_print("aaa", prob_curr, prob_cum_curr, prob_cum_curr[:, :, None], p)
#         # tl.device_print("aaa", prob_curr)
#         upper = (prob_cum_curr >= p).to(tl.int32)
#         # tl.static_print("aaa", prob_cum_curr)
#         # tl.device_print("aaa", prob_cum_curr)
#         # BLOCK_B
#         sample_curr = i * BLOCK_V + tl.argmin(upper, axis=1)[:, None]
#         # tl.static_print("bbb", upper)
#         # tl.static_print("ccc", sample_curr)
#         # tl.static_print("ddd", sample)
#         # tl.device_print("aaa", sample_curr)

#         # sample by binomial
#         # m = max(ma, mb)
#         # lse(a, b) = log(exp(lse(a)) + exp(lse(b))) = log(exp(lse(a) - m) + exp(lse(b) - m)) + m
#         max_value = tl.where(max_value > max_value_curr, max_value, max_value_curr)
#         lse = tl.log(tl.exp(lse - max_value) + tl.exp(lse_curr - max_value)) + max_value
#         # BLOCK_B, 1
#         prob = tl.exp(lse_curr - lse)
#         # x = 1: sample_curr
#         # x = 0: sample
#         # BLOCK_B, 1
#         index = tl.rand(seed, rand_block_ptr2) < prob
#         # tl.static_print("ccc", index, sample_curr, sample)
#         # tl.static_print("eee", index)
#         # sample = index * sample_curr + (1 - index) * sample

#         sample = tl.where(
#             index,
#             sample_curr,
#             sample,
#         )

#         # sample = sample_curr

#     # tl.static_print("aaa", sample_block_ptr, sample, k_mask)
#     tl.store(sample_block_ptr, sample.to(sample_block_ptr.dtype.element_ty))


def online_multinomial_triton(x, W, num_samples, lse=None, max_value=None):
    """
    x: b d
    W: d v
    lse: b
    max_value: b
    """
    b, d = x.shape
    d, v = W.shape
    sample = torch.empty((b, num_samples), dtype=torch.int32, device=x.device)
    load_lse = lse is not None
    load_max_value = max_value is not None
    BLOCK_K = max(16, next_power_of_two(num_samples))
    seed = 0

    # def grid(meta):
    #     return (triton.cdiv(b, meta["BLOCK_B"]), num_samples)

    def grid(meta):
        return (triton.cdiv(b, meta["BLOCK_B"]),)

    _online_multinomial_triton[grid](
        x,
        W,
        sample,
        lse,
        max_value,
        seed,
        load_lse,
        load_max_value,
        b,
        d,
        v,
        num_samples,
        BLOCK_K,
    )

    return sample.to(torch.int64)


if __name__ == "__main__":
    # unit test
    b = 2
    d = 2048
    V = 4096
    num_samples = 32
    dtype = torch.float32
    device = torch.cuda.current_device()

    x = torch.randn(b, d, dtype=dtype, device=device)
    W = torch.randn(d, V, dtype=dtype, device=device)
    sample = online_multinomial_triton(x, W, num_samples)
