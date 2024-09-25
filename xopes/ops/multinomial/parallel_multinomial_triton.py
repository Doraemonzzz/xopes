import torch
import triton
import triton.language as tl

from xopes.utils import generate_configs, next_power_of_two


@triton.autotune(
    generate_configs(
        {
            "BLOCK_B": [32, 64, 128],
            "BLOCK_D": [32, 64, 128],
            "num_warps": [2, 4, 8],
        }
    ),
    key=["b", "d", "v", "k"],
)
@triton.jit
def _parallel_multinomial_triton(
    X,
    W,
    Sample,
    Lse,
    Max_value,
    Lse_cache,
    Max_value_cache,
    seed,
    load_lse: tl.constexpr,
    load_max_value: tl.constexpr,
    b: tl.constexpr,
    d: tl.constexpr,
    v: tl.constexpr,
    k: tl.constexpr,  # num samples
    BLOCK_V: tl.constexpr,
    NUM_BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_v = tl.program_id(1)
    # compute offset
    offset_b = off_b * BLOCK_B
    offset_x = offset_b * d
    offset_v = off_v * BLOCK_V
    offset_sample = offset_b * NUM_BLOCK_V * k + off_v * k
    offset_lse_max_value = offset_b * NUM_BLOCK_V + off_v
    # mask
    b_mask = (offset_b + tl.arange(0, BLOCK_B)) < b

    # 1, BLOCK_K
    sample_block_ptr = (
        Sample
        + offset_sample
        + tl.arange(0, BLOCK_B)[:, None] * NUM_BLOCK_V * k
        + tl.arange(0, k)[None, :]
    )
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
        + offset_v
        + tl.arange(0, BLOCK_D)[:, None] * v
        + tl.arange(0, BLOCK_V)[None, :]
    )
    lse_cache_ptr = (
        Lse_cache + offset_lse_max_value + tl.arange(0, BLOCK_B)[:, None] * NUM_BLOCK_V
    )
    max_value_cache_ptr = (
        Max_value_cache
        + offset_lse_max_value
        + tl.arange(0, BLOCK_B)[:, None] * NUM_BLOCK_V
    )
    # for random
    # BLOCK_B, 1, k
    rand_block_ptr = tl.zeros([BLOCK_B, 1, k], dtype=tl.float32)

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

    logits = tl.zeros([BLOCK_B, BLOCK_V], dtype=tl.float32)
    v_mask = (offset_v + tl.arange(0, BLOCK_V)) < v

    for i in range(tl.cdiv(d, BLOCK_D)):
        d_mask = (i * BLOCK_D + tl.arange(0, BLOCK_D)) < d
        x = tl.load(x_block_ptr, mask=b_mask[:, None] * d_mask[None, :], other=0)
        w = tl.load(w_block_ptr, mask=d_mask[:, None] * v_mask[None, :], other=0)
        logits = tl.dot(x, w, logits)

        x_block_ptr += BLOCK_D
        w_block_ptr += BLOCK_D * v

    logits = tl.where(v_mask[None, :], logits, value)

    # sample by multinomial
    max_value_curr = tl.max(logits, axis=1)[:, None]
    numerator = tl.exp(logits - max_value_curr)
    denominator = tl.sum(numerator, axis=1)[:, None]
    # lse(x) = lse(x - a) + a
    lse_curr = tl.log(denominator) + max_value_curr
    prob_curr = numerator / denominator
    # BLOCK_B, BLOCK_V
    prob_cum_curr = tl.cumsum(prob_curr, axis=1)
    # sample by uniform
    # BLOCK_B, 1, k
    p = tl.rand(seed, rand_block_ptr)
    # find k, such that p1 + ... + p(k-1) < p <= p1 + ... + pk
    # e.g.
    # prob = [0.1, 0.2, 0.6, 0.1], p = 0.35 => k = 2
    # prob_cum = [0.1, 0.3, 0.9, 1.0]
    # upper = [0, 0, 1, 1]
    # (BLOCK_B, BLOCK_V, k)

    upper = (prob_cum_curr[:, :, None] >= p).to(tl.int32)
    # (BLOCK_B, k)
    sample = offset_v + tl.argmax(upper, axis=1)

    tl.store(
        sample_block_ptr,
        sample.to(sample_block_ptr.dtype.element_ty),
        mask=b_mask[:, None],
    )
    tl.store(
        lse_cache_ptr, lse_curr.to(lse_cache_ptr.dtype.element_ty), mask=b_mask[:, None]
    )
    tl.store(
        max_value_cache_ptr,
        max_value_curr.to(max_value_cache_ptr.dtype.element_ty),
        mask=b_mask[:, None],
    )


@triton.autotune(
    generate_configs(
        {
            "BLOCK_B": [32, 64, 128],
            "num_warps": [2, 4, 8],
        }
    ),
    key=["b", "NUM_BLOCK_V", "k"],
)
@triton.jit
def _parallel_multinomial_reduce_triton(
    Sample,
    Lse_cache,
    Max_value_cache,
    Sample_out,
    seed,
    load_lse: tl.constexpr,
    load_max_value: tl.constexpr,
    b: tl.constexpr,
    d: tl.constexpr,
    v: tl.constexpr,
    k: tl.constexpr,  # num samples
    BLOCK_V: tl.constexpr,
    NUM_BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_k = tl.program_id(1)
    # compute offset
    offset_b = off_b * BLOCK_B
    offset_k = off_k
    offset_sample = offset_b * NUM_BLOCK_V * k + offset_k
    offset_sample_out = offset_b * k + offset_k
    offset_lse_max_value = offset_b * NUM_BLOCK_V
    # mask
    b_mask = (offset_b + tl.arange(0, BLOCK_B)) < b

    # BLOCK_B,
    sample_out_block_ptr = (
        Sample_out + offset_sample_out + tl.arange(0, BLOCK_B)[:, None] * k
    )
    # BLOCK_B, NUM_BLOCK_V
    lse_cache_ptr = (
        Lse_cache
        + offset_lse_max_value
        + tl.arange(0, BLOCK_B)[:, None] * NUM_BLOCK_V
        + tl.arange(0, NUM_BLOCK_V)[None, :]
    )
    max_value_cache_ptr = (
        Max_value_cache
        + offset_lse_max_value
        + tl.arange(0, BLOCK_B)[:, None] * NUM_BLOCK_V
        + tl.arange(0, NUM_BLOCK_V)[None, :]
    )
    # for random
    # BLOCK_B, 1
    rand_block_ptr = tl.zeros([BLOCK_B, 1], dtype=tl.float32)

    value = -float("inf")

    logits = tl.load(lse_cache_ptr, mask=b_mask[:, None], other=value)
    max_value = tl.load(max_value_cache_ptr, mask=b_mask[:, None], other=value)

    # sample by multinomial
    max_value_curr = tl.max(logits, axis=1)[:, None]
    numerator = tl.exp(logits - max_value_curr)
    denominator = tl.sum(numerator, axis=1)[:, None]
    # lse(x) = lse(x - a) + a
    prob_curr = numerator / denominator
    # BLOCK_B, NUM_BLOCK_V
    prob_cum_curr = tl.cumsum(prob_curr, axis=1)
    # sample by uniform
    # BLOCK_B, 1
    p = tl.rand(seed, rand_block_ptr)
    # find k, such that p1 + ... + p(k-1) < p <= p1 + ... + pk
    # e.g.
    # prob = [0.1, 0.2, 0.6, 0.1], p = 0.35 => k = 2
    # prob_cum = [0.1, 0.3, 0.9, 1.0]
    # upper = [0, 0, 1, 1]
    # BLOCK_B, NUM_BLOCK_V
    upper = (prob_cum_curr >= p).to(tl.int32)
    # BLOCK_B,
    index = tl.argmax(upper, axis=1)

    # BLOCK_B, 1
    sample_index_block_ptr = Sample + offset_sample + index[:, None] * k
    sample_out = tl.load(sample_index_block_ptr, mask=b_mask[:, None])

    tl.store(
        sample_out_block_ptr,
        sample_out.to(sample_out_block_ptr.dtype.element_ty),
        mask=b_mask[:, None],
    )


def parallel_multinomial_triton(x, W, num_samples, lse=None, max_value=None):
    """
    x: b d
    W: d v
    lse: b
    max_value: b
    """
    b, d = x.shape
    d, v = W.shape

    BLOCK_V = 128
    NUM_BLOCK_V = (v + BLOCK_V - 1) // BLOCK_V
    sample = torch.empty(
        (b, NUM_BLOCK_V, num_samples), dtype=torch.int32, device=x.device
    )
    lse_cache = torch.empty((b, NUM_BLOCK_V), dtype=torch.float32, device=x.device)
    max_value_cache = torch.empty(
        (b, NUM_BLOCK_V), dtype=torch.float32, device=x.device
    )

    load_lse = lse is not None
    load_max_value = max_value is not None
    BLOCK_K = max(16, next_power_of_two(num_samples))
    seed = 0

    def grid(meta):
        return (triton.cdiv(b, meta["BLOCK_B"]), triton.cdiv(v, BLOCK_V))

    _parallel_multinomial_triton[grid](
        x,
        W,
        sample,
        lse,
        max_value,
        lse_cache,
        max_value_cache,
        seed,
        load_lse,
        load_max_value,
        b,
        d,
        v,
        num_samples,
        BLOCK_V,
        NUM_BLOCK_V,
        BLOCK_K,
    )

    def grid(meta):
        return (triton.cdiv(b, meta["BLOCK_B"]), num_samples)

    sample_out = torch.empty((b, num_samples), dtype=torch.int32, device=x.device)

    _parallel_multinomial_reduce_triton[grid](
        sample,
        lse_cache,
        max_value_cache,
        sample_out,
        seed,
        load_lse,
        load_max_value,
        b,
        d,
        v,
        num_samples,
        BLOCK_V,
        NUM_BLOCK_V,
        BLOCK_K,
    )

    return sample_out.to(torch.int64)


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
    sample = parallel_multinomial_triton(x, W, num_samples)
