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
def _parallel_gumbel_multinomial_triton(
    X,
    W,
    Sample,
    Lse,
    Lse_cache,
    seed,
    load_lse: tl.constexpr,
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
    offset_lse = offset_b * NUM_BLOCK_V + off_v
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
        Lse_cache + offset_lse + tl.arange(0, BLOCK_B)[:, None] * NUM_BLOCK_V
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

    logits = tl.zeros([BLOCK_B, BLOCK_V], dtype=tl.float32)
    v_mask = (offset_v + tl.arange(0, BLOCK_V)) < v

    for i in range(tl.cdiv(d, BLOCK_D)):
        d_mask = (i * BLOCK_D + tl.arange(0, BLOCK_D)) < d
        x = tl.load(x_block_ptr, mask=b_mask[:, None] & d_mask[None, :], other=0)
        w = tl.load(w_block_ptr, mask=d_mask[:, None] & v_mask[None, :], other=0)
        logits = tl.dot(x, w, logits)

        x_block_ptr += BLOCK_D
        w_block_ptr += BLOCK_D * v

    logits = tl.where(b_mask[:, None] & v_mask[None, :], logits, value)
    # use Gumbel Max to sample
    # sample from p1, ..., pk is equivalent to sample
    # argmax {log pi - log(-log(ui))} = argmax {logits - log(-log(ui))}, ui ~ U(0,1)
    # (BLOCK_B, 1, k)
    u = tl.rand(seed, rand_block_ptr)
    stat = logits[:, :, None] - tl.log(-tl.log(u))
    # (BLOCK_B, k)
    sample = offset_v + tl.argmax(stat, axis=1)

    # compute lse
    max_value = tl.max(logits, axis=1)[:, None]
    numerator = tl.exp(logits - max_value)
    denominator = tl.sum(numerator, axis=1)[:, None]
    # lse(x) = lse(x - a) + a
    lse = tl.log(denominator) + max_value

    tl.store(
        sample_block_ptr,
        sample.to(sample_block_ptr.dtype.element_ty),
        mask=b_mask[:, None],
    )
    tl.store(
        lse_cache_ptr, lse.to(lse_cache_ptr.dtype.element_ty), mask=b_mask[:, None]
    )


# @triton.autotune(
#     generate_configs(
#         {
#             "BLOCK_B": [32, 64, 128],
#             "num_warps": [2, 4, 8],
#         }
#     ),
#     key=["b", "NUM_BLOCK_V", "k"],
# )
# @triton.jit
# def _parallel_gumbel_multinomial_reduce_triton(
#     Sample,
#     Lse_cache,
#     Sample_out,
#     seed,
#     load_lse: tl.constexpr,
#     b: tl.constexpr,
#     d: tl.constexpr,
#     v: tl.constexpr,
#     k: tl.constexpr,  # num samples
#     BLOCK_V: tl.constexpr,
#     NUM_BLOCK_V: tl.constexpr,
#     BLOCK_K: tl.constexpr,
#     BLOCK_B: tl.constexpr,
# ):
#     off_b = tl.program_id(0)
#     off_k = tl.program_id(1)
#     # compute offset
#     offset_b = off_b * BLOCK_B
#     offset_k = off_k
#     offset_sample = offset_b * NUM_BLOCK_V * k + offset_k
#     offset_sample_out = offset_b * k + offset_k
#     offset_lse = offset_b * NUM_BLOCK_V
#     # mask
#     b_mask = (offset_b + tl.arange(0, BLOCK_B)) < b

#     # BLOCK_B,
#     sample_out_block_ptr = (
#         Sample_out + offset_sample_out + tl.arange(0, BLOCK_B)[:, None] * k
#     )
#     # BLOCK_B, NUM_BLOCK_V
#     lse_cache_ptr = (
#         Lse_cache
#         + offset_lse
#         + tl.arange(0, BLOCK_B)[:, None] * NUM_BLOCK_V
#         + tl.arange(0, NUM_BLOCK_V)[None, :]
#     )
#     # for random
#     # BLOCK_B, 1
#     rand_block_ptr = tl.zeros([BLOCK_B, 1], dtype=tl.float32)

#     value = -float("inf")

#     logits = tl.load(lse_cache_ptr, mask=b_mask[:, None], other=value)
#     # use Gumbel Max to sample
#     # sample from p1, ..., pk is equivalent to sample
#     # argmax {log pi - log(-log(ui))} = argmax {logits - log(-log(ui))}, ui ~ U(0,1)
#     # (BLOCK_B, 1)
#     u = tl.rand(seed, rand_block_ptr)
#     stat = logits - tl.log(-tl.log(u))
#     # (BLOCK_B,)
#     index = tl.argmax(stat, axis=1)

#     # BLOCK_B, 1
#     sample_index_block_ptr = Sample + offset_sample + index[:, None] * k
#     sample_out = tl.load(sample_index_block_ptr, mask=b_mask[:, None])

#     tl.store(
#         sample_out_block_ptr,
#         sample_out.to(sample_out_block_ptr.dtype.element_ty),
#         mask=b_mask[:, None],
#     )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [2, 4, 8],
        }
    ),
    key=["b", "NUM_BLOCK_V", "k"],
)
@triton.jit
def _parallel_gumbel_multinomial_reduce_triton(
    Sample,
    Lse_cache,
    Sample_out,
    Lse_out,
    seed,
    output_lse: tl.constexpr,
    b: tl.constexpr,
    d: tl.constexpr,
    v: tl.constexpr,
    k: tl.constexpr,  # num samples
    BLOCK_V: tl.constexpr,
    NUM_BLOCK_V: tl.constexpr,
    NUM_BLOCK_V_PAD: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_k = tl.program_id(1)
    # compute offset
    offset_b = off_b
    offset_k = off_k
    offset_sample = offset_b * NUM_BLOCK_V * k + offset_k
    offset_sample_out = offset_b * k + offset_k
    offset_lse = offset_b * NUM_BLOCK_V
    # mask
    num_block_v_mask = tl.arange(0, NUM_BLOCK_V_PAD) < NUM_BLOCK_V

    # 1, 1
    sample_out_block_ptr = Sample_out + offset_sample_out + tl.arange(0, 1)[:, None] * k
    # 1, NUM_BLOCK_V
    lse_cache_ptr = Lse_cache + offset_lse + tl.arange(0, NUM_BLOCK_V_PAD)[None, :]
    # for random
    # 1, 1
    rand_block_ptr = tl.zeros([1, 1], dtype=tl.float32)

    value = -float("inf")

    logits = tl.load(lse_cache_ptr, mask=num_block_v_mask[None, :], other=value)
    # use Gumbel Max to sample
    # sample from p1, ..., pk is equivalent to sample
    # argmax {log pi - log(-log(ui))} = argmax {logits - log(-log(ui))}, ui ~ U(0,1)
    # (1, 1)
    u = tl.rand(seed, rand_block_ptr)
    stat = logits - tl.log(-tl.log(u))
    # (1,)
    index = tl.argmax(stat, axis=1)

    # 1, 1
    sample_index_block_ptr = Sample + offset_sample + index[:, None] * k
    sample_out = tl.load(sample_index_block_ptr)
    tl.store(
        sample_out_block_ptr,
        sample_out.to(sample_out_block_ptr.dtype.element_ty),
    )

    if output_lse:  # only save once
        if off_k == 0:  # work around compiler bug
            lse_out_block_ptr = Lse_out + offset_b + tl.arange(0, 1)[:, None]
            max_value = tl.max(logits, axis=1)[:, None]
            numerator = tl.exp(logits - max_value)
            denominator = tl.sum(numerator, axis=1)[:, None]
            # lse(x) = lse(x - a) + a
            lse = tl.log(denominator) + max_value
            tl.store(lse_out_block_ptr, lse.to(lse_out_block_ptr.dtype.element_ty))


def parallel_gumbel_multinomial_triton(x, W, num_samples=1, lse=None, output_lse=False):
    """
    x: b d or b 1 d
    W: d v
    lse: b
    max_value: b
    """
    b = x.shape[0]
    d = x.shape[1]
    d, v = W.shape
    x = x.contiguous()
    W = W.contiguous()

    # BLOCK_V = min(128, v)
    BLOCK_V = 128
    NUM_BLOCK_V = (v + BLOCK_V - 1) // BLOCK_V
    sample = torch.empty(
        (b, NUM_BLOCK_V, num_samples), dtype=torch.int32, device=x.device
    )
    lse_cache = torch.empty((b, NUM_BLOCK_V), dtype=torch.float32, device=x.device)

    load_lse = lse is not None
    BLOCK_K = max(16, next_power_of_two(num_samples))
    seed = 0

    def grid(meta):
        return (triton.cdiv(b, meta["BLOCK_B"]), triton.cdiv(v, BLOCK_V))

    _parallel_gumbel_multinomial_triton[grid](
        x,
        W,
        sample,
        lse,
        lse_cache,
        seed,
        load_lse,
        b,
        d,
        v,
        num_samples,
        BLOCK_V,
        NUM_BLOCK_V,
        BLOCK_K,
    )
    # print("bbb")
    # print(lse_cache)

    def grid(meta):
        return (b, num_samples)

    sample_out = torch.empty((b, num_samples), dtype=torch.int32, device=x.device)
    lse_out = None
    if output_lse:
        lse_out = torch.empty((b, 1), dtype=torch.float32, device=x.device)

    NUM_BLOCK_V_PAD = next_power_of_two(NUM_BLOCK_V)
    _parallel_gumbel_multinomial_reduce_triton[grid](
        sample,
        lse_cache,
        sample_out,
        lse_out,
        seed,
        output_lse,
        b,
        d,
        v,
        num_samples,
        BLOCK_V,
        NUM_BLOCK_V,
        NUM_BLOCK_V_PAD,
        BLOCK_K,
    )

    return sample_out.to(torch.int64), lse_out


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
    sample = parallel_gumbel_multinomial_triton(x, W, num_samples)
