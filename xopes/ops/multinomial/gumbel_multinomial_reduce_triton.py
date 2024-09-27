import torch
import triton
import triton.language as tl

from xopes.utils import generate_configs, next_power_of_two


@triton.autotune(
    generate_configs(
        {
            "num_warps": [2, 4, 8],
        }
    ),
    key=["b", "k"],
)
@triton.jit
def _gumbel_multinomial_reduce_triton(
    Sample,  # b k m
    Lse,  # b m
    Sample_out,  # b k
    seed,
    b: tl.constexpr,
    k: tl.constexpr,
    m: tl.constexpr,  # num samples
    BLOCK_M: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_k = tl.program_id(1)
    # compute offset
    offset_b = off_b
    offset_k = off_k
    offset_sample = offset_b * k * m + offset_k * m
    offset_sample_out = offset_b * k + offset_k
    offset_lse = offset_b * m
    # mask
    m_mask = tl.arange(0, BLOCK_M) < m

    # 1, 1
    sample_out_block_ptr = Sample_out + offset_sample_out + tl.arange(0, 1)[:, None] * k
    # 1, m
    lse_ptr = Lse + offset_lse + tl.arange(0, BLOCK_M)[None, :]
    # for random
    # 1, 1
    rand_block_ptr = tl.zeros([1, 1], dtype=tl.float32)

    value = -float("inf")

    logits = tl.load(lse_ptr, mask=m_mask[None, :], other=value)
    # use Gumbel Max to sample
    # sample from p1, ..., pk is equivalent to sample
    # argmax {log pi - log(-log(ui))} = argmax {logits - log(-log(ui))}, ui ~ U(0,1)
    # (1, 1)
    u = tl.rand(seed, rand_block_ptr)
    stat = logits - tl.log(-tl.log(u))
    # (1,)
    index = tl.argmax(stat, axis=1)

    # 1, 1
    sample_index_block_ptr = Sample + offset_sample + index[:, None]
    sample_out = tl.load(sample_index_block_ptr)
    tl.store(
        sample_out_block_ptr,
        sample_out.to(sample_out_block_ptr.dtype.element_ty),
    )


def gumbel_multinomial_reduce_triton(sample, lse):
    """
    sample: b k m
    lse: b m
    """
    b, k, m = sample.shape

    def grid(meta):
        return (b, k)

    sample_out = torch.empty((b, k), dtype=torch.int32, device=sample.device)
    seed = 0
    BLOCK_M = next_power_of_two(m)

    _gumbel_multinomial_reduce_triton[grid](
        sample, lse, sample_out, seed, b, k, m, BLOCK_M
    )

    return sample_out.to(torch.int64)


if __name__ == "__main__":
    # unit test
    b = 2
    m = 8
    k = 8
    dtype = torch.float32
    device = torch.cuda.current_device()

    sample = torch.rand(b, m, k, dtype=dtype, device=device)
    lse = torch.randn(b, m, dtype=dtype, device=device)
    sample_out = gumbel_multinomial_reduce_triton(sample, lse)
