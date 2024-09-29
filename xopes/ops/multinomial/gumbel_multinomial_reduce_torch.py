import torch


def gumbel_multinomial_reduce_torch(sample, lse, top_k=-1):
    """
    sample: b k m
    lse: b m
    """
    b, k, m = sample.shape

    q = torch.empty((b, k, m), dtype=sample.dtype, device=sample.device).exponential_(1)
    stat = lse.unsqueeze(-2) - q
    sample = torch.argmax(stat, dim=-2).to(dtype=torch.int)

    return sample.to(torch.int64)


if __name__ == "__main__":
    # unit test
    b = 2
    m = 8
    k = 8
    dtype = torch.float32
    device = torch.cuda.current_device()

    sample = torch.rand(b, m, k, dtype=dtype, device=device)
    lse = torch.randn(b, m, dtype=dtype, device=device)
    sample_out = gumbel_multinomial_reduce_torch(sample, lse)
