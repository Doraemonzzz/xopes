import torch


@torch.no_grad()
def parallel_gumbel_multinomial_torch(
    x, W, num_samples=1, lse=None, output_lse=False, top_k=-1, block_size=1024
):
    d, V = W.shape
    b = x.shape[0]
    (V + block_size - 1) // block_size
    logits = torch.matmul(x, W)

    lse_out = None
    if output_lse:
        lse_out = torch.logsumexp(logits, dim=-1, keepdim=True)

    q = torch.empty((b, V, num_samples), dtype=x.dtype, device=x.device).exponential_(1)
    stat = logits.unsqueeze(-1) - q
    sample = torch.argmax(stat, dim=-2).to(dtype=torch.int)

    return sample, lse_out


if __name__ == "__main__":
    # unit test
    b = 2
    d = 2048
    V = 4096
    num_samples = 16

    x = torch.randn(b, d)
    W = torch.randn(d, V)
    sample, lse = parallel_gumbel_multinomial_torch(x, W, num_samples)
