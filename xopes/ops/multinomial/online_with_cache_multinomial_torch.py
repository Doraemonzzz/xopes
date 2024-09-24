import torch
import torch.nn.functional as F
from einops import pack, rearrange, unpack


@torch.no_grad()
def online_with_cache_multinomial_torch(x, W, num_samples, block_size=128):
    d, V = W.shape
    m = (V + block_size - 1) // block_size
    lse_list = []
    sample_list = []

    for i in range(m):
        start = i * block_size
        end = min((i + 1) * block_size, V)
        l = (i + 1) * block_size - end
        weight = W[:, start:end]
        if l > 0:
            weight = F.pad(weight, (0, l), value=0)
        logits = torch.matmul(x, weight)
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)
        prob = torch.exp(logits - lse)
        sample = start + torch.multinomial(prob, num_samples, replacement=True)

        sample_list.append(sample)
        lse_list.append(lse)

    # b 1 g
    lse = torch.stack(lse_list, dim=-1).unsqueeze(-2)
    # b k g
    sample = torch.stack(sample_list, dim=-1)
    lse, ps = pack([lse], "* m")
    sample, ps = pack([sample], "* m")
    lse_max = torch.max(lse, dim=-1, keepdim=True).values
    prob = torch.exp(lse - lse_max)
    # b k
    index = torch.multinomial(prob, num_samples, replacement=True)
    index = rearrange(index, "b m -> (b m) 1")

    # sample by group
    sample = torch.gather(sample, dim=1, index=index).squeeze(-1)
    sample = unpack(sample, ps, "*")[0]

    return sample


if __name__ == "__main__":
    # unit test
    b = 2
    d = 2048
    V = 512
    num_samples = 16

    x = torch.randn(b, d)
    W = torch.randn(d, V)
    sample = online_with_cache_multinomial_torch(x, W, num_samples)
