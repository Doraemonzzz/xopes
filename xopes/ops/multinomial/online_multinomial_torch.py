import torch
import torch.nn.functional as F


@torch.no_grad()
def online_multinomial_torch(
    x, W, num_samples, lse=None, max_value=None, block_size=1024
):
    d, V = W.shape
    b = x.shape[0]
    m = (V + block_size - 1) // block_size
    if lse == None:
        lse = torch.full((b, 1), -float("inf"), device=x.device)

    if max_value == None:
        max_value = torch.full((b, 1), -float("inf"), device=x.device)

    sample = torch.empty((b, num_samples), dtype=torch.int64, device=x.device)

    for i in range(m):
        start = i * block_size
        end = min((i + 1) * block_size, V)
        l = (i + 1) * block_size - end
        weight = W[:, start:end]
        if l > 0:
            weight = F.pad(weight, (0, l), value=0)

        # sample current block
        logits_curr = torch.matmul(x, weight)
        lse_curr = torch.logsumexp(logits_curr, dim=-1, keepdim=True)
        max_value_curr = torch.max(logits_curr, dim=-1, keepdim=True).values.to(
            max_value.dtype
        )
        prob_curr = torch.exp(logits_curr - lse_curr)
        sample_curr = start + torch.multinomial(
            prob_curr, num_samples, replacement=True
        )

        # sample by binomial
        # m = max(ma, mb)
        # lse(a, b) = log(exp(lse(a)) + exp(lse(b))) = log(exp(lse(a) - m) + exp(lse(b) - m)) + m
        mask = max_value_curr > max_value
        max_value[mask] = max_value_curr[mask]
        lse = (
            torch.log(torch.exp(lse - max_value) + torch.exp(lse_curr - max_value))
            + max_value
        )
        prob = torch.exp(lse_curr - lse)

        # x = 1: sample_curr
        # x = 0: sample
        index = (torch.rand(b, num_samples, device=x.device) < prob).to(torch.int64)
        mask = index == 1
        sample[mask] = sample_curr[mask]
    print(sample)
    return sample


if __name__ == "__main__":
    # unit test
    b = 2
    d = 2048
    V = 4096
    num_samples = 16

    x = torch.randn(b, d)
    W = torch.randn(d, V)
    sample = online_multinomial_torch(x, W, num_samples)
