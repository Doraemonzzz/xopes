import torch


def multinomial_torch(x, W, num_samples, top_k=-1):
    logits = torch.matmul(x, W)

    if top_k != -1:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("inf"), logits)

    prob = torch.softmax(logits, dim=-1)

    return torch.multinomial(prob, num_samples, replacement=True)
