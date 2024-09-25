import torch


def multinomial_torch(x, W, num_samples):
    logits = torch.matmul(x, W)
    prob = torch.softmax(logits, dim=-1)

    return torch.multinomial(prob, num_samples, replacement=True)
