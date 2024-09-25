import torch


def multinomial_torch(x, W, num_samples):
    logits = torch.matmul(x, W)
    prob = torch.softmax(logits, dim=-1)
    print(prob)
    print("bbb", torch.multinomial(prob, num_samples, replacement=True))

    return torch.multinomial(prob, num_samples, replacement=True)
