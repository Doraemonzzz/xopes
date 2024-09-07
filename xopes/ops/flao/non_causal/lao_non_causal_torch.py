import torch


def lao_non_causal_torch(q, k, v, g):
    # q: b h n d
    # k: b h m d
    # v: b h m e
    # g: b h n e
    kv = torch.einsum("... n d, ... n e -> ... d e", k, v)
    qkv = torch.matmul(q, kv)
    o = g * qkv

    return o
