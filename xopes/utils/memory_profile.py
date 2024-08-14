import torch


def get_memory(device):
    mb_used = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    torch.cuda.reset_peak_memory_stats(device)

    return mb_used
