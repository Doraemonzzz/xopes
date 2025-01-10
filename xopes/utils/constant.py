import os

import torch

XOPES_DEBUG = eval(os.environ.get("XOPES_DEBUG", default="False"))


HEAD_DIM = 128


THRESHOLD_DICT = {
    torch.float32: [1e-2, 1e-2],
    torch.float16: [5e-2, 1e-2],
    torch.bfloat16: [5e-2, 1e-2],
    # torch.bfloat16: [1e-1, 1e-1],
}

ACT_SET = set(
    [
        "relu",
        "sigmoid",
        "silu",
        "none",
        "softmax",
        "softmax_no_cache",
    ]
)
