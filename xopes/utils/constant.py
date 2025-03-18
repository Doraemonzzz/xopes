import os
from functools import lru_cache

import torch
import triton

XOPES_DEBUG = eval(os.environ.get("XOPES_DEBUG", default="False"))
if XOPES_DEBUG:
    XOPES_DTYPE = "ieee"
else:
    XOPES_DTYPE = "tf32"

HEAD_DIM = 128


THRESHOLD_DICT = {
    torch.float32: [1e-2, 1e-2],
    torch.float16: [5e-2, 1e-2],
    torch.bfloat16: [5e-2, 1e-2],
}

ACT_SET = set(
    [
        "relu",
        "sigmoid",
        "silu",
        "none",
        "softmax",
    ]
)


@lru_cache(maxsize=None)
def get_sm_count():
    info = triton.runtime.driver.active.utils.get_device_properties(0)
    return info["multiprocessor_count"]


SM_COUNT = get_sm_count()
