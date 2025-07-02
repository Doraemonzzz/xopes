import os
from functools import lru_cache

import torch
import triton

HEAD_DIM = 128

MIN_BLOCK = 16

THRESHOLD_DICT = {
    torch.float32: [1e-2, 1e-2],
    torch.float16: [5e-2, 1e-2],
    torch.bfloat16: [5e-2, 5e-2],
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


@lru_cache(maxsize=None)
def get_xopes_debug():
    xopes_debug = eval(os.environ.get("XOPES_DEBUG", default="False"))
    if xopes_debug:
        xopes_dtype = "ieee"
    else:
        xopes_dtype = "tf32"
    return xopes_debug, xopes_dtype


SM_COUNT = get_sm_count()
XOPES_DEBUG, XOPES_DTYPE = get_xopes_debug()
