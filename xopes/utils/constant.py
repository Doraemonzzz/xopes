import os

import torch

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


def get_gpu_sm_count(device_id=None):
    """
    Get the SM (Streaming Multiprocessor) count for a specific GPU

    Args:
        device_id: GPU device ID (None for current device)

    Returns:
        Dictionary with device info and SM count
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    # Set device_id to current device if not specified
    if device_id is None:
        device_id = torch.cuda.current_device()

    # SM count mapping for common GPU models
    sm_count_map = {
        # Data Center GPUs - Hopper architecture
        "H100 SXM": 132,
        "H100 PCIe": 114,
        "H100": 132,  # Default to SXM version if not specified
        "H800": 132,
        # Data Center GPUs - Ampere architecture
        "A100": 108,
        "A100-SXM": 108,
        "A100-PCIe": 108,
        "A800": 108,
        "A40": 84,
        "A30": 84,
        "A10": 72,
        # Consumer GPUs - Ada Lovelace
        "RTX 4090": 128,
        "RTX 4080": 76,
        "RTX 4070": 60,
        "RTX 4060": 34,
        # Consumer GPUs - Ampere
        "RTX 3090": 82,
        "RTX 3080": 68,
        "RTX 3070": 46,
        "RTX 3060": 28,
        # Consumer GPUs - Turing
        "RTX 2080": 46,
        "RTX 2070": 36,
        "RTX 2060": 30,
    }

    try:
        # Get device name
        device_name = torch.cuda.get_device_name(device_id)
        sm_count = -1

        # Find matching GPU model
        for model, count in sm_count_map.items():
            if model in device_name:
                sm_count = count
                break

        return {
            "device_id": device_id,
            "name": device_name,
            "sm_count": sm_count,
            "compute_capability": f"{torch.cuda.get_device_capability(device_id)[0]}.{torch.cuda.get_device_capability(device_id)[1]}",
        }

    except Exception as e:
        return {"error": f"Error getting GPU information: {str(e)}"}


_SM_COUNT_CACHE = None


def get_sm_count():
    global _SM_COUNT_CACHE
    if _SM_COUNT_CACHE is None:
        gpu_info = get_gpu_sm_count()
        _SM_COUNT_CACHE = gpu_info.get("sm_count", -1)
    return _SM_COUNT_CACHE


SM_COUNT = get_sm_count()
