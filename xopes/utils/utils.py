import functools
import itertools
import math
from functools import reduce
from operator import mul

import torch
import triton
from einops import pack as pack_
from einops import unpack as unpack_

from .constant import ACT_SET, THRESHOLD_DICT, XOPES_DEBUG


def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(
            *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
            **{
                k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
                for k, v in kwargs.items()
            },
        )

    return wrapper


def max_power_of_2_divisor(n):
    d = 2
    while n % d == 0:
        d *= 2

    if not (n % d == 0):
        d /= 2

    if d <= 32:
        d = 32

    d = int(d)

    return d


def next_power_of_two(x):
    return 2 ** (math.ceil(math.log(x, 2)))


def last_power_of_two(x):
    return 2 ** (math.floor(math.log(x, 2)))


def pack(tensors, pattern):
    if not isinstance(tensors, list):
        tensors = [tensors]
        is_list = False
    else:
        is_list = True

    return *pack_(tensors, pattern), is_list


def unpack(tensor, packed_shapes, pattern, is_list):
    output = unpack_(tensor, packed_shapes, pattern)

    if not is_list:
        output = output[0]

    return output


def generate_configs(input_dict):
    num_stages_list = input_dict.pop("num_stages", [2])
    num_warps_list = input_dict.pop("num_warps", [4])

    # Extract keys and values from the input dictionary
    keys = list(input_dict.keys())
    values = list(input_dict.values())

    # Generate the Cartesian product of the values
    combinations = list(itertools.product(*values))

    # Create a list of dictionaries from the combinations
    results = [{keys[i]: combo[i] for i in range(len(keys))} for combo in combinations]

    configs = []
    for num_stages in num_stages_list:
        for num_warps in num_warps_list:
            for config in results:
                configs.append(
                    triton.Config(config, num_stages=num_stages, num_warps=num_warps)
                )

    # we only need one config for debug
    if XOPES_DEBUG:
        return configs[:1]
    else:
        return configs


def filter_configs(array):
    # we only need one config for debug
    if XOPES_DEBUG:
        return array[:1]
    else:
        return array


def get_threshold(dtype):
    assert dtype in [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ], "dtype {dtype} not supported"
    return THRESHOLD_DICT[dtype]


def get_memory(device):
    mb_used = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    torch.cuda.reset_peak_memory_stats(device)

    return mb_used


def identity(x, **kwargs):
    return x


def identity_fwd(x, **kwargs):
    return x


def identity_bwd(x, do, **kwargs):
    return do


def is_act_valid(act):
    assert act in ACT_SET, f"act {act} not supported"


def is_op_valid(op):
    assert op in ["add", "mul", "sub", "div"], f"op {op} not supported"


def is_dim_valid(shape1, shape2):
    assert len(shape1) >= len(
        shape2
    ), "shape1 must have more dimensions or equal to shape2"
    for i in range(len(shape2)):
        if shape1[i] != shape2[i]:
            return False
    return True


def prod(shape, start_dim=0, end_dim=None):
    """
    Calculate the product of dimensions in a tensor shape

    Args:
        shape: The shape to calculate the product (tuple, list or torch.Size)
        start_dim: Starting dimension (inclusive)
        end_dim: Ending dimension (inclusive), None means until the last dimension

    Returns:
        Product of the dimensions
    """
    if end_dim is None:
        end_dim = len(shape) - 1

    # Ensure dimensions are within valid range
    start_dim = max(0, start_dim)
    end_dim = min(len(shape) - 1, end_dim)

    # Extract dimensions for product calculation
    dims = shape[start_dim : end_dim + 1]

    # Return 1 if the dimension list is empty
    if not dims:
        return 1

    return reduce(mul, dims, 1)
