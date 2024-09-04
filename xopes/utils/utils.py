import functools
import itertools

import torch
import triton
from einops import pack as pack_
from einops import unpack as unpack_

from .constant import THRESHOLD_DICT, XOPES_DEBUG


def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(
            ctx,
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


def get_threshold(dtype):
    assert dtype in [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ], "dtype {dtype} not supported"
    return THRESHOLD_DICT[dtype]
