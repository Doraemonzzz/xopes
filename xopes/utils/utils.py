import functools

import torch
from einops import pack as pack_
from einops import unpack as unpack_


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
