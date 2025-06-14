import pytest
import torch

from xopes.ops.flash_attn.tpa.tpa_decode_torch import (
    tpa_decode_naive_torch,
    tpa_decode_torch,
)
from xopes.ops.flash_attn.tpa.tpa_decode_triton import (
    tpa_decode_parallel_b_triton,
    tpa_decode_parallel_bh_triton,
    tpa_decode_parallel_bn_triton,
)
from xopes.utils import get_threshold


def get_params():
    shapes = [
        # standard shapes
        (2, 256, 16, 16, 128, 64),
        (2, 2048, 16, 16, 128, 64),
        # special seqlen
        (2, 2049, 16, 16, 128, 64),
        (2, 2047, 16, 16, 128, 64),
        # special rank
        (2, 2049, 17, 16, 128, 64),
        (2, 2047, 31, 16, 128, 64),
        # special num heads
        (2, 2049, 17, 31, 128, 64),
        (2, 2047, 31, 17, 128, 64),
        # special head dim
        (2, 2049, 17, 31, 129, 63),
        (2, 2047, 31, 17, 127, 65),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_tpa_decode(shape, dtype):
    # Set random seed for reproducibility
    torch.manual_seed(2024)
    device = torch.device("cuda")

    # Unpack shape parameters
    b, m, h, r, d, e = shape
    n = 1

    # Generate input tensors
    aq = torch.randn((b, n, h, r), dtype=dtype, device=device).requires_grad_()
    ak = torch.randn((b, m, h), dtype=dtype, device=device).requires_grad_()
    av = torch.randn((b, m, h), dtype=dtype, device=device).requires_grad_()
    bq = torch.randn((b, n, r, d), dtype=dtype, device=device).requires_grad_()
    bk = torch.randn((b, m, d), dtype=dtype, device=device).requires_grad_()
    bv = torch.randn((b, m, e), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((b, n, h, e), dtype=dtype, device=device)

    # Optional scale parameters
    scale = d**-0.5
    scale_q = 1 / r
    scale_k = 1.0
    scale_v = 1.0

    ##### Forward pass
    o_naive_torch = tpa_decode_naive_torch(
        aq=aq,
        ak=ak,
        av=av,
        bq=bq,
        bk=bk,
        bv=bv,
        scale=scale,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )

    # PyTorch implementation
    o_torch = tpa_decode_torch(
        aq=aq,
        ak=ak,
        av=av,
        bq=bq,
        bk=bk,
        bv=bv,
        scale=scale,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )

    # Triton implementation
    o_parallel_b_triton = tpa_decode_parallel_b_triton(
        aq=aq,
        ak=ak,
        av=av,
        bq=bq,
        bk=bk,
        bv=bv,
        scale=scale,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )

    o_parallel_bh_triton = tpa_decode_parallel_bh_triton(
        aq=aq,
        ak=ak,
        av=av,
        bq=bq,
        bk=bk,
        bv=bv,
        scale=scale,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )

    o_parallel_bn_triton = tpa_decode_parallel_bn_triton(
        aq=aq,
        ak=ak,
        av=av,
        bq=bq,
        bk=bk,
        bv=bv,
        scale=scale,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )

    # Get tolerance thresholds based on dtype
    atol, rtol = get_threshold(dtype)

    ##### Check forward pass results
    print(
        "o diff max (torch vs naive): ",
        torch.abs(o_torch - o_naive_torch).max().item(),
    )
    print(
        "o diff norm (torch vs naive): ",
        torch.norm(o_torch - o_naive_torch).item(),
    )
    assert torch.allclose(o_torch, o_naive_torch, atol=atol, rtol=rtol)

    print(
        "o diff max (torch vs triton parallel_b): ",
        torch.abs(o_torch - o_parallel_b_triton).max().item(),
    )
    print(
        "o diff norm (torch vs triton parallel_b): ",
        torch.norm(o_torch - o_parallel_b_triton).item(),
    )
    assert torch.allclose(o_torch, o_parallel_b_triton, atol=atol, rtol=rtol)

    print(
        "o diff max (torch vs triton parallel_bh): ",
        torch.abs(o_torch - o_parallel_bh_triton).max().item(),
    )
    print(
        "o diff norm (torch vs triton parallel_bh): ",
        torch.norm(o_torch - o_parallel_bh_triton).item(),
    )
    assert torch.allclose(o_torch, o_parallel_bh_triton, atol=atol, rtol=rtol)

    print(
        "o diff max (torch vs triton parallel_bn): ",
        torch.abs(o_torch - o_parallel_bn_triton).max().item(),
    )
    print(
        "o diff norm (torch vs triton parallel_bn): ",
        torch.norm(o_torch - o_parallel_bn_triton).item(),
    )
    assert torch.allclose(o_torch, o_parallel_bn_triton, atol=atol, rtol=rtol)
