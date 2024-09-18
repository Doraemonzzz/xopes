import pytest
import torch
from einops import pack

from xopes.ops.lrpe.cosine._md import md_lrpe_cosine_torch, md_lrpe_cosine_triton, md_lrpe_cosine_cache_triton
from xopes.utils import get_threshold


def get_params():
    shape = [
        # 1d
        (6, 8, 128, 64),
        (6, 8, 127, 128),
        # 2d
        (6, 8, 32, 32, 64),
        (6, 8, 2, 3, 64),
        # 3d
        (6, 8, 8, 32, 32, 64),
        # 4d
        (6, 8, 8, 8, 8, 8, 64),
    ]

    return shape


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("l", [0, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(shape, l, dtype):
    torch.manual_seed(2024)
    device = torch.cuda.current_device()

    b = shape[0]
    h = shape[1]
    d = shape[-1]
    m = len(shape) - 3
    e = (d + m - 1) // m

    x = torch.randn(shape, dtype=dtype, device=device)
    x, ps_x = pack([x], "b h * d")

    if l > 0:
        token = torch.randn((b, h, l, d), dtype=dtype, device=device)
        x = torch.cat([token, x], dim=-2)
    x = x.requires_grad_()

    theta = torch.randn((h, e), dtype=dtype, device=device)
    shape = shape[:-1] + (shape[-1] * 2,)

    do = torch.randn(shape, dtype=dtype, device=device)
    do, ps_do = pack([do], "b h * d")
    if l > 0:
        do_token = torch.randn((b, h, l, 2 * d), dtype=dtype, device=device)
        do = torch.cat([do_token, do], dim=-2)

    # forward
    o_md_lrpe_cosine_torch = md_lrpe_cosine_torch(x, theta, shape=shape[2:-1], l=l)
    o_md_lrpe_cosine_triton = md_lrpe_cosine_triton(x, theta, shape=shape[2:-1], l=l)
    o_md_lrpe_cosine_cache_triton = md_lrpe_cosine_cache_triton(
        x, theta, shape=shape[2:-1], l=l
    )

    # backward
    o_md_lrpe_cosine_torch.backward(do, retain_graph=True)
    dx_md_lrpe_cosine_torch, x.grad = x.grad.clone(), None

    o_md_lrpe_cosine_triton.backward(do, retain_graph=True)
    dx_md_lrpe_cosine_triton, x.grad = x.grad.clone(), None

    o_md_lrpe_cosine_cache_triton.backward(do, retain_graph=True)
    dx_md_lrpe_cosine_cache_triton, x.grad = x.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    # forward
    assert torch.allclose(
        o_md_lrpe_cosine_torch, o_md_lrpe_cosine_triton, atol=atol, rtol=rtol
    ), f"o diff: {torch.abs(o_md_lrpe_cosine_torch - o_md_lrpe_cosine_triton).max().item()}"
    assert torch.allclose(
        o_md_lrpe_cosine_torch, o_md_lrpe_cosine_cache_triton, atol=atol, rtol=rtol
    ), f"o diff: {torch.abs(o_md_lrpe_cosine_torch - o_md_lrpe_cosine_cache_triton).max().item()}"

    # backward
    assert torch.allclose(
        dx_md_lrpe_cosine_torch, dx_md_lrpe_cosine_triton, atol=atol, rtol=rtol
    ), f"dx diff: {torch.abs(dx_md_lrpe_cosine_torch - dx_md_lrpe_cosine_triton).max().item()}"
    assert torch.allclose(
        dx_md_lrpe_cosine_torch, dx_md_lrpe_cosine_cache_triton, atol=atol, rtol=rtol
    ), f"dx diff: {torch.abs(dx_md_lrpe_cosine_torch - dx_md_lrpe_cosine_cache_triton).max().item()}"
