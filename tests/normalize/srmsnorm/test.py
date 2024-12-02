import pytest
import torch

from xopes.ops.normalize.srmsnorm import srmsnorm_torch, srmsnorm_triton
from xopes.utils import get_threshold


def get_params():
    shape = [
        (6, 128, 64),
        (6, 256, 33),
    ]

    return shape


@pytest.mark.parametrize("shape", get_params())
# @pytest.mark.parametrize("scale", [False])
@pytest.mark.parametrize("scale", [True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(shape, scale, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, n, d = shape
    x = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((b, n, d), dtype=dtype, device=device)

    # forward
    o_srmsnorm_torch = srmsnorm_torch(x, scale=scale)
    o_srmsnorm_triton = srmsnorm_triton(x, scale=scale)

    # backward
    o_srmsnorm_torch.backward(do, retain_graph=True)
    dx_srmsnorm_torch, x.grad = x.grad.clone(), None

    o_srmsnorm_triton.backward(do, retain_graph=True)
    dx_srmsnorm_triton, x.grad = x.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    assert torch.allclose(
        o_srmsnorm_torch, o_srmsnorm_triton, atol=atol, rtol=rtol
    ), f"o diff: {torch.abs(o_srmsnorm_torch - o_srmsnorm_triton).max().item()}"

    assert torch.allclose(
        dx_srmsnorm_torch, dx_srmsnorm_triton, atol=atol, rtol=rtol
    ), f"o diff: {torch.abs(dx_srmsnorm_torch - dx_srmsnorm_triton).max().item()}"
