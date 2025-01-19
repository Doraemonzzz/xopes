import pytest
import torch

from xopes.ops.linear_cross_entropy import (
    linear_cross_entropy_torch,
    linear_cross_entropy_triton,
)
from xopes.utils import get_threshold


def get_params():
    shapes = [(512, 1024, 2048), (1024, 2048, 4096), (12288, 1024, 50257)]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32]
)  # with fp16, naive implement maybe nan
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])

# @pytest.mark.parametrize(
#     "dtype", [torch.bfloat16]
# )  # with fp16, naive implement maybe nan
# @pytest.mark.parametrize("reduction", ["mean"])
# @pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test(shape, dtype, reduction, label_smoothing, ignore_index=-100):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, d, v = shape
    b_ignore = b // 2

    # Generate input tensors
    x = torch.randn((b, d), dtype=dtype, device=device).requires_grad_()
    y_ignore = torch.randint(0, v, (b_ignore,), device=device)
    y = torch.randint(0, v, (b - b_ignore,), device=device)
    y = torch.cat([y_ignore, y], dim=0)
    W = torch.randn((v, d), dtype=dtype, device=device).requires_grad_()
    bias = torch.randn((v,), dtype=dtype, device=device).requires_grad_()

    # Forward
    o_lce_torch = linear_cross_entropy_torch(
        x=x,
        y=y,
        W=W,
        bias=bias,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )

    o_lce_triton = linear_cross_entropy_triton(
        x=x,
        y=y,
        W=W,
        bias=bias,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    do = torch.rand_like(o_lce_torch)

    # Backward
    o_lce_torch.backward(do, retain_graph=True)
    dx_lce_torch, x.grad = x.grad.clone(), None
    dW_lce_torch, W.grad = W.grad.clone(), None

    o_lce_triton.backward(do, retain_graph=True)
    dx_lce_triton, x.grad = x.grad.clone(), None
    dW_lce_triton, W.grad = W.grad.clone(), None

    atol, rtol = get_threshold(dtype)
    # dx_atol, dx_rtol = THRESHOLD_DICT = {
    #     torch.float32: [5e-2, 1e-2],
    #     torch.bfloat16: [1e-1, 1e-1],
    # }[dtype]
    dx_atol, dx_rtol = atol, rtol

    print(f"o_lce_torch: {o_lce_torch}")
    print(f"o_lce_triton: {o_lce_triton}")

    # Forward check
    print(
        "o diff max: ",
        torch.abs(o_lce_torch - o_lce_triton).max().item(),
    )
    print(
        "o diff norm: ",
        torch.norm(o_lce_torch - o_lce_triton).item(),
    )
    print(o_lce_torch.dtype, o_lce_triton.dtype)
    assert torch.allclose(o_lce_torch.float(), o_lce_triton, atol=atol, rtol=rtol)

    # Backward check
    print(
        "dx diff max: ",
        torch.abs(dx_lce_torch - dx_lce_triton).max().item(),
    )
    print(
        "dx diff norm: ",
        torch.norm(dx_lce_torch - dx_lce_triton).item(),
    )
    assert torch.allclose(dx_lce_torch, dx_lce_triton, atol=dx_atol, rtol=dx_rtol)

    print(
        "dW diff max: ",
        torch.abs(dW_lce_torch - dW_lce_triton).max().item(),
    )
    print(
        "dW diff norm: ",
        torch.norm(dW_lce_torch - dW_lce_triton).item(),
    )
    assert torch.allclose(dW_lce_torch, dW_lce_triton, atol=atol, rtol=rtol)
