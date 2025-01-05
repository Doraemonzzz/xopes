import pytest
import torch

from xopes.ops.linear_cross_entropy import (
    linear_cross_entropy_split_torch,
    linear_cross_entropy_torch,
)
from xopes.utils import get_threshold


def get_params():
    shapes = [
        (512, 1024, 2048),
        (1024, 2048, 4096),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float32]
)  # with fp16, naive implement maybe nan
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("chunk_size", [128, 512])
def test(shape, dtype, reduction, label_smoothing, chunk_size, ignore_index=-100):
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
    do = torch.randn((), dtype=dtype, device=device)

    # Forward
    o_lce_torch = linear_cross_entropy_torch(
        x,
        y,
        W,
        reduction=reduction,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
    )
    o_lce_split = linear_cross_entropy_split_torch(
        x,
        y,
        W,
        reduction=reduction,
        label_smoothing=label_smoothing,
        chunk_size=chunk_size,
        ignore_index=ignore_index,
    )

    # Backward
    o_lce_torch.backward(do, retain_graph=True)
    dx_lce_torch, x.grad = x.grad.clone(), None
    dW_lce_torch, W.grad = W.grad.clone(), None

    o_lce_split.backward(do, retain_graph=True)
    dx_lce_split, x.grad = x.grad.clone(), None
    dW_lce_split, W.grad = W.grad.clone(), None

    atol, rtol = get_threshold(dtype)
    dx_atol, dx_rtol = THRESHOLD_DICT = {
        torch.float32: [5e-2, 1e-2],
        torch.bfloat16: [1e-1, 1e-1],
    }[dtype]

    print(f"o_lce_torch: {o_lce_torch}")
    print(f"o_lce_split: {o_lce_split}")

    # Forward check
    print(
        "o diff max: ",
        torch.abs(o_lce_torch - o_lce_split).max().item(),
    )
    print(
        "o diff norm: ",
        torch.norm(o_lce_torch - o_lce_split).item(),
    )
    assert torch.allclose(o_lce_torch, o_lce_split, atol=atol, rtol=rtol)

    # Backward check
    print(
        "dx diff max: ",
        torch.abs(dx_lce_torch - dx_lce_split).max().item(),
    )
    print(
        "dx diff norm: ",
        torch.norm(dx_lce_torch - dx_lce_split).item(),
    )
    if dtype == torch.float32:
        assert torch.allclose(dx_lce_torch, dx_lce_split, atol=dx_atol, rtol=dx_rtol)

    print(
        "dW diff max: ",
        torch.abs(dW_lce_torch - dW_lce_split).max().item(),
    )
    print(
        "dW diff norm: ",
        torch.norm(dW_lce_torch - dW_lce_split).item(),
    )
    assert torch.allclose(dW_lce_torch, dW_lce_split, atol=atol, rtol=rtol)
