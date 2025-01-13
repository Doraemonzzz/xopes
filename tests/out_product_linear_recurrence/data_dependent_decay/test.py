import pytest
import torch
import torch.nn.functional as F

from xopes.ops.out_product_linear_recurrence import (
    oplr_ddd_ag_torch,
    oplr_ddd_torch,
    oplr_ddd_triton,
    oplr_ddd_ya_ag_torch,
)
from xopes.utils import get_threshold


def get_params():
    shapes = [
        (2, 128, 64, 64),
        # (2, 32, 64, 48),
        # (2, 32, 64, 32),
        # (1, 4, 4, 16),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
# @pytest.mark.parametrize("use_log_decay", [True, False])
# @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_log_decay", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test(shape, use_log_decay, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")

    # Unpack shape parameters
    b, n, d, e = shape

    # Prepare input data
    xv = torch.randn((b, n, e), dtype=dtype, device=device).requires_grad_()
    xk = torch.randn((b, n, d), dtype=dtype, device=device)

    if use_log_decay:
        xk.requires_grad_()
        log_decay = F.logsigmoid(
            torch.randn((b, n, d), dtype=dtype, device=device)
        ).requires_grad_()
    else:
        xk = torch.sigmoid(xk).requires_grad_()  # Ensure xk <= 1
        log_decay = None

    # Prepare gradient
    do = torch.randn((b, n, d, e), dtype=dtype, device=device)

    # Forward pass
    o_torch = oplr_ddd_torch(xk, xv, log_decay)
    o_auto_grad_torch = oplr_ddd_ag_torch(xk, xv, log_decay)
    o_ya_auto_grad_torch = oplr_ddd_ya_ag_torch(
        xk, xv, log_decay
    )  # only test dxk and dlog_decay since we only change them
    o_triton = oplr_ddd_triton(xk, xv, log_decay)

    # Backward pass
    o_torch.backward(do, retain_graph=True)
    dxk_torch, xk.grad = xk.grad.clone(), None
    dxv_torch, xv.grad = xv.grad.clone(), None
    if use_log_decay:
        dlog_decay_torch, log_decay.grad = log_decay.grad.clone(), None

    o_auto_grad_torch.backward(do, retain_graph=True)
    dxk_auto_grad_torch, xk.grad = xk.grad.clone(), None
    dxv_auto_grad_torch, xv.grad = xv.grad.clone(), None
    if use_log_decay:
        dlog_decay_auto_grad_torch, log_decay.grad = log_decay.grad.clone(), None

    o_ya_auto_grad_torch.backward(do, retain_graph=True)
    dxk_ya_auto_grad_torch, xk.grad = xk.grad.clone(), None
    dxv_ya_auto_grad_torch, xv.grad = xv.grad.clone(), None
    if use_log_decay:
        dlog_decay_ya_auto_grad_torch, log_decay.grad = log_decay.grad.clone(), None

    o_triton.backward(do, retain_graph=True)
    dxk_triton, xk.grad = xk.grad.clone(), None
    dxv_triton, xv.grad = xv.grad.clone(), None
    if use_log_decay:
        dlog_decay_triton, log_decay.grad = log_decay.grad.clone(), None

    # Get error thresholds
    atol, rtol = get_threshold(dtype)

    # Check forward pass results
    ##### o
    print("o diff max (Vs triton):", torch.abs(o_torch - o_triton).max().item())
    print("o diff norm (Vs triton):", torch.norm(o_torch - o_triton).item())
    assert torch.allclose(o_torch, o_triton, atol=atol, rtol=rtol)

    print(
        "o diff max (Vs auto grad torch):",
        torch.abs(o_torch - o_auto_grad_torch).max().item(),
    )
    print(
        "o diff norm (Vs auto grad torch):",
        torch.norm(o_torch - o_auto_grad_torch).item(),
    )
    assert torch.allclose(o_torch, o_auto_grad_torch, atol=atol, rtol=rtol)

    # Check backward pass results
    ##### dxk
    print("dxk diff max (Vs triton):", torch.abs(dxk_torch - dxk_triton).max().item())
    print("dxk diff norm (Vs triton):", torch.norm(dxk_torch - dxk_triton).item())
    assert torch.allclose(dxk_torch, dxk_triton, atol=atol, rtol=rtol)

    print(
        "dxk diff max (Vs auto grad torch):",
        torch.abs(dxk_torch - dxk_auto_grad_torch).max().item(),
    )
    print(
        "dxk diff norm (Vs auto grad torch):",
        torch.norm(dxk_torch - dxk_auto_grad_torch).item(),
    )
    assert torch.allclose(dxk_torch, dxk_auto_grad_torch, atol=atol, rtol=rtol)

    print(
        "dxk diff max (Vs ya auto grad torch):",
        torch.abs(dxk_torch - dxk_ya_auto_grad_torch).max().item(),
    )
    print(
        "dxk diff norm (Vs ya auto grad torch):",
        torch.norm(dxk_torch - dxk_ya_auto_grad_torch).item(),
    )
    assert torch.allclose(dxk_torch, dxk_ya_auto_grad_torch, atol=atol, rtol=rtol)

    ##### dxv
    print(
        "dxv diff max (Vs auto grad torch):",
        torch.abs(dxv_torch - dxv_auto_grad_torch).max().item(),
    )
    print(
        "dxv diff norm (Vs auto grad torch):",
        torch.norm(dxv_torch - dxv_auto_grad_torch).item(),
    )
    assert torch.allclose(dxv_torch, dxv_auto_grad_torch, atol=atol, rtol=rtol)

    print("dxv diff max (Vs triton):", torch.abs(dxv_torch - dxv_triton).max().item())
    print("dxv diff norm (Vs triton):", torch.norm(dxv_torch - dxv_triton).item())
    assert torch.allclose(dxv_torch, dxv_triton, atol=atol, rtol=rtol)

    ##### dlog_decay
    if use_log_decay:
        print(
            "dlog_decay diff max (Vs auto grad torch):",
            torch.abs(dlog_decay_torch - dlog_decay_auto_grad_torch).max().item(),
        )
        print(
            "dlog_decay diff norm (Vs auto grad torch):",
            torch.norm(dlog_decay_torch - dlog_decay_auto_grad_torch).item(),
        )
        assert torch.allclose(
            dlog_decay_torch, dlog_decay_auto_grad_torch, atol=atol, rtol=rtol
        )

        print(
            "dlog_decay diff max (Vs ya auto grad torch):",
            torch.abs(dlog_decay_torch - dlog_decay_ya_auto_grad_torch).max().item(),
        )
        print(
            "dlog_decay diff norm (Vs ya auto grad torch):",
            torch.norm(dlog_decay_torch - dlog_decay_ya_auto_grad_torch).item(),
        )
        assert torch.allclose(
            dlog_decay_torch, dlog_decay_ya_auto_grad_torch, atol=atol, rtol=rtol
        )

        print(
            "dlog_decay diff max (Vs triton):",
            torch.abs(dlog_decay_torch - dlog_decay_triton).max().item(),
        )
        print(
            "dlog_decay diff norm (Vs triton):",
            torch.norm(dlog_decay_torch - dlog_decay_triton).item(),
        )
        assert torch.allclose(dlog_decay_torch, dlog_decay_triton, atol=atol, rtol=rtol)

    # assert False
