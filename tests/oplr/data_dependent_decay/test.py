import pytest
import torch
import torch.nn.functional as F

from xopes.ops.out_product_linear_recurrence import (
    oplr_data_dependent_decay_torch,
    oplr_data_dependent_decay_triton,
)
from xopes.utils import get_threshold


def get_params():
    shapes = [
        (2, 128, 64, 64),  # (B, N, D, E)
        (4, 256, 128, 128),
        (1, 512, 32, 32),
    ]
    # shapes = [(2, 8, 64, 64)]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_log_decay", [True, False])
# @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("use_log_decay", [True])
@pytest.mark.parametrize("dtype", [torch.float32])
def test(shape, use_log_decay, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")

    # Unpack shape parameters
    b, n, d, e = shape

    # Prepare input data
    xv = torch.randn((b, n, e), dtype=dtype, device=device).requires_grad_()
    xk = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()

    if use_log_decay:
        log_decay = F.logsigmoid(
            torch.randn((b, n, d), dtype=dtype, device=device)
        ).requires_grad_(True)
    else:
        xk = torch.sigmoid(xk)  # Ensure xk <= 1
        log_decay = None

    # Prepare gradient
    do = torch.randn((b, n, d, e), dtype=dtype, device=device)

    # Forward pass
    o_torch = oplr_data_dependent_decay_torch(xk, xv, log_decay)
    o_triton = oplr_data_dependent_decay_triton(xk, xv, log_decay)

    # # Backward pass
    # o_torch.backward(do, retain_graph=True)
    # dxk_torch, xk.grad = xk.grad.clone(), None
    # dxv_torch, xv.grad = xv.grad.clone(), None
    # if use_log_decay:
    #     dlog_decay_torch, log_decay.grad = log_decay.grad.clone(), None

    # o_triton.backward(do, retain_graph=True)
    # dxk_triton, xk.grad = xk.grad.clone(), None
    # dxv_triton, xv.grad = xv.grad.clone(), None
    # if use_log_decay:
    #     dlog_decay_triton, log_decay.grad = log_decay.grad.clone(), None

    # Get error thresholds
    atol, rtol = get_threshold(dtype)

    # Check forward pass results
    print(torch.norm((o_torch - o_triton)[:, 1]).item())
    print("o diff max:", torch.abs(o_torch - o_triton).max().item())
    print("o diff norm:", torch.norm(o_torch - o_triton).item())
    assert torch.allclose(o_torch, o_triton, atol=atol, rtol=rtol)

    # # Check backward pass results
    # print("dxk diff max:", torch.abs(dxk_torch - dxk_triton).max().item())
    # print("dxk diff norm:", torch.norm(dxk_torch - dxk_triton).item())
    # assert torch.allclose(dxk_torch, dxk_triton, atol=atol, rtol=rtol)

    # print("dxv diff max:", torch.abs(dxv_torch - dxv_triton).max().item())
    # print("dxv diff norm:", torch.norm(dxv_torch - dxv_triton).item())
    # assert torch.allclose(dxv_torch, dxv_triton, atol=atol, rtol=rtol)

    # if use_log_decay:
    #     print(
    #         "dlog_decay diff max:",
    #         torch.abs(dlog_decay_torch - dlog_decay_triton).max().item(),
    #     )
    #     print(
    #         "dlog_decay diff norm:",
    #         torch.norm(dlog_decay_torch - dlog_decay_triton).item(),
    #     )
    #     assert torch.allclose(dlog_decay_torch, dlog_decay_triton, atol=atol, rtol=rtol)
