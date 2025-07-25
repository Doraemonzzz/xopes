import os

import pytest
import torch

from xopes.ops.gate_linear import gate_linear_ag, gate_linear_torch, gate_linear_triton
from xopes.utils import get_threshold

os.environ["TRITON_F32_DEFAULT"] = "ieee"


def get_params():
    shapes = [(6, 128), (4, 8, 256), (8, 115)]

    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("act", ["none", "relu", "sigmoid", "silu"])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("use_residual", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test(shape, act, use_bias, use_residual, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")

    # Generate input tensors
    x1 = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    x2 = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    d1 = shape[-1]
    d2 = d1 // 2  # Example output dimension
    weight = torch.randn((d2, d1), dtype=dtype, device=device).requires_grad_()

    bias = None
    if use_bias:
        bias = torch.randn(d2, dtype=dtype, device=device).requires_grad_()

    residual = None
    if use_residual:
        residual_shape = list(shape[:-1]) + [d2]
        residual = torch.randn(
            residual_shape, dtype=dtype, device=device
        ).requires_grad_()

    do = torch.randn(list(shape[:-1]) + [d2], dtype=dtype, device=device)

    # forward
    o_torch = gate_linear_torch(x1, x2, weight, bias, residual, act)
    o_triton = gate_linear_triton(x1, x2, weight, bias, residual, act)
    o_ag = gate_linear_ag(x1, x2, weight, bias, residual, act)
    # backward
    o_torch.backward(do, retain_graph=True)
    dx1_torch, x1.grad = x1.grad.clone(), None
    dx2_torch, x2.grad = x2.grad.clone(), None
    dw_torch, weight.grad = weight.grad.clone(), None
    if use_bias:
        db_torch, bias.grad = bias.grad.clone(), None
    if use_residual:
        dr_torch, residual.grad = residual.grad.clone(), None

    o_triton.backward(do, retain_graph=True)
    dx1_triton, x1.grad = x1.grad.clone(), None
    dx2_triton, x2.grad = x2.grad.clone(), None
    dw_triton, weight.grad = weight.grad.clone(), None
    if use_bias:
        db_triton, bias.grad = bias.grad.clone(), None
    if use_residual:
        dr_triton, residual.grad = residual.grad.clone(), None

    o_ag.backward(do, retain_graph=True)
    dx1_ag, x1.grad = x1.grad.clone(), None
    dx2_ag, x2.grad = x2.grad.clone(), None
    dw_ag, weight.grad = weight.grad.clone(), None
    if use_bias:
        db_ag, bias.grad = bias.grad.clone(), None
    if use_residual:
        dr_ag, residual.grad = residual.grad.clone(), None
    atol, rtol = get_threshold(dtype)

    # forward check
    print("o diff max (Vs triton): ", torch.abs(o_torch - o_triton).max().item())
    print("o diff norm (Vs triton): ", torch.norm(o_torch - o_triton).item())
    assert torch.allclose(o_torch, o_triton, atol=atol, rtol=rtol)

    print("o diff max (Vs ag): ", torch.abs(o_torch - o_ag).max().item())
    print("o diff norm (Vs ag): ", torch.norm(o_torch - o_ag).item())
    assert torch.allclose(o_torch, o_ag, atol=atol, rtol=rtol)

    # backward check
    print("dx1 diff max (Vs triton): ", torch.abs(dx1_torch - dx1_triton).max().item())
    print("dx1 diff norm (Vs triton): ", torch.norm(dx1_torch - dx1_triton).item())
    assert torch.allclose(dx1_torch, dx1_triton, atol=atol, rtol=rtol)

    print("dx1 diff max (Vs ag): ", torch.abs(dx1_torch - dx1_ag).max().item())
    print("dx1 diff norm (Vs ag): ", torch.norm(dx1_torch - dx1_ag).item())
    assert torch.allclose(dx1_torch, dx1_ag, atol=atol, rtol=rtol)

    print("dx2 diff max (Vs triton): ", torch.abs(dx2_torch - dx2_triton).max().item())
    print("dx2 diff norm (Vs triton): ", torch.norm(dx2_torch - dx2_triton).item())
    assert torch.allclose(dx2_torch, dx2_triton, atol=atol, rtol=rtol)

    print("dx2 diff max (Vs ag): ", torch.abs(dx2_torch - dx2_ag).max().item())
    print("dx2 diff norm (Vs ag): ", torch.norm(dx2_torch - dx2_ag).item())
    assert torch.allclose(dx2_torch, dx2_ag, atol=atol, rtol=rtol)

    print("dw diff max (Vs triton): ", torch.abs(dw_torch - dw_triton).max().item())
    print("dw diff norm (Vs triton): ", torch.norm(dw_torch - dw_triton).item())
    assert torch.allclose(dw_torch, dw_triton, atol=atol, rtol=rtol)

    print("dw diff max (Vs ag): ", torch.abs(dw_torch - dw_ag).max().item())
    print("dw diff norm (Vs ag): ", torch.norm(dw_torch - dw_ag).item())
    assert torch.allclose(dw_torch, dw_ag, atol=atol, rtol=rtol)

    if use_bias:
        print("db diff max (Vs triton): ", torch.abs(db_torch - db_triton).max().item())
        print("db diff norm (Vs triton): ", torch.norm(db_torch - db_triton).item())
        assert torch.allclose(db_torch, db_triton, atol=atol, rtol=rtol)

        print("db diff max (Vs ag): ", torch.abs(db_torch - db_ag).max().item())
        print("db diff norm (Vs ag): ", torch.norm(db_torch - db_ag).item())
        assert torch.allclose(db_torch, db_ag, atol=atol, rtol=rtol)

    if use_residual:
        print("dr diff max (Vs triton): ", torch.abs(dr_torch - dr_triton).max().item())
        print("dr diff norm (Vs triton): ", torch.norm(dr_torch - dr_triton).item())
        assert torch.allclose(dr_torch, dr_triton, atol=atol, rtol=rtol)

        print("dr diff max (Vs ag): ", torch.abs(dr_torch - dr_ag).max().item())
        print("dr diff norm (Vs ag): ", torch.norm(dr_torch - dr_ag).item())
        assert torch.allclose(dr_torch, dr_ag, atol=atol, rtol=rtol)
