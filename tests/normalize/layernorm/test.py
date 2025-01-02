import pytest
import torch

from xopes.ops.normalize import layernorm_torch, layernorm_triton
from xopes.utils import get_threshold


def get_params():
    shape = [(6, 128), (4, 8, 256)]

    return shape


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_residual", [True, False])
@pytest.mark.parametrize("c", [1, 16])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(shape, use_residual, c, eps, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    d = shape[-1]
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    do = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((d,), dtype=dtype, device=device).requires_grad_()
    bias = torch.randn((d,), dtype=dtype, device=device).requires_grad_()

    if use_residual:
        residual = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    else:
        residual = None

    # forward
    o_layernorm_torch, o_update_residual_torch = layernorm_torch(
        x,
        weight=weight,
        bias=bias,
        dim=d,
        eps=eps,
        residual=residual,
    )
    if use_residual:
        o_layernorm_torch = o_layernorm_torch + o_update_residual_torch

    o_layernorm_triton, o_update_residual_triton = layernorm_triton(
        x,
        weight=weight,
        bias=bias,
        dim=d,
        eps=eps,
        residual=residual,
    )
    if use_residual:
        o_layernorm_triton = o_layernorm_triton + o_update_residual_triton

    # backward
    o_layernorm_torch.backward(do, retain_graph=True)
    dx_layernorm_torch, x.grad = x.grad.clone(), None
    dw_layernorm_torch, weight.grad = weight.grad.clone(), None
    db_layernorm_torch, bias.grad = bias.grad.clone(), None

    if use_residual:
        dr_layernorm_torch, residual.grad = residual.grad.clone(), None
    else:
        dr_layernorm_torch = None

    o_layernorm_triton.backward(do, retain_graph=True)
    dx_layernorm_triton, x.grad = x.grad.clone(), None
    dw_layernorm_triton, weight.grad = weight.grad.clone(), None
    db_layernorm_triton, bias.grad = bias.grad.clone(), None

    if use_residual:
        dr_layernorm_triton, residual.grad = residual.grad.clone(), None
    else:
        dr_layernorm_triton = None

    atol, rtol = get_threshold(dtype)

    ##### fwd
    print(
        "o diff max: ", torch.abs(o_layernorm_torch - o_layernorm_triton).max().item()
    )
    print("o diff norm: ", torch.norm(o_layernorm_torch - o_layernorm_triton).item())
    assert torch.allclose(o_layernorm_torch, o_layernorm_triton, atol=atol, rtol=rtol)

    ##### bwd
    print(
        "dx diff max: ",
        torch.abs(dx_layernorm_torch - dx_layernorm_triton).max().item(),
    )
    print("dx diff norm: ", torch.norm(dx_layernorm_torch - dx_layernorm_triton).item())
    assert torch.allclose(dx_layernorm_torch, dx_layernorm_triton, atol=atol, rtol=rtol)

    print(
        "dw diff max: ",
        torch.abs(dw_layernorm_torch - dw_layernorm_triton).max().item(),
    )
    print(
        "dw diff norm: ",
        torch.norm(dw_layernorm_torch - dw_layernorm_triton).item(),
    )
    assert torch.allclose(dw_layernorm_torch, dw_layernorm_triton, atol=atol, rtol=rtol)

    print(
        "db diff max: ",
        torch.abs(db_layernorm_torch - db_layernorm_triton).max().item(),
    )
    print(
        "db diff norm: ",
        torch.norm(db_layernorm_torch - db_layernorm_triton).item(),
    )
    assert torch.allclose(db_layernorm_torch, db_layernorm_triton, atol=atol, rtol=rtol)

    if use_residual:
        print(
            "dr diff max: ",
            torch.abs(dr_layernorm_torch - dr_layernorm_triton).max().item(),
        )
        print(
            "dr diff norm: ",
            torch.norm(dr_layernorm_torch - dr_layernorm_triton).item(),
        )
        assert torch.allclose(
            dr_layernorm_torch, dr_layernorm_triton, atol=atol, rtol=rtol
        )
