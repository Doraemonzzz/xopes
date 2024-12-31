import pytest
import torch

from xopes.ops.normalize import normalize_torch, normalize_triton
from xopes.utils import get_threshold


def get_params():
    shape = [(6, 128), (4, 8, 256)]

    return shape


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("num_groups", [1, 4])
@pytest.mark.parametrize("use_mean", [True, False])
@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("use_residual", [True, False])
@pytest.mark.parametrize("c", [1, 16])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(
    shape, num_groups, use_mean, use_weight, use_bias, use_residual, c, eps, dtype
):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    d = shape[-1]
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    do = torch.randn(shape, dtype=dtype, device=device)

    if use_weight:
        weight = torch.randn((d,), dtype=dtype, device=device).requires_grad_()
    else:
        weight = None

    if use_bias:
        bias = torch.randn((d,), dtype=dtype, device=device).requires_grad_()
    else:
        bias = None

    if use_residual:
        residual = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    else:
        residual = None

    # forward
    o_normalize_torch = normalize_torch(
        x,
        weight=weight,
        bias=bias,
        residual=residual,
        c=c,
        eps=eps,
        use_mean=use_mean,
        num_groups=num_groups,
    )
    o_normalize_triton = normalize_triton(
        x,
        weight=weight,
        bias=bias,
        residual=residual,
        c=c,
        eps=eps,
        use_mean=use_mean,
        num_groups=num_groups,
    )

    # backward
    o_normalize_torch.backward(do, retain_graph=True)
    dx_normalize_torch, x.grad = x.grad.clone(), None
    if use_weight:
        dw_normalize_torch, weight.grad = weight.grad.clone(), None
    else:
        dw_normalize_torch = None

    if use_bias:
        db_normalize_torch, bias.grad = bias.grad.clone(), None
    else:
        db_normalize_torch = None

    if use_residual:
        dr_normalize_torch, residual.grad = residual.grad.clone(), None
    else:
        dr_normalize_torch = None

    o_normalize_triton.backward(do, retain_graph=True)
    dx_normalize_triton, x.grad = x.grad.clone(), None
    if use_weight:
        dw_normalize_triton, weight.grad = weight.grad.clone(), None
    else:
        dw_normalize_triton = None

    if use_bias:
        db_normalize_triton, bias.grad = bias.grad.clone(), None
    else:
        db_normalize_triton = None

    if use_residual:
        dr_normalize_triton, residual.grad = residual.grad.clone(), None
    else:
        dr_normalize_triton = None

    atol, rtol = get_threshold(dtype)

    ##### fwd
    print(
        "o diff max: ", torch.abs(o_normalize_torch - o_normalize_triton).max().item()
    )
    print("o diff norm: ", torch.norm(o_normalize_torch - o_normalize_triton).item())
    assert torch.allclose(o_normalize_torch, o_normalize_triton, atol=atol, rtol=rtol)

    ##### bwd
    print(
        "dx diff max: ",
        torch.abs(dx_normalize_torch - dx_normalize_triton).max().item(),
    )
    print("dx diff norm: ", torch.norm(dx_normalize_torch - dx_normalize_triton).item())
    assert torch.allclose(dx_normalize_torch, dx_normalize_triton, atol=atol, rtol=rtol)

    if use_weight:
        print(
            "dw diff max: ",
            torch.abs(dw_normalize_torch - dw_normalize_triton).max().item(),
        )
        print(
            "dw diff norm: ",
            torch.norm(dw_normalize_torch - dw_normalize_triton).item(),
        )
        assert torch.allclose(
            dw_normalize_torch, dw_normalize_triton, atol=atol, rtol=rtol
        )

    if use_bias:
        print(
            "db diff max: ",
            torch.abs(db_normalize_torch - db_normalize_triton).max().item(),
        )
        print(
            "db diff norm: ",
            torch.norm(db_normalize_torch - db_normalize_triton).item(),
        )
        assert torch.allclose(
            db_normalize_torch, db_normalize_triton, atol=atol, rtol=rtol
        )

    if use_residual:
        print(
            "dr diff max: ",
            torch.abs(dr_normalize_torch - dr_normalize_triton).max().item(),
        )
        print(
            "dr diff norm: ",
            torch.norm(dr_normalize_torch - dr_normalize_triton).item(),
        )
        assert torch.allclose(
            dr_normalize_torch, dr_normalize_triton, atol=atol, rtol=rtol
        )
