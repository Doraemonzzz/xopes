import pytest
import torch
import torch.nn.functional as F

from xopes.ops.normalize import group_norm_torch, group_norm_triton
from xopes.utils import get_threshold


def group_norm_torch_official(
    x, weight, bias, dim, eps, residual, return_residual, num_groups
):
    dtype = x.dtype
    x = x.float()

    # Handle residual connection
    if residual is not None:
        x = x + residual.float()
        residual = x.to(dtype)
    else:
        if return_residual:
            residual = x.to(dtype)

    if weight is not None:
        weight = weight.float()
    if bias is not None:
        bias = bias.float()

    x_shape = x.shape
    x_ = F.group_norm(
        x.reshape(-1, x.shape[-1]), num_groups, weight, bias, eps
    ).reshape(x_shape)

    return x_.to(dtype), residual


def get_params():
    shape = [(6, 128), (4, 8, 256), (6, 2048, 768)]

    return shape


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_residual", [True, False])
@pytest.mark.parametrize("return_residual", [True, False])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_groups", [4, 16])
def test(shape, use_residual, return_residual, eps, dtype, num_groups):
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
    o_torch_official = group_norm_torch_official(
        x,
        num_groups=num_groups,
        weight=weight,
        bias=bias,
        dim=d,
        eps=eps,
        residual=residual,
        return_residual=return_residual,
    )

    if isinstance(o_torch_official, tuple):
        o_group_norm_torch_official, o_update_residual_torch_official = o_torch_official
    else:
        o_group_norm_torch_official = o_torch_official
        o_update_residual_torch_official = None

    if use_residual and return_residual:
        o_group_norm_torch_official = (
            o_group_norm_torch_official + o_update_residual_torch_official
        )

    o_torch = group_norm_torch(
        x,
        weight=weight,
        bias=bias,
        dim=d,
        eps=eps,
        residual=residual,
        return_residual=return_residual,
        num_groups=num_groups,
    )

    if isinstance(o_torch, tuple):
        o_group_norm_torch, o_update_residual_torch = o_torch
    else:
        o_group_norm_torch = o_torch
        o_update_residual_torch = None

    if use_residual and return_residual:
        o_group_norm_torch = o_group_norm_torch + o_update_residual_torch

    o_triton = group_norm_triton(
        x,
        weight=weight,
        bias=bias,
        dim=d,
        eps=eps,
        residual=residual,
        return_residual=return_residual,
        num_groups=num_groups,
    )

    if isinstance(o_triton, tuple):
        o_group_norm_triton, o_update_residual_triton = o_triton
    else:
        o_group_norm_triton = o_triton
        o_update_residual_triton = None

    if use_residual and return_residual:
        o_group_norm_triton = o_group_norm_triton + o_update_residual_triton

    # backward
    o_group_norm_torch_official.backward(do, retain_graph=True)
    dx_group_norm_torch_official, x.grad = x.grad.clone(), None
    dw_group_norm_torch_official, weight.grad = weight.grad.clone(), None
    db_group_norm_torch_official, bias.grad = bias.grad.clone(), None

    if use_residual:
        dr_group_norm_torch_official, residual.grad = residual.grad.clone(), None
    else:
        pass

    o_group_norm_torch.backward(do, retain_graph=True)
    dx_group_norm_torch, x.grad = x.grad.clone(), None
    dw_group_norm_torch, weight.grad = weight.grad.clone(), None
    db_group_norm_torch, bias.grad = bias.grad.clone(), None

    if use_residual:
        dr_group_norm_torch, residual.grad = residual.grad.clone(), None
    else:
        dr_group_norm_torch = None

    o_group_norm_triton.backward(do, retain_graph=True)
    dx_group_norm_triton, x.grad = x.grad.clone(), None
    dw_group_norm_triton, weight.grad = weight.grad.clone(), None
    db_group_norm_triton, bias.grad = bias.grad.clone(), None

    if use_residual:
        dr_group_norm_triton, residual.grad = residual.grad.clone(), None
    else:
        dr_group_norm_triton = None

    atol, rtol = get_threshold(dtype)

    ##### fwd
    print(
        "o diff max (Vs official): ",
        torch.abs(o_group_norm_torch_official - o_group_norm_torch).max().item(),
    )
    print(
        "o diff norm (Vs official): ",
        torch.norm(o_group_norm_torch_official - o_group_norm_torch).item(),
    )
    assert torch.allclose(
        o_group_norm_torch_official, o_group_norm_torch, atol=atol, rtol=rtol
    )

    print(
        "o diff max (Vs triton): ",
        torch.abs(o_group_norm_torch - o_group_norm_triton).max().item(),
    )
    print(
        "o diff norm (Vs triton): ",
        torch.norm(o_group_norm_torch - o_group_norm_triton).item(),
    )
    assert torch.allclose(o_group_norm_torch, o_group_norm_triton, atol=atol, rtol=rtol)

    ##### bwd
    print(
        "dx diff max (Vs official): ",
        torch.abs(dx_group_norm_torch - dx_group_norm_torch_official).max().item(),
    )
    print(
        "dx diff norm (Vs official): ",
        torch.norm(dx_group_norm_torch - dx_group_norm_torch_official).item(),
    )
    assert torch.allclose(
        dx_group_norm_torch, dx_group_norm_torch_official, atol=atol, rtol=rtol
    )

    print(
        "dx diff max (Vs triton): ",
        torch.abs(dx_group_norm_torch - dx_group_norm_triton).max().item(),
    )
    print(
        "dx diff norm (Vs triton): ",
        torch.norm(dx_group_norm_torch - dx_group_norm_triton).item(),
    )
    assert torch.allclose(
        dx_group_norm_torch, dx_group_norm_triton, atol=atol, rtol=rtol
    )

    print(
        "dw diff max (Vs official): ",
        torch.abs(dw_group_norm_torch - dw_group_norm_torch_official).max().item(),
    )
    print(
        "dw diff norm (Vs official): ",
        torch.norm(dw_group_norm_torch - dw_group_norm_torch_official).item(),
    )
    assert torch.allclose(
        dw_group_norm_torch, dw_group_norm_torch_official, atol=atol, rtol=rtol
    )

    print(
        "dw diff max (Vs triton): ",
        torch.abs(dw_group_norm_torch - dw_group_norm_triton).max().item(),
    )
    print(
        "dw diff norm (Vs triton): ",
        torch.norm(dw_group_norm_torch - dw_group_norm_triton).item(),
    )
    assert torch.allclose(
        dw_group_norm_torch, dw_group_norm_triton, atol=atol, rtol=rtol
    )

    print(
        "db diff max (Vs official): ",
        torch.abs(db_group_norm_torch - db_group_norm_torch_official).max().item(),
    )
    print(
        "db diff norm (Vs official): ",
        torch.norm(db_group_norm_torch - db_group_norm_torch_official).item(),
    )
    assert torch.allclose(
        db_group_norm_torch, db_group_norm_torch_official, atol=atol, rtol=rtol
    )

    print(
        "db diff max (Vs triton): ",
        torch.abs(db_group_norm_torch - db_group_norm_triton).max().item(),
    )
    print(
        "db diff norm (Vs triton): ",
        torch.norm(db_group_norm_torch - db_group_norm_triton).item(),
    )
    assert torch.allclose(
        db_group_norm_torch, db_group_norm_triton, atol=atol, rtol=rtol
    )

    if use_residual:
        print(
            "dr diff max: ",
            torch.abs(dr_group_norm_torch - dr_group_norm_triton).max().item(),
        )
        print(
            "dr diff norm: ",
            torch.norm(dr_group_norm_torch - dr_group_norm_triton).item(),
        )
        assert torch.allclose(
            dr_group_norm_torch, dr_group_norm_triton, atol=atol, rtol=rtol
        )
