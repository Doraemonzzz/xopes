import pytest
import torch

from xopes.ops.normalize import groupsrmsnorm_torch, groupsrmsnorm_triton
from xopes.utils import get_threshold


def get_params():
    shape = [(6, 128), (4, 8, 256)]
    return shape


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_residual", [True, False])
@pytest.mark.parametrize("return_residual", [True, False])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_groups", [4, 16])
def test(shape, use_residual, return_residual, eps, dtype, num_groups):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    d = shape[-1]
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    do = torch.randn(shape, dtype=dtype, device=device)

    if use_residual:
        residual = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    else:
        residual = None

    # forward
    o_groupsrmsnorm_torch, o_update_residual_torch = groupsrmsnorm_torch(
        x,
        dim=d,
        eps=eps,
        residual=residual,
        return_residual=return_residual,
        num_groups=num_groups,
    )
    if use_residual and return_residual:
        o_groupsrmsnorm_torch = o_groupsrmsnorm_torch + o_update_residual_torch

    o_groupsrmsnorm_triton, o_update_residual_triton = groupsrmsnorm_triton(
        x,
        dim=d,
        eps=eps,
        residual=residual,
        return_residual=return_residual,
        num_groups=num_groups,
    )
    if use_residual and return_residual:
        o_groupsrmsnorm_triton = o_groupsrmsnorm_triton + o_update_residual_triton

    # backward
    o_groupsrmsnorm_torch.backward(do, retain_graph=True)
    dx_groupsrmsnorm_torch, x.grad = x.grad.clone(), None

    if use_residual:
        dr_groupsrmsnorm_torch, residual.grad = residual.grad.clone(), None
    else:
        dr_groupsrmsnorm_torch = None

    o_groupsrmsnorm_triton.backward(do, retain_graph=True)
    dx_groupsrmsnorm_triton, x.grad = x.grad.clone(), None

    if use_residual:
        dr_groupsrmsnorm_triton, residual.grad = residual.grad.clone(), None
    else:
        dr_groupsrmsnorm_triton = None

    atol, rtol = get_threshold(dtype)

    ##### fwd
    print(
        "o diff max: ",
        torch.abs(o_groupsrmsnorm_torch - o_groupsrmsnorm_triton).max().item(),
    )
    print(
        "o diff norm: ",
        torch.norm(o_groupsrmsnorm_torch - o_groupsrmsnorm_triton).item(),
    )
    assert torch.allclose(
        o_groupsrmsnorm_torch, o_groupsrmsnorm_triton, atol=atol, rtol=rtol
    )

    ##### bwd
    print(
        "dx diff max: ",
        torch.abs(dx_groupsrmsnorm_torch - dx_groupsrmsnorm_triton).max().item(),
    )
    print(
        "dx diff norm: ",
        torch.norm(dx_groupsrmsnorm_torch - dx_groupsrmsnorm_triton).item(),
    )
    assert torch.allclose(
        dx_groupsrmsnorm_torch, dx_groupsrmsnorm_triton, atol=atol, rtol=rtol
    )

    if use_residual:
        print(
            "dr diff max: ",
            torch.abs(dr_groupsrmsnorm_torch - dr_groupsrmsnorm_triton).max().item(),
        )
        print(
            "dr diff norm: ",
            torch.norm(dr_groupsrmsnorm_torch - dr_groupsrmsnorm_triton).item(),
        )
        assert torch.allclose(
            dr_groupsrmsnorm_torch, dr_groupsrmsnorm_triton, atol=atol, rtol=rtol
        )
