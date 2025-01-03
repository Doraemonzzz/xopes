import pytest
import torch

from xopes.ops.logsumexp import lse_parallel_triton, lse_recurrence_triton, lse_torch
from xopes.utils import get_threshold


def get_params():
    shapes = [
        (512,),
        (6, 128),
        (6, 129),
        (4, 4, 255),
        (4, 8, 256),
    ]

    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(shape, dim, keepdim, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    if len(shape) == 1:
        dim = 0

    # Generate input tensor
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    do = torch.randn_like(lse_torch(x, dim=dim, keepdim=keepdim))

    # forward
    o_torch = lse_torch(x, dim=dim, keepdim=keepdim)
    o_parallel_triton = lse_parallel_triton(x, dim=dim, keepdim=keepdim)
    o_recurrence_triton = lse_recurrence_triton(x, dim=dim, keepdim=keepdim)

    # backward
    o_torch.backward(do, retain_graph=True)
    dx_torch, x.grad = x.grad.clone(), None

    o_parallel_triton.backward(do, retain_graph=True)
    dx_parallel_triton, x.grad = x.grad.clone(), None

    o_recurrence_triton.backward(do, retain_graph=True)
    dx_recurrence_triton, x.grad = x.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    # forward check
    print(
        "o diff max: ",
        torch.abs(o_torch - o_parallel_triton).max().item(),
    )
    print(
        "o diff norm: ",
        torch.norm(o_torch - o_parallel_triton).item(),
    )
    assert torch.allclose(o_torch, o_parallel_triton, atol=atol, rtol=rtol)

    # print("o diff max: ", torch.abs(o_torch - o_recurrence_triton).max().item())
    # print("o diff norm: ", torch.norm(o_torch - o_recurrence_triton).item())
    # assert torch.allclose(o_torch, o_recurrence_triton, atol=atol, rtol=rtol)

    # backward check
    print(
        "dx diff max: ",
        torch.abs(dx_torch - dx_parallel_triton).max().item(),
    )
    print(
        "dx diff norm: ",
        torch.norm(dx_torch - dx_parallel_triton).item(),
    )
    assert torch.allclose(dx_torch, dx_parallel_triton, atol=atol, rtol=rtol)

    # print("dx diff max: ", torch.abs(dx_torch - dx_recurrence_triton).max().item())
    # print("dx diff norm: ", torch.norm(dx_torch - dx_recurrence_triton).item())
    # assert torch.allclose(dx_torch, dx_recurrence_triton, atol=atol, rtol=rtol)
