import pytest
import torch

from xopes.ops.cumsum import cumsum_torch, cumsum_triton
from xopes.utils import get_threshold


def get_params():
    shapes = [(6, 128), (4, 8, 256), (4, 1024, 4096), (12, 32, 15)]

    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("use_cu_seqlens", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(shape, dim, reverse, use_cu_seqlens, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")

    if use_cu_seqlens:
        n = int(torch.randint(1024, 4096, (1,)).item())
        m = n // 5
        cu_seqlens = torch.tensor(
            [0, m - 2, 2 * m + 1, 3 * m - 1, 4 * m, n], dtype=torch.long, device=device
        )
        d = 768
        dim = 0
        shape = (n, d)
    else:
        cu_seqlens = None

    # Generate input tensor
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    do = torch.randn(shape, dtype=dtype, device=device)

    # Skip invalid dim values
    if abs(dim) >= len(shape):
        return

    # forward
    o_cumsum_torch = cumsum_torch(x, dim=dim, reverse=reverse, cu_seqlens=cu_seqlens)
    o_cumsum_triton = cumsum_triton(x, dim=dim, reverse=reverse, cu_seqlens=cu_seqlens)

    # backward
    o_cumsum_torch.backward(do, retain_graph=True)
    dx_cumsum_torch, x.grad = x.grad.clone(), None

    o_cumsum_triton.backward(do, retain_graph=True)
    dx_cumsum_triton, x.grad = x.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    # forward check
    print(
        "o diff max: ",
        torch.abs(o_cumsum_torch - o_cumsum_triton).max().item(),
    )
    print(
        "o diff norm: ",
        torch.norm(o_cumsum_torch - o_cumsum_triton).item(),
    )
    assert torch.allclose(o_cumsum_torch, o_cumsum_triton, atol=atol, rtol=rtol)

    # backward check
    print(
        "dx diff max: ",
        torch.abs(dx_cumsum_torch - dx_cumsum_triton).max().item(),
    )
    print(
        "dx diff norm: ",
        torch.norm(dx_cumsum_torch - dx_cumsum_triton).item(),
    )
    assert torch.allclose(dx_cumsum_torch, dx_cumsum_triton, atol=atol, rtol=rtol)
