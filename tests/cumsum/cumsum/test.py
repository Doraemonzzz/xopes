import pytest
import torch

from xopes.ops.cumsum.cumsum import (
    cumsum_chunk_loop_triton,
    cumsum_no_reshape_triton,
    cumsum_torch,
    cumsum_triton,
)
from xopes.utils import get_threshold


def get_params():
    shapes = [
        (6, 128),
        (4, 8, 256),
        (1024, 4096, 4),
        (1023, 4095, 12),
        (1025, 4097, 12),
    ]

    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("use_cu_seqlens", [False])
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
    o_cumsum_chunk_loop_triton = cumsum_chunk_loop_triton(
        x, dim=dim, reverse=reverse, cu_seqlens=cu_seqlens
    )
    o_cumsum_no_reshape_triton = cumsum_no_reshape_triton(
        x, dim=dim, reverse=reverse, cu_seqlens=cu_seqlens
    )

    # backward
    o_cumsum_torch.backward(do, retain_graph=True)
    dx_cumsum_torch, x.grad = x.grad.clone(), None

    o_cumsum_triton.backward(do, retain_graph=True)
    dx_cumsum_triton, x.grad = x.grad.clone(), None

    o_cumsum_chunk_loop_triton.backward(do, retain_graph=True)
    dx_cumsum_chunk_loop_triton, x.grad = x.grad.clone(), None

    o_cumsum_no_reshape_triton.backward(do, retain_graph=True)
    dx_cumsum_no_reshape_triton, x.grad = x.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    # forward check
    print(
        "o diff max (Vs triton):",
        torch.abs(o_cumsum_torch - o_cumsum_triton).max().item(),
    )
    print(
        "o diff norm (Vs triton):",
        torch.norm(o_cumsum_torch - o_cumsum_triton).item(),
    )
    assert torch.allclose(o_cumsum_torch, o_cumsum_triton, atol=atol, rtol=rtol)

    print(
        "o diff max (Vs chunk loop triton):",
        torch.abs(o_cumsum_torch - o_cumsum_chunk_loop_triton).max().item(),
    )
    print(
        "o diff norm (Vs chunk loop triton):",
        torch.norm(o_cumsum_torch - o_cumsum_chunk_loop_triton).item(),
    )
    assert torch.allclose(
        o_cumsum_torch, o_cumsum_chunk_loop_triton, atol=atol, rtol=rtol
    )

    print(
        "o diff max (Vs triton no reshape):",
        torch.abs(o_cumsum_torch - o_cumsum_no_reshape_triton).max().item(),
    )
    print(
        "o diff norm (Vs triton no reshape):",
        torch.norm(o_cumsum_torch - o_cumsum_no_reshape_triton).item(),
    )
    assert torch.allclose(
        o_cumsum_torch, o_cumsum_no_reshape_triton, atol=atol, rtol=rtol
    )

    # backward check
    print(
        "dx diff max (Vs triton):",
        torch.abs(dx_cumsum_torch - dx_cumsum_triton).max().item(),
    )
    print(
        "dx diff norm (Vs triton):",
        torch.norm(dx_cumsum_torch - dx_cumsum_triton).item(),
    )
    assert torch.allclose(dx_cumsum_torch, dx_cumsum_triton, atol=atol, rtol=rtol)

    print(
        "dx diff max (Vs chunk loop triton):",
        torch.abs(dx_cumsum_torch - dx_cumsum_chunk_loop_triton).max().item(),
    )
    print(
        "dx diff norm (Vs chunk loop triton):",
        torch.norm(dx_cumsum_torch - dx_cumsum_chunk_loop_triton).item(),
    )
    assert torch.allclose(
        dx_cumsum_torch, dx_cumsum_chunk_loop_triton, atol=atol, rtol=rtol
    )

    print(
        "dx diff max (Vs triton no reshape):",
        torch.abs(dx_cumsum_torch - dx_cumsum_no_reshape_triton).max().item(),
    )
    print(
        "dx diff norm (Vs triton no reshape):",
        torch.norm(dx_cumsum_torch - dx_cumsum_no_reshape_triton).item(),
    )
    assert torch.allclose(
        dx_cumsum_torch, dx_cumsum_no_reshape_triton, atol=atol, rtol=rtol
    )
