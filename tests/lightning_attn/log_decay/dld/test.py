import pytest
import torch

from xopes.ops.lightning_attn.log_decay import compute_dld_torch, compute_dld_triton
from xopes.utils import get_threshold


def get_params():
    shapes = [
        # standard shape
        (2, 1024, 8, 32, 16),
        # BLOCK_N +- 1
        (2, 257, 8, 64, 32),
        (2, 255, 8, 64, 32),
        (2, 65, 7, 33, 63),
        # BLOCK_N +- C
        (2, 270, 8, 64, 32),
        (2, 270, 8, 33, 16),
        (2, 1125, 8, 43, 33),
        # LARGE D, E
        (2, 1125, 8, 255, 257),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_final_state", [True, False])
@pytest.mark.parametrize("use_varlen", [False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_dld(shape, use_final_state, use_varlen, dtype):
    # Set random seed for reproducibility
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, n, h, d, e = shape

    if use_varlen:
        b = 1
        m = n // 5
        cu_seqlens = torch.tensor(
            [0, m - 2, 2 * m + 1, 3 * m - 1, 4 * m, n], dtype=torch.long, device=device
        )
    else:
        cu_seqlens = None

    # Generate input tensors
    dld_q = torch.randn((b, n, h), dtype=dtype, device=device)
    dld_k = torch.randn((b, n, h), dtype=dtype, device=device)

    if use_final_state:
        final_state = torch.randn((b, h, d, e), dtype=dtype, device=device)
        dfinal_state = torch.randn((b, h, d, e), dtype=dtype, device=device)
    else:
        final_state = None
        dfinal_state = None

    # Compute using PyTorch implementation
    dld_torch = compute_dld_torch(
        dld_q=dld_q,
        dld_k=dld_k,
        final_state=final_state,
        dfinal_state=dfinal_state,
        cu_seqlens=cu_seqlens,
    )

    # Compute using Triton implementation
    dld_triton = compute_dld_triton(
        dld_q=dld_q,
        dld_k=dld_k,
        final_state=final_state,
        dfinal_state=dfinal_state,
        cu_seqlens=cu_seqlens,
    )

    # Get tolerance thresholds based on dtype
    atol, rtol = get_threshold(dtype)

    c = 64
    m = (n + c - 1) // c
    print(torch.norm((dld_torch - dld_triton)[:, 0, :]).item())
    for i in range(m):
        start = i * c
        end = min(start + c, n)
        print(
            i,
            torch.norm(dld_torch[:, start:end, :] - dld_triton[:, start:end, :]).item(),
        )

    # Check results
    print(
        "dld diff max (torch vs triton): ",
        torch.abs(dld_torch - dld_triton).max().item(),
    )
    print(
        "dld diff norm (torch vs triton): ",
        torch.norm(dld_torch - dld_triton).item(),
    )

    # Assert that the results are close
    assert torch.allclose(dld_torch, dld_triton, atol=atol, rtol=rtol)
