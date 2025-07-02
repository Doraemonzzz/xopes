import pytest
import torch
import torch.nn.functional as F

from xopes.ops.kernel_regression.causal_linear.krcl_parallel_triton import (
    krcl_parallel_inverse,
)
from xopes.ops.kernel_regression.causal_linear.torch_utils import krcl_inverse_torch
from xopes.utils import assert_close


def get_params():
    """
    Generate test parameter combinations for different tensor shapes.
    Returns various shapes to test edge cases and typical usage scenarios.
    """
    shapes = [
        (2, 256, 12, 128),
        (2, 1024, 8, 32),
        (2, 257, 8, 64),
        (2, 255, 8, 64),
        (2, 65, 7, 33),
        (2, 270, 8, 64),
        (2, 270, 8, 33),
        (2, 1125, 8, 43),
        (8, 2048, 12, 64),
        (2, 128, 12, 128),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_q", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("use_varlen", [False])
@pytest.mark.parametrize("c", [0.1, 10])  # Scaling factor for log decay
@pytest.mark.parametrize("dtype", [torch.float32])
def test_krcl(shape, use_q, reverse, use_varlen, c, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    scale = 0.01
    b, n, h, d = shape

    # Setup variable length sequences if requested
    if use_varlen:
        b = 1
        m = n // 5
        cu_seqlens = torch.tensor(
            [0, m - 2, 2 * m + 1, 3 * m - 1, 4 * m, n], dtype=torch.long, device=device
        )
    else:
        cu_seqlens = None

    # Generate input tensors
    if use_q:
        q = (
            F.normalize(torch.randn((b, n, h, d), dtype=dtype, device=device), dim=-1)
        ).requires_grad_()
    else:
        q = None
    k = (
        F.normalize(torch.randn((b, n, h, d), dtype=dtype, device=device), dim=-1)
    ).requires_grad_()

    ld = F.logsigmoid(
        (1 + scale * torch.ones((b, n, h), dtype=dtype, device=device)) * c
    ).requires_grad_()

    alpha = (
        F.sigmoid(torch.randn((b, n, h), dtype=dtype, device=device))
    ).requires_grad_()
    beta = (
        F.sigmoid(torch.randn((b, n, h), dtype=dtype, device=device))
    ).requires_grad_()
    BLOCK_N = 64

    ##### Forward pass comparison
    # PyTorch reference implementation
    o_torch = krcl_inverse_torch(
        q=q.clone() if use_q else None,
        k=k.clone(),
        ld=ld.clone(),
        alpha=alpha.clone(),
        beta=beta.clone(),
        cu_seqlens=cu_seqlens,
        reverse=reverse,
        BLOCK_N=BLOCK_N,
    )

    # Triton optimized implementation
    o_triton = krcl_parallel_inverse(
        q=q.clone() if use_q else None,
        k=k.clone(),
        ld=ld.clone(),
        alpha=alpha.clone(),
        beta=beta.clone(),
        cu_seqlens=cu_seqlens,
        reverse=reverse,
        BLOCK_N=BLOCK_N,
    )

    # Set tolerance for numerical comparisons
    atol = 5e-3
    rtol = 5e-3
    ld_atol = 7e-2 if dtype == torch.bfloat16 else atol
    ld_rtol = 7e-2 if dtype == torch.bfloat16 else rtol

    ##### Forward pass validation
    print("o diff max (torch vs triton): ", torch.abs(o_torch - o_triton).max().item())
    print("o diff norm (torch vs triton): ", torch.norm(o_torch - o_triton).item())
    m = o_torch.shape[2]
    print(o_torch[0, 0, 0, :4, :4])
    for i in range(m):
        print(i, torch.norm(o_torch[:, :, i] - o_triton[:, :, i]).item())
    assert_close(o_torch, o_triton, atol=atol, rtol=rtol)
