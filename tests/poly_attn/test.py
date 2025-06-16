import pytest
import torch

from xopes.ops.poly_attn import poly_attn_chunk, poly_attn_log_torch
from xopes.utils import assert_close, print_diff


def get_params():
    """
    Generate test parameter combinations for different tensor shapes.
    Returns various shapes to test edge cases and typical usage scenarios.
    """
    shapes = [(1, 512, 16, 16), (2, 1024, 128, 128)]

    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("p", [2, 4, 8])  # Polynomial order
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
    ],
)
def test_poly_attn_implementations(shape, p, causal, dtype):
    """
    Test different polynomial attention implementations for consistency.

    This test compares:
    1. poly_attn_log_torch (baseline) vs poly_attn_chunk (chunked implementation)
    """
    torch.manual_seed(2024)
    device = torch.device("cuda")

    b, n, h, d = shape
    chunk_size = 64  # For chunked implementation

    # Generate input tensors
    q = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    k = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    v = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    do = torch.randn(shape, dtype=dtype, device=device)

    # Forward pass - baseline implementation
    o_poly_log_torch = poly_attn_log_torch(
        q.clone(), k.clone(), v.clone(), p=p, causal=causal
    )

    # Forward pass - chunked implementation
    o_poly_chunk_torch = poly_attn_chunk(
        q.clone(), k.clone(), v.clone(), p=p, chunk_size=chunk_size, causal=causal
    )

    # Backward pass - poly_attn_log_torch
    o_poly_log_torch.backward(do, retain_graph=True)
    dq_poly_log_torch, q.grad = q.grad.clone(), None
    dk_poly_log_torch, k.grad = k.grad.clone(), None
    dv_poly_log_torch, v.grad = v.grad.clone(), None

    # Backward pass - poly_attn_chunk
    o_poly_chunk_torch.backward(do, retain_graph=True)
    dq_poly_chunk_torch, q.grad = q.grad.clone(), None
    dk_poly_chunk_torch, k.grad = k.grad.clone(), None
    dv_poly_chunk_torch, v.grad = v.grad.clone(), None

    # Set tolerance based on dtype
    if dtype == torch.float32:
        atol, rtol = 1e-4, 1e-3
    elif dtype == torch.float16:
        atol, rtol = 1e-2, 1e-2
    else:  # bfloat16
        atol, rtol = 2e-2, 2e-2

    # Forward check - poly_attn_log_torch vs poly_attn_chunk
    print("=== Forward pass comparison: poly_attn_log_torch vs poly_attn_chunk ===")
    print(
        "o diff max: ",
        torch.abs(o_poly_log_torch - o_poly_chunk_torch).max().item(),
    )
    print("o diff norm: ", torch.norm(o_poly_log_torch - o_poly_chunk_torch).item())
    print_diff(o_poly_log_torch, o_poly_chunk_torch, n)
    assert_close(o_poly_log_torch, o_poly_chunk_torch, atol=atol, rtol=rtol)

    # Backward check - dq gradients
    print("=== Backward pass comparison: dq gradients ===")
    print(
        "dq diff max (log vs chunk): ",
        torch.abs(dq_poly_log_torch - dq_poly_chunk_torch).max().item(),
    )
    print(
        "dq diff norm (log vs chunk): ",
        torch.norm(dq_poly_log_torch - dq_poly_chunk_torch).item(),
    )
    print("ccc", torch.max(dq_poly_log_torch), torch.max(dq_poly_chunk_torch))
    print_diff(dq_poly_log_torch, dq_poly_chunk_torch, n)
    assert_close(dq_poly_log_torch, dq_poly_chunk_torch, atol=atol, rtol=rtol)

    # Backward check - dk gradients
    print("=== Backward pass comparison: dk gradients ===")
    print(
        "dk diff max (log vs chunk): ",
        torch.abs(dk_poly_log_torch - dk_poly_chunk_torch).max().item(),
    )
    print(
        "dk diff norm (log vs chunk): ",
        torch.norm(dk_poly_log_torch - dk_poly_chunk_torch).item(),
    )
    print_diff(dk_poly_log_torch, dk_poly_chunk_torch, n)
    assert_close(dk_poly_log_torch, dk_poly_chunk_torch, atol=atol, rtol=rtol)

    # Backward check - dv gradients
    print("=== Backward pass comparison: dv gradients ===")
    print(
        "dv diff max (log vs chunk): ",
        torch.abs(dv_poly_log_torch - dv_poly_chunk_torch).max().item(),
    )
    print(
        "dv diff norm (log vs chunk): ",
        torch.norm(dv_poly_log_torch - dv_poly_chunk_torch).item(),
    )
    print_diff(dv_poly_log_torch, dv_poly_chunk_torch, n)
    assert_close(dv_poly_log_torch, dv_poly_chunk_torch, atol=atol, rtol=rtol)
