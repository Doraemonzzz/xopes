import pytest
import torch

from xopes.ops.cross_entropy import (
    cross_entropy_parallel_triton,
    cross_entropy_torch,
    cross_entropy_triton,
)
from xopes.utils import get_threshold


def get_params():
    shapes = [(512, 2048), (1024, 4096), (512, 2000), (12288, 50257)]
    # shapes = [(12288, 50257)]
    # shapes = [(1, 50257)]
    # shapes = [(1, 65536)]
    # shapes = [(1, 128 * 1024)]

    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])

# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("reduction", ["none"])
# @pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
def test(shape, dtype, reduction, label_smoothing, ignore_index=-100):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, v = shape
    b_ignore = b // 2

    # Generate input tensors
    z = torch.randn((b, v), dtype=dtype, device=device).requires_grad_()
    y_ignore = torch.full((b_ignore,), ignore_index, device=device)
    y = torch.randint(0, v, (b - b_ignore,), device=device)
    y = torch.cat([y_ignore, y], dim=0)
    y = torch.randint(0, v, (b,), device=device)

    # Forward
    o_ce_torch = cross_entropy_torch(
        z.float(),
        y,
        reduction=reduction,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
    )
    o_ce_triton = cross_entropy_triton(
        z.clone(),  # important, since this op use inplace operation
        y.clone(),
        reduction=reduction,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
    )
    o_ce_parallel_triton = cross_entropy_parallel_triton(
        z,
        y,
        reduction=reduction,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index,
    )

    do = torch.rand_like(o_ce_torch)

    # Backward
    o_ce_torch.backward(do, retain_graph=True)
    dz_ce_torch, z.grad = z.grad.clone(), None

    o_ce_triton.backward(do, retain_graph=True)
    dz_ce_triton, z.grad = z.grad.clone(), None

    o_ce_parallel_triton.backward(do, retain_graph=True)
    dz_ce_parallel_triton, z.grad = z.grad.clone(), None

    atol, rtol = get_threshold(dtype)

    if reduction != "none":
        print(f"o_ce_torch: {o_ce_torch}")
        print(f"o_ce_triton: {o_ce_triton}")
        print(f"o_ce_parallel_triton: {o_ce_parallel_triton}")

    # Forward check
    print(
        "o diff max (Vs naive triton): ",
        torch.abs(o_ce_torch - o_ce_triton).max().item(),
    )
    print(
        "o diff norm (Vs naive triton): ",
        torch.norm(o_ce_torch - o_ce_triton).item(),
    )
    assert torch.allclose(
        o_ce_torch.to(o_ce_triton.dtype), o_ce_triton, atol=atol, rtol=rtol
    )

    print(
        f"o diff max (Vs parallel triton): {torch.abs(o_ce_torch - o_ce_parallel_triton).max().item()}"
    )
    print(
        f"o diff norm (Vs parallel triton): {torch.norm(o_ce_torch - o_ce_parallel_triton).item()}"
    )
    assert torch.allclose(
        o_ce_torch.to(o_ce_parallel_triton.dtype),
        o_ce_parallel_triton,
        atol=atol,
        rtol=rtol,
    )

    # Backward check
    print(
        "dz diff max (Vs naive triton): ",
        torch.abs(dz_ce_torch - dz_ce_triton).max().item(),
    )
    print(
        "dz diff norm (Vs naive triton): ",
        torch.norm(dz_ce_torch - dz_ce_triton).item(),
    )
    assert torch.allclose(dz_ce_torch, dz_ce_triton, atol=atol, rtol=rtol)

    print(
        "dz diff max (Vs parallel triton): ",
        torch.abs(dz_ce_torch - dz_ce_parallel_triton).max().item(),
    )
    print(
        "dz diff norm (Vs parallel triton): ",
        torch.norm(dz_ce_torch - dz_ce_parallel_triton).item(),
    )

    assert torch.allclose(
        dz_ce_torch.to(dz_ce_parallel_triton.dtype),
        dz_ce_parallel_triton,
        atol=atol,
        rtol=rtol,
    )
