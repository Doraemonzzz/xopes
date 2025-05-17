import pytest
import torch
import torch.nn.functional as F
import triton

from xopes.ops.lightning_attn.vector_decay.lavd_parallel_triton import (
    lavd_parallel_sub_intra,
    lavd_parallel_sub_intra_sep,
)
from xopes.ops.lightning_attn.vector_decay.torch_utils import lavd_intra_torch
from xopes.utils import get_threshold


def get_params():
    shapes = [
        # standard shape
        (2, 256, 12, 128, 128),
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
        (2, 1025, 8, 255, 257),
        # Train shape
        (8, 2048, 12, 64, 64),
    ]

    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_ldk", [True])
@pytest.mark.parametrize("use_ldv", [True])
@pytest.mark.parametrize("share_k", [True, False])
@pytest.mark.parametrize("share_v", [True, False])
@pytest.mark.parametrize(
    "reverse",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize("BLOCK_N", [64])  # 16 for only sub intra, 64 for full test
@pytest.mark.parametrize("c", [10])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_lavd_intra(
    shape, use_ldk, use_ldv, share_k, share_v, reverse, BLOCK_N, c, dtype
):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    scale = 0.01

    # Generate input tensors
    b, n, h, d, e = shape

    q = torch.randn(b, n, h, d, dtype=dtype, device=device).requires_grad_()

    if share_k:
        use_ldk = True
        ldk = F.logsigmoid(
            (1 + scale * torch.randn(b, n, h, d, dtype=dtype, device=device)) * c
        ).requires_grad_()
        k = None
    else:
        k = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()

        if use_ldk:
            ldk = F.logsigmoid(
                (1 + scale * torch.randn(b, n, h, d, dtype=dtype, device=device)) * c
            ).requires_grad_()
        else:
            ldk = None

    if share_v:
        use_ldv = True
        ldv = F.logsigmoid(
            (1 + scale * torch.randn(b, n, h, e, dtype=dtype, device=device)) * c
        ).requires_grad_()
        v = None
    else:
        v = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
        if use_ldv:
            ldv = F.logsigmoid(
                (1 + scale * torch.randn(b, n, h, e, dtype=dtype, device=device)) * c
            ).requires_grad_()
        else:
            ldv = None

    # Set block sizes for Triton kernel
    MAX_BLOCK_N = triton.next_power_of_2(n)
    MAX_BLOCK_C = MAX_BLOCK_N
    MAX_BLOCK_E = triton.next_power_of_2(e)
    MAX_BLOCK_D = triton.next_power_of_2(d)
    BLOCK_C = 16

    # Forward pass with PyTorch reference implementation
    o_torch = lavd_intra_torch(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        reverse=reverse,
        BLOCK_N=BLOCK_C,
    )

    # Forward pass with Triton parallel implementation
    o_triton = lavd_parallel_sub_intra(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        reverse=reverse,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
    )

    o_triton_sep = lavd_parallel_sub_intra_sep(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        reverse=reverse,
        MAX_BLOCK_N=MAX_BLOCK_N,
        MAX_BLOCK_C=MAX_BLOCK_C,
        MAX_BLOCK_E=MAX_BLOCK_E,
        MAX_BLOCK_D=MAX_BLOCK_D,
        BLOCK_N=BLOCK_N,
        BLOCK_C=BLOCK_C,
    )

    # Get thresholds based on dtype for numerical comparison
    atol, rtol = get_threshold(dtype)

    # Print test configuration and results
    print(f"\nShape: {shape}, E: {e}, reverse: {reverse}, dtype: {dtype}")
    print(
        "o diff max (torch parallel Vs triton parallel): ",
        torch.abs(o_torch - o_triton).max().item(),
    )
    print(
        "o diff norm (torch parallel Vs triton parallel): ",
        torch.norm(o_torch - o_triton).item(),
    )

    print(
        "o diff max (torch parallel Vs triton parallel sep): ",
        torch.abs(o_torch - o_triton_sep).max().item(),
    )
    print(
        "o diff norm (torch parallel Vs triton parallel sep): ",
        torch.norm(o_torch - o_triton_sep).item(),
    )

    l = (n + BLOCK_C - 1) // BLOCK_C
    for i in range(l):
        start = i * BLOCK_C
        end = min(start + BLOCK_C, n)
        print(
            f"block {i} diff norm (torch parallel Vs triton parallel): {torch.norm(o_torch[:, start:end] - o_triton[:, start:end]).item()}"
        )
        print(
            f"block {i} diff norm (torch parallel Vs triton parallel sep): {torch.norm(o_torch[:, start:end] - o_triton_sep[:, start:end]).item()}"
        )

    # Assert that the results are close enough
    assert torch.allclose(o_torch, o_triton, atol=atol, rtol=rtol)
    assert torch.allclose(o_torch, o_triton_sep, atol=atol, rtol=rtol)
