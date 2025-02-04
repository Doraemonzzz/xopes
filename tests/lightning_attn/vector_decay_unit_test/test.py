import math

import pytest
import torch
import torch.nn.functional as F

from xopes.ops.lightning_attn.vector_decay import (
    lavd_chunk_parallel_torch,
    lavd_chunk_parallel_triton,
)
from xopes.utils import get_threshold


def get_params():
    shapes = [
        (2, 128, 8, 64, 32),
        (2, 127, 16, 64, 128),
        (2, 1023, 16, 128, 64),
        (2, 64, 16, 128, 64),
        (2, 63, 16, 128, 64),
        (4, 256, 12, 64, 128),
        # (2, 127, 16, 128, 128),
        # (2, 128, 16, 128, 128),
        # (2, 32, 16, 64, 128),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("share_k", [False, True])
@pytest.mark.parametrize("share_v", [False, True])
@pytest.mark.parametrize("use_initial_state", [True])
@pytest.mark.parametrize("use_zero_ld", [False, True])

# @pytest.mark.parametrize("share_k", [False])
# @pytest.mark.parametrize("share_v", [False])
# @pytest.mark.parametrize("use_initial_state", [True])
# @pytest.mark.parametrize("use_zero_ld", [True])
@pytest.mark.parametrize("dtype", [torch.float32])
def test(shape, share_k, share_v, use_initial_state, use_zero_ld, dtype):
    use_ldk = True
    use_ldv = True
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, n, h, d, e = shape
    test_chunk = n <= 128
    chunk_size = int(2 ** (int(math.log2(n)) - 1))

    # Generate input tensors
    q = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()

    if share_k:
        k = F.sigmoid(
            torch.randn((b, n, h, d), dtype=dtype, device=device)
        ).requires_grad_()
        ldk = None
    else:
        k = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
        if use_zero_ld:
            ldk = torch.zeros((b, n, h, d), dtype=dtype, device=device).requires_grad_()
        else:
            ldk = F.logsigmoid(
                torch.randn((b, n, h, d), dtype=dtype, device=device)
            ).requires_grad_()

    if share_v:
        v = F.sigmoid(
            torch.randn((b, n, h, e), dtype=dtype, device=device)
        ).requires_grad_()
        ldv = None
    else:
        v = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
        if use_zero_ld:
            ldv = torch.zeros((b, n, h, e), dtype=dtype, device=device).requires_grad_()
        else:
            ldv = F.logsigmoid(
                torch.randn((b, n, h, e), dtype=dtype, device=device)
            ).requires_grad_()

    if use_initial_state:
        initial_state = torch.randn(
            (b, h, d, e), dtype=dtype, device=device
        ).requires_grad_()
    else:
        initial_state = None

    do = torch.randn((), dtype=dtype, device=device)

    ##### Forward pass
    # chunk parallel torch
    (o_chunk_parallel, s_chunk_parallel,) = lavd_chunk_parallel_torch(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )
    # output_chunk_parallel = o_chunk_parallel.sum() + s_chunk_parallel.sum()

    # chunk parallel triton
    (o_chunk_parallel_triton, s_chunk_parallel_triton,) = lavd_chunk_parallel_triton(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        initial_state=initial_state,
    )
    # output_chunk_parallel_triton = o_chunk_parallel_triton.sum() + s_chunk_parallel_triton.sum()

    atol, rtol = get_threshold(dtype)

    ##### Check forward pass results
    print(
        "o diff max: ",
        torch.abs(o_chunk_parallel - o_chunk_parallel_triton).max().item(),
    )
    print(
        "o diff norm: ", torch.norm(o_chunk_parallel - o_chunk_parallel_triton).item()
    )
    assert torch.allclose(
        o_chunk_parallel, o_chunk_parallel_triton, atol=atol, rtol=rtol
    )

    print(
        "s diff max: ",
        torch.abs(s_chunk_parallel - s_chunk_parallel_triton).max().item(),
    )
    print(
        "s diff norm: ", torch.norm(s_chunk_parallel - s_chunk_parallel_triton).item()
    )
    assert torch.allclose(
        s_chunk_parallel, s_chunk_parallel_triton, atol=atol, rtol=rtol
    )
