
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
        # (4, 256, 12, 64, 128),
        # (2, 127, 16, 128, 128),
        # (2, 128, 16, 128, 128),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
# @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])


@pytest.mark.parametrize("share_k", [False, True])
@pytest.mark.parametrize("share_v", [False, True])
@pytest.mark.parametrize("use_initial_state", [True])
@pytest.mark.parametrize("use_zero_ld", [False, True])

# @pytest.mark.parametrize("share_k", [True])
# @pytest.mark.parametrize("share_v", [True])
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
    # chunk_size = int(2 ** (int(math.log2(n)) - 1))
    chunk_size = 64

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
    (
        o_chunk_parallel,
        s_chunk_parallel,
        state_array_chunk_parallel,
        log_pi,
        log_rho,
    ) = lavd_chunk_parallel_torch(
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
    (
        o_chunk_parallel_triton,
        s_chunk_parallel_triton,
        state_array_chunk_parallel_triton,
        log_pi_triton,
        log_rho_triton,
    ) = lavd_chunk_parallel_triton(
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
    # for i in range(n):
    #     print(i)
    #     start = i
    #     end = i + 1
    #     print("o diff max: ", torch.abs(o_chunk_parallel[0, start:end, 0] - o_chunk_parallel_triton[0, start:end, 0]).max().item())
    #     print("o diff norm: ", torch.norm(o_chunk_parallel[0, start:end, 0] - o_chunk_parallel_triton[0, start:end, 0]).item())
    assert torch.allclose(
        o_chunk_parallel, o_chunk_parallel_triton, atol=atol, rtol=rtol
    )

    # print("s diff max: ", torch.abs(s_chunk_parallel - s_chunk_parallel_triton).max().item())
    # print("s diff norm: ", torch.norm(s_chunk_parallel - s_chunk_parallel_triton).item())
    # assert torch.allclose(s_chunk_parallel, s_chunk_parallel_triton, atol=atol, rtol=rtol)

    m = (n + chunk_size - 1) // chunk_size + 1
    print(state_array_chunk_parallel.shape)
    for i in range(m):
        print(i)
        print(
            "state_array diff max: ",
            torch.abs(
                state_array_chunk_parallel[:, i]
                - state_array_chunk_parallel_triton[:, i]
            )
            .max()
            .item(),
        )
        print(
            "state_array diff norm: ",
            torch.norm(
                state_array_chunk_parallel[:, i]
                - state_array_chunk_parallel_triton[:, i]
            ).item(),
        )
    print(
        "state_array diff max: ",
        torch.abs(state_array_chunk_parallel - state_array_chunk_parallel_triton)
        .max()
        .item(),
    )
    print(
        "state_array diff norm: ",
        torch.norm(
            state_array_chunk_parallel - state_array_chunk_parallel_triton
        ).item(),
    )
    assert torch.allclose(
        state_array_chunk_parallel,
        state_array_chunk_parallel_triton,
        atol=atol,
        rtol=rtol,
    )

    # for i in range(n):
    #     print(i)
    #     start = i
    #     end = i + 1
    #     print("log_pi diff max: ", torch.abs(log_pi[:, start:end] - log_pi_triton[:, start:end]).max().item())
    #     print("log_pi diff norm: ", torch.norm(log_pi[:, start:end] - log_pi_triton[:, start:end]).item())
    #     assert torch.allclose(log_pi[:, start:end], log_pi_triton[:, start:end], atol=atol, rtol=rtol)
    print("log_pi diff max: ", torch.abs(log_pi - log_pi_triton).max().item())
    print("log_pi diff norm: ", torch.norm(log_pi - log_pi_triton).item())
    assert torch.allclose(log_pi, log_pi_triton, atol=atol, rtol=rtol)

    print("log_rho diff max: ", torch.abs(log_rho - log_rho_triton).max().item())
    print("log_rho diff norm: ", torch.norm(log_rho - log_rho_triton).item())
    assert torch.allclose(log_rho, log_rho_triton, atol=atol, rtol=rtol)
