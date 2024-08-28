import pytest
import torch
import torch.nn.functional as F

from xopes.ops import grpe_recurrence_torch, grpe_recurrence_triton


def get_params():
    array = [
        # (6, 8, 1, 128, 64),
        # (6, 8, 2, 128, 64),
        # standard shape
        # (1, 1, 64, 128, 64),
        (6, 8, 128, 128, 64),
        # (6, 8, 128, 128, 128),
        # (6, 8, 10, 64, 64),
        # (6, 8, 20, 64, 64),
        # (6, 8, 128, 64, 64),
        # special shape
        # (6, 8, 128, 127, 129),
        # (6, 8, 230, 127, 129),
    ]

    return array


@pytest.mark.parametrize("b, h, n, d, e", get_params())
@pytest.mark.parametrize("dtype", [torch.float32])
# @pytest.mark.parametrize("dtype", [torch.float16])
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("use_initial_state", [True, False])
# @pytest.mark.parametrize("output_final_state", [True, False])
@pytest.mark.parametrize(
    "use_initial_state",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "output_final_state",
    [
        True,
    ],
)
@pytest.mark.parametrize("BLOCK_SIZE", [32])
def test(b, h, n, d, e, dtype, use_initial_state, output_final_state, BLOCK_SIZE):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    q = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    k = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    v = (torch.randn((b, h, n, e), dtype=dtype, device=device)).requires_grad_()
    lower_bound = 0.95
    alpha = torch.log(
        lower_bound
        + (1 - lower_bound)
        * F.sigmoid(torch.randn((b, h, n, d), dtype=dtype, device=device))
    ).requires_grad_()
    beta = torch.log(
        lower_bound
        + (1 - lower_bound)
        * F.sigmoid(torch.randn((b, h, n), dtype=dtype, device=device))
    ).requires_grad_()
    gamma = F.normalize(
        torch.randn((b, h, n, d), dtype=dtype, device=device)
    ).requires_grad_()
    do = torch.randn((b, h, n, e), dtype=dtype, device=device)

    if use_initial_state:
        initial_state = torch.randn((b, h, d, e), dtype=dtype, device=device)
    else:
        initial_state = None

    if dtype in [torch.float32]:
        pass
    elif dtype in [torch.float16]:
        pass
    else:
        pass

    # forward
    # naive recurrence torch
    o_recurrence_torch, final_state_recurrence_torch = grpe_recurrence_torch(
        q, k, v, alpha, beta, gamma, initial_state, output_final_state
    )
    o_recurrence_triton, final_state_recurrence_triton = grpe_recurrence_triton(
        q, k, v, alpha, beta, gamma, initial_state, output_final_state
    )
    # (
    #     o_block_recurrence_torch,
    #     final_state_block_recurrence_torch,
    # ) = grpe_block_recurrence_torch(
    #     q,
    #     k,
    #     v,
    #     alpha,
    #     beta,
    #     gamma,
    #     initial_state,
    #     output_final_state=output_final_state,
    #     BLOCK_SIZE=BLOCK_SIZE,
    # )

    print(torch.norm(o_recurrence_torch - o_recurrence_triton))
    print(torch.norm(final_state_recurrence_torch - final_state_recurrence_triton))
    # print(torch.norm(o_recurrence_torch - o_block_recurrence_torch))
    # print(torch.norm(final_state_recurrence_torch - final_state_recurrence_triton))

    # print(torch.norm(o_recurrence_torch[:, :, 0] - o_block_recurrence_torch[:, :, 0]))
    # print(torch.norm(o_recurrence_torch[:, :, 1] - o_block_recurrence_torch[:, :, 1]))
    # print(torch.norm(o_recurrence_torch[:, :, -1] - o_block_recurrence_torch[:, :, -1]))
    print(o_recurrence_torch[-1, -1, -1, -5:])
    print(o_recurrence_triton[-1, -1, -1, -5:])

    print(final_state_recurrence_torch[0, 0, :5, :5])
    print(final_state_recurrence_triton[0, 0, :5, :5])

    # print(o_recurrence_torch[-1, 0, -1, :])
    # print(o_recurrence_triton[-1, 0, -1, :])

    assert False
