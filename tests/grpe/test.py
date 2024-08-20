import pytest
import torch
import torch.nn.functional as F

from xopes.ops import grpe_recurrence_torch


def get_params():
    array = [
        # standard shape
        (6, 8, 128, 128, 64),
        (6, 8, 128, 128, 128),
        (6, 8, 128, 256, 128),
        # special shape
        (6, 8, 128, 127, 129),
        (6, 8, 230, 127, 129),
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
        False,
    ],
)
@pytest.mark.parametrize(
    "output_final_state",
    [
        False,
    ],
)
def test(b, h, n, d, e, dtype, use_initial_state, output_final_state):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    q = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    k = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    v = (torch.randn((b, h, n, e), dtype=dtype, device=device)).requires_grad_()
    alpha = F.logsigmoid(
        torch.randn((b, h, n, d), dtype=dtype, device=device)
    ).requires_grad_()
    beta = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
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
        q, k, v, alpha, beta, initial_state, output_final_state=output_final_state
    )

    assert False
