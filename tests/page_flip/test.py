import pytest
import torch
import torch.nn.functional as F

from xopes.ops import (
    page_flip_recurrence_torch,
    page_flip_naive_torch,
    page_flip_recurrence_triton
)


def get_params():
    array = [
        # standard shape
        (6, 128, 8, 128, 64),
        # (6, 128, 8, 128, 128),
        # (6, 128, 8, 256, 128),
        # # special shape
        # (6, 128, 8, 127, 129),
        # (6, 230, 8, 127, 129),
    ]

    return array


@pytest.mark.parametrize("b, n, h, d, e", get_params())
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
        True,
    ],
)
@pytest.mark.parametrize(
    "use_normalize",
    [
        True,
    ],
)
def test_lightning2(b, n, h, d, e, dtype, use_initial_state, output_final_state, use_normalize):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    f = F.softplus
    q = (torch.randn((b, n, h, d), dtype=dtype, device=device)).requires_grad_()
    # k = (torch.randn((b, n, h, d), dtype=dtype, device=device)).requires_grad_()
    k = None
    v = (torch.randn((b, n, h, e), dtype=dtype, device=device)).requires_grad_()
    w = (f(torch.randn((b, n, h, d), dtype=dtype, device=device))).requires_grad_()
    # do = torch.randn((b, n, h, e), dtype=dtype, device=device)

    if use_initial_state:
        state1 = f(torch.randn((b, h, d), dtype=dtype, device=device))
        state2 = f(torch.randn((b, h, d), dtype=dtype, device=device))
        state3 = torch.randn((b, h, d, e), dtype=dtype, device=device)
        state4 = torch.randn((b, h, d, e), dtype=dtype, device=device)
        initial_state = [state1, state2, state3, state4]
    else:
        initial_state = None

    if dtype in [torch.float32]:
        atol = 1e-2
        rtol = 1e-2
    elif dtype in [torch.float16]:
        atol = 5e-2
        rtol = 5e-2
    else:
        atol = 1e-1
        rtol = 1e-1

    # forward
    # naive torch
    if (not use_initial_state) and use_normalize:
        o_naive_torch, final_state_naive_torch = page_flip_naive_torch(q, v, w, k=k, initial_state=initial_state, output_final_state=output_final_state, use_normalize=use_normalize)
    # recurrence torch
    o_recurrence_torch, final_state_recurrence_torch = page_flip_recurrence_torch(
        q, v, w, k=k, initial_state=initial_state, output_final_state=output_final_state
    )
    # # recurrence triton
    # (
    #     o_recurrence_triton,
    #     final_state_recurrence_triton,
    # ) = page_flip_recurrence_triton(
    #     q, v, w, k=k, initial_state=initial_state, output_final_state=output_final_state
    # )

    print(f"{'==' * 10} Output test {'==' * 10}")
    if (not use_initial_state) and use_normalize:
        print(
            f"naive torch Vs recurrence torch (diff norm): {torch.norm(o_naive_torch - o_recurrence_torch).item()}"
        )
        print(
            f"naive torch Vs recurrence torch (diff max): {torch.abs(o_naive_torch - o_recurrence_torch).max()}"
        )
    
    # print(
    #     f"recurrence torch Vs recurrence triton (diff norm): {torch.norm(o_recurrence_torch - o_recurrence_triton).item()}"
    # )
    # print(
    #     f"recurrence torch Vs recurrence triton (diff max): {torch.abs(o_recurrence_torch - o_recurrence_triton).max()}"
    # )

    if output_final_state:
        if not use_initial_state:
            print(f"recurrence torch state1 Vs recurrence torch state2 (diff norm): {torch.norm(final_state_naive_torch[0] - final_state_recurrence_torch[1]).item()}")
            print(f"recurrence torch state1 Vs recurrence torch state2 (diff max): {torch.norm(final_state_naive_torch[0] - final_state_recurrence_torch[1]).max()}")
            print(f"recurrence torch state2 Vs recurrence torch state4 (diff norm): {torch.norm(final_state_naive_torch[1] - final_state_recurrence_torch[3]).item()}")
            print(f"recurrence torch state2 Vs recurrence torch state4 (diff max): {torch.norm(final_state_naive_torch[1] - final_state_recurrence_torch[3]).max()}")
            
        # for i in range(4):
        #     print(f"recurrence torch state{i+1} Vs recurrence triton state{i+1} (diff norm): {torch.norm(final_state_recurrence_torch[i] - final_state_recurrence_triton[i]).item()}")
        #     print(f"recurrence torch state{i+1} Vs recurrence triton state{i+1} (diff max): {torch.norm(final_state_recurrence_torch[i] - final_state_recurrence_triton[i]).max()}")
            
        

    assert False
    # assert torch.allclose(
    #     o_recurrence_torch, o_recurrence_triton, atol=atol, rtol=rtol
    # )
