import pytest
import torch

from xopes.ops import (
    additive_rule_recurrence_stable_torch,
    additive_rule_recurrence_torch,
    additive_rule_recurrence_triton,
)


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
# @pytest.mark.parametrize("dtype", [torch.float32])
# @pytest.mark.parametrize("dtype", [torch.float16])
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_initial_state", [True, False])
@pytest.mark.parametrize("output_final_state", [True, False])
# @pytest.mark.parametrize("use_initial_state", [True, ])
# @pytest.mark.parametrize("output_final_state", [True, ])
def test_lightning2(b, h, n, d, e, dtype, use_initial_state, output_final_state):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    q = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    k = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    v = (torch.randn((b, h, n, e), dtype=dtype, device=device)).requires_grad_()
    g = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    do = torch.randn((b, h, n, e), dtype=dtype, device=device)

    if use_initial_state:
        s_initial_state = torch.randn((b, h, d, e), dtype=dtype, device=device)
        denom_initial_state = torch.randn((b, h, d, 1), dtype=dtype, device=device) ** 2
        m_initial_state = torch.randn((b, h, d, 1), dtype=dtype, device=device)
        initial_state = (s_initial_state, denom_initial_state, m_initial_state)
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
    o_recurrence_torch, final_state_recurrence_torch = additive_rule_recurrence_torch(
        q, k, v, g, initial_state, output_final_state=output_final_state
    )
    (
        o_recurrence_stable_torch,
        final_state_recurrence_stable_torch,
    ) = additive_rule_recurrence_stable_torch(
        q, k, v, g, initial_state, output_final_state=output_final_state
    )
    (
        o_recurrence_triton,
        final_state_recurrence_triton,
    ) = additive_rule_recurrence_triton(
        q, k, v, g, initial_state, output_final_state=output_final_state
    )

    # # backward
    # o_ref.backward(do, retain_graph=True)
    # dq_ref, q.grad = q.grad.clone(), None
    # dk_ref, k.grad = k.grad.clone(), None
    # dv_ref, v.grad = v.grad.clone(), None

    # o.backward(do, retain_graph=True)
    # dq, q.grad = q.grad.clone(), None
    # dk, k.grad = k.grad.clone(), None
    # dv, v.grad = v.grad.clone(), None

    print(f"{'==' * 10} Output test {'==' * 10}")
    # print(f"recurrence torch Vs recurrence stable torch: {torch.norm(o_recurrence_torch - o_recurrence_stable_torch)}")
    # print(torch.min(o_recurrence_stable_torch), torch.max(o_recurrence_stable_torch))
    # print(torch.min(o_recurrence_triton), torch.max(o_recurrence_triton))
    print(
        f"recurrence stable torch Vs recurrence triton(diff norm): {torch.norm(o_recurrence_stable_torch - o_recurrence_triton).item()}"
    )
    print(
        f"recurrence stable torch Vs recurrence triton(diff max): {torch.abs(o_recurrence_stable_torch - o_recurrence_triton).max()}"
    )

    assert torch.allclose(
        o_recurrence_triton, o_recurrence_stable_torch, atol=atol, rtol=rtol
    )
    # assert torch.allclose(o_recurrence_torch, o_recurrence_triton, atol=1e-2, rtol=rtol)

    if output_final_state:
        print(f"{'==' * 10} State test {'==' * 10}")
        n = len(final_state_recurrence_stable_torch)
        print(f"The number of states: {n}")
        for i in range(n):
            print(
                torch.norm(
                    final_state_recurrence_stable_torch[i]
                    - final_state_recurrence_triton[i]
                ).item()
            )
            # print(
            #     torch.min(final_state_recurrence_stable_torch[i]).item(),
            #     torch.min(final_state_recurrence_triton[i]).item(),
            # )
            # print(
            #     torch.max(final_state_recurrence_stable_torch[i]).item(),
            #     torch.max(final_state_recurrence_triton[i]).item(),
            # )

            assert torch.allclose(
                final_state_recurrence_stable_torch[i],
                final_state_recurrence_triton[i],
                atol=atol,
                rtol=rtol,
            )

        # assert False
    # assert False

    # print(torch.norm(o_recurrence_torch - o_recurrence_triton))
    # print(torch.norm(dq - dq_ref))
    # print(torch.norm(dk - dk_ref))
    # print(torch.norm(dv - dv_ref))
    # assert False
