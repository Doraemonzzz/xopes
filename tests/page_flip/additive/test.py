import pytest
import torch
import torch.nn.functional as F
from xopes.ops.page_flip.additive import (page_flip_additive_naive_torch,
                                          page_flip_additive_recurrence_torch,
                                          page_flip_additive_recurrence_triton)
from xopes.utils import get_threshold


def get_params():
    array = [
        # standard shape
        (6, 128, 8, 128, 64),
        # (1, 2, 1, 1, 1),
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
        True,
        # False,
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
    [True, False],
)
@pytest.mark.parametrize(
    "use_k",
    [True, False],
)
@pytest.mark.parametrize(
    "b_state",
    # [True, False],
    [True],
)
def test(
    b,
    n,
    h,
    d,
    e,
    dtype,
    use_initial_state,
    output_final_state,
    use_normalize,
    use_k,
    b_state,
):
    if (not use_k) and (not use_normalize):
        return

    torch.manual_seed(2024)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    f = F.softplus
    q = (torch.randn((b, n, h, d), dtype=dtype, device=device)).requires_grad_()
    if use_k:
        k = (torch.randn((b, n, h, d), dtype=dtype, device=device)).requires_grad_()
    else:
        k = None
    v = (torch.randn((b, n, h, e), dtype=dtype, device=device)).requires_grad_()
    w = (f(torch.randn((b, n, h, d), dtype=dtype, device=device))).requires_grad_()
    # do = torch.randn((b, n, h, e), dtype=dtype, device=device)

    if use_initial_state:
        if b_state:
            state1 = f(torch.randn((b, h, d), dtype=dtype, device=device))
            state2 = f(torch.randn((b, h, d), dtype=dtype, device=device))
            state3 = torch.randn((b, h, d, e), dtype=dtype, device=device)
            state4 = torch.randn((b, h, d, e), dtype=dtype, device=device)
        else:
            state1 = f(torch.randn((h, d), dtype=dtype, device=device))
            state2 = f(torch.randn((h, d), dtype=dtype, device=device))
            state3 = torch.randn((h, d, e), dtype=dtype, device=device)
            state4 = torch.randn((h, d, e), dtype=dtype, device=device)
        initial_state = [state1, state2, state3, state4]
    else:
        initial_state = None

    # forward
    # naive torch
    if not use_initial_state:
        o_naive_torch, final_state_naive_torch = page_flip_additive_naive_torch(
            q,
            v,
            w,
            k=k,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_normalize=use_normalize,
        )
    print("before", initial_state[-2].mean())
    # recurrence torch
    (
        o_recurrence_torch,
        final_state_recurrence_torch,
    ) = page_flip_additive_recurrence_torch(
        q,
        v,
        w,
        k=k,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_normalize=use_normalize,
    )
    print("after", initial_state[-2].mean())

    # recurrence triton
    (
        o_recurrence_triton,
        final_state_recurrence_triton,
    ) = page_flip_additive_recurrence_triton(
        q,
        v,
        w,
        k=k,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_normalize=use_normalize,
    )

    atol, rtol = get_threshold(dtype)
    check_naive = (not use_initial_state) and use_normalize and use_k

    # print(f"{'==' * 10} Output test {'==' * 10}")
    # if check_naive:
    #     print(
    #         f"naive torch Vs recurrence torch (diff norm): {torch.norm(o_naive_torch - o_recurrence_torch).item()}"
    #     )
    #     print(
    #         f"naive torch Vs recurrence torch (diff max): {torch.abs(o_naive_torch - o_recurrence_torch).max()}"
    #     )
    #     assert torch.allclose(o_naive_torch, o_recurrence_torch, atol=atol, rtol=rtol)

    # print(
    #     f"recurrence triton Vs recurrence torch (diff norm): {torch.norm(o_recurrence_triton - o_recurrence_torch).item()}"
    # )
    # print(
    #     f"recurrence triton Vs recurrence torch (diff max): {torch.abs(o_recurrence_triton - o_recurrence_torch).max()}"
    # )
    # assert torch.allclose(o_recurrence_triton, o_recurrence_torch, atol=atol, rtol=rtol)

    if output_final_state:
        if check_naive:
            print(
                f"naive torch state0 Vs recurrence torch state1 (diff norm): {torch.norm(final_state_naive_torch[0] - final_state_recurrence_torch[1]).item()}"
            )
            print(
                f"naive torch state0 Vs recurrence torch state1 (diff max): {torch.norm(final_state_naive_torch[0] - final_state_recurrence_torch[1]).max()}"
            )
            print(
                f"naive torch state1 Vs recurrence torch state3 (diff norm): {torch.norm(final_state_naive_torch[1] - final_state_recurrence_torch[3]).item()}"
            )
            print(
                f"naive torch state1 Vs recurrence torch state3 (diff max): {torch.norm(final_state_naive_torch[1] - final_state_recurrence_torch[3]).max()}"
            )
            assert torch.allclose(
                final_state_naive_torch[0],
                final_state_recurrence_torch[1],
                atol=atol,
                rtol=rtol,
            )
            assert torch.allclose(
                final_state_naive_torch[1],
                final_state_recurrence_torch[3],
                atol=atol,
                rtol=rtol,
            )

        for i in range(4):
            print(
                f"recurrence triton state{i} Vs recurrence torch state{i} (diff norm): {torch.norm(final_state_recurrence_triton[i] - final_state_recurrence_torch[i]).item()}"
            )
            print(
                f"recurrence triton state{i} Vs recurrence torch state{i} (diff max): {torch.norm(final_state_recurrence_triton[i] - final_state_recurrence_torch[i]).max()}"
            )
            assert torch.allclose(
                final_state_recurrence_triton[i],
                final_state_recurrence_torch[i],
                atol=atol,
                rtol=rtol,
            )
