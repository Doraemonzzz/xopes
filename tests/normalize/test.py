import pytest
import torch

from xopes.ops.normalize import normalize_torch, normalize_triton
from xopes.utils import get_threshold


def get_params():
    shape = [
        (6, 128),
    ]

    return shape


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("num_groups", [1, 4, 8])
@pytest.mark.parametrize("c", [1, 128.0])
@pytest.mark.parametrize("eps", [1e-5])
# @pytest.mark.parametrize("use_mean", [True, False])
# @pytest.mark.parametrize("use_weight", [True, False])
# @pytest.mark.parametrize("use_bias", [True, False])
# @pytest.mark.parametrize("use_residual", [True, False])
# @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("use_mean", [False])
# @pytest.mark.parametrize("use_mean", [True, False])
# @pytest.mark.parametrize("use_weight", [True, False])
# @pytest.mark.parametrize("use_bias", [True, False])
# @pytest.mark.parametrize("use_residual", [True, False])
@pytest.mark.parametrize("use_mean", [False])
@pytest.mark.parametrize("use_weight", [False])
@pytest.mark.parametrize("use_bias", [False])
@pytest.mark.parametrize("use_residual", [False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test(
    shape, num_groups, use_mean, use_weight, use_bias, use_residual, c, eps, dtype
):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, d = shape
    x = torch.randn((b, d), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((b, d), dtype=dtype, device=device)

    if use_weight:
        weight = torch.randn((d), dtype=dtype, device=device)
    else:
        weight = None

    if use_bias:
        bias = torch.randn((d), dtype=dtype, device=device)
    else:
        bias = None

    if use_residual:
        residual = torch.randn((b, d), dtype=dtype, device=device)
    else:
        residual = None

    # forward
    o_normalize_torch = normalize_torch(
        x,
        weight=weight,
        bias=bias,
        residual=residual,
        c=c,
        eps=eps,
        use_mean=use_mean,
        num_groups=num_groups,
    )
    o_normalize_triton = normalize_triton(
        x,
        weight=weight,
        bias=bias,
        residual=residual,
        c=c,
        eps=eps,
        use_mean=use_mean,
        num_groups=num_groups,
    )

    # # backward
    # o_normalize_torch.backward(do, retain_graph=True)
    # dx_normalize_torch = x.grad.clone()
    # if use_weight:
    #     dw_normalize_torch = weight.grad.clone()
    # else:
    #     dw_normalize_torch = None

    # if use_bias:
    #     db_normalize_torch = bias.grad.clone()
    # else:
    #     db_normalize_torch = None

    # if use_residual:
    #     dr_normalize_torch = residual.grad.clone()
    # else:
    #     dr_normalize_torch = None

    # x.grad = None

    # o_normalize_triton.backward(do, retain_graph=True)
    # dx_normalize_triton = x.grad.clone()
    # if use_weight:
    #     dw_normalize_triton = weight.grad.clone()
    # else:
    #     dw_normalize_triton = None

    # if use_bias:
    #     db_normalize_triton = bias.grad.clone()
    # else:
    #     db_normalize_triton = None

    # if use_residual:
    #     dr_normalize_triton = residual.grad.clone()
    # else:
    #     dr_normalize_triton = None

    atol, rtol = get_threshold(dtype)
    print(
        "o diff max: ", torch.abs(o_normalize_torch - o_normalize_triton).max().item()
    )
    print("o diff norm: ", torch.norm(o_normalize_torch - o_normalize_triton).item())
    assert torch.allclose(o_normalize_torch, o_normalize_triton, atol=atol, rtol=rtol)

    # assert torch.allclose(
    #     dx_normalize_torch, dx_normalize_triton, atol=atol, rtol=rtol
    # ), f"dx diff: {torch.abs(dx_normalize_torch - dx_normalize_triton).max().item()}"

    # if use_weight:
    #     assert torch.allclose(
    #         dw_normalize_torch, dw_normalize_triton, atol=atol, rtol=rtol
    #     ), f"dw diff: {torch.abs(dw_normalize_torch - dw_normalize_triton).max().item()}"

    # if use_bias:
    #     assert torch.allclose(
    #         db_normalize_torch, db_normalize_triton, atol=atol, rtol=rtol
    #     ), f"db diff: {torch.abs(db_normalize_torch - db_normalize_triton).max().item()}"

    # if use_residual:
    #     assert torch.allclose(
    #         dr_normalize_torch, dr_normalize_triton, atol=atol, rtol=rtol
    #     ), f"dr diff: {torch.abs(dr_normalize_torch - dr_normalize_triton).max().item()}"
