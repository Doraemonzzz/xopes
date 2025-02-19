import pytest
import torch
import torch.nn.functional as F

from xopes.ops.lightning_attn.scalar_decay import lasd_recurrence_triton, lasd_torch
from xopes.utils import get_threshold


def get_params():
    shapes = [
        (2, 128, 8, 64, 32),
        (2, 127, 16, 64, 128),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
# @pytest.mark.parametrize("use_initial_state", [True, False])
# @pytest.mark.parametrize("use_varlen", [True, False])
# @pytest.mark.parametrize("act", ["none", "relu", "sigmoid", "silu", "softmax"])
# @pytest.mark.parametrize("norm", [True, False])
# @pytest.mark.parametrize("use_varlen", [True])
@pytest.mark.parametrize("use_initial_state", [False])
@pytest.mark.parametrize("use_varlen", [False])
@pytest.mark.parametrize(
    "act",
    [
        "none",
        "relu",
    ],
)
@pytest.mark.parametrize("norm", [False])
@pytest.mark.parametrize("dtype", [torch.float32])
def test(shape, use_initial_state, use_varlen, act, norm, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, n, h, d, e = shape
    q_act = act
    k_act = act
    v_act = act
    q_norm = norm
    k_norm = norm
    v_norm = norm

    if use_varlen:
        b = 1
        m = n // 5
        cu_seqlens = torch.tensor(
            [0, m - 2, 2 * m + 1, 3 * m - 1, 4 * m, n], dtype=torch.long, device=device
        )
    else:
        cu_seqlens = None

    # Generate input tensors
    q = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
    ld = F.logsigmoid(torch.randn(h, dtype=dtype, device=device)).requires_grad_()

    if use_initial_state:
        initial_state = torch.randn(
            (b, h, d, e), dtype=dtype, device=device
        ).requires_grad_()
    else:
        initial_state = None

    # Forward pass - PyTorch reference implementation
    o_torch, s_torch = lasd_torch(
        q=q,
        k=k,
        v=v,
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        q_act=q_act,
        k_act=k_act,
        v_act=v_act,
        q_norm=q_norm,
        k_norm=k_norm,
        v_norm=v_norm,
    )

    # Forward pass - Triton implementation
    o_triton, s_triton = lasd_recurrence_triton(
        q=q,
        k=k,
        v=v,
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        q_act=q_act,
        k_act=k_act,
        v_act=v_act,
        q_norm=q_norm,
        k_norm=k_norm,
        v_norm=v_norm,
    )

    atol, rtol = get_threshold(dtype)

    # Check forward pass results
    print(
        "o diff max: ",
        torch.abs(o_torch - o_triton).max().item(),
    )
    print("o diff norm: ", torch.norm(o_torch - o_triton).item())
    # assert torch.allclose(
    #     o_torch, o_triton, atol=atol, rtol=rtol
    # )

    print(
        "s diff max: ",
        torch.abs(s_torch - s_triton).max().item(),
    )
    print("s diff norm: ", torch.norm(s_torch - s_triton).item())
    assert torch.allclose(s_torch, s_triton, atol=atol, rtol=rtol)
