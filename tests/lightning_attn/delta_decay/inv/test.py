from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from xopes.ops.implicit_attn.inverse_attn import ilav_torch
from xopes.ops.lightning_attn.delta_decay import ladd_torch
from xopes.ops.lightning_attn.scalar_decay import lasd_torch
from xopes.utils import assert_close


def ladd_composed_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
):
    v_, state1 = ilav_torch(k, k * beta.unsqueeze(-1), v, ld, normalize=False)
    o, state2 = lasd_torch(q, k * beta.unsqueeze(-1), v_, ld)

    return o, state2


def inverse_by_delta(
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    **kwargs,
):
    b, n, h, d = k.shape
    e = v.shape[-1]

    q_ = k * torch.exp(ld.unsqueeze(-1))
    zero_qk = torch.zeros(b, 1, h, d, device=k.device)
    zero_vo = torch.zeros(b, 1, h, e, device=k.device)
    q_ = torch.cat(
        [
            q_[:, 1:],
            zero_qk,
        ],
        dim=1,
    )

    o, state = ladd_torch(
        q=q_,
        k=k,
        v=v,
        ld=ld,
        beta=beta,
    )

    v = v - torch.cat(
        [
            zero_vo,
            o[:, :-1],
        ],
        dim=1,
    )

    return v, state


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
@pytest.mark.parametrize("use_initial_state", [False])
@pytest.mark.parametrize("use_varlen", [False])
@pytest.mark.parametrize(
    "no_dstate",
    [
        True,
    ],
)
@pytest.mark.parametrize("use_chunk_loop", [False])
@pytest.mark.parametrize("c", [1, 10])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_lasd(
    shape, use_initial_state, use_varlen, no_dstate, use_chunk_loop, c, dtype
):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    scale = 0.01

    b, n, h, d, e = shape

    if use_varlen:
        b = 1
        m = n // 5
        cu_seqlens = torch.tensor(
            [0, m - 2, 2 * m + 1, 3 * m - 1, 4 * m, n], dtype=torch.long, device=device
        )
    else:
        pass

    # Generate input tensors
    q = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    k = (
        F.normalize(torch.randn((b, n, h, d), dtype=dtype, device=device), dim=-1)
    ).requires_grad_()
    v = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
    ld = F.logsigmoid(
        (1 + scale * torch.randn((b, n, h), dtype=dtype, device=device)) * c
    ).requires_grad_()
    beta = F.sigmoid(
        torch.randn((b, n, h), dtype=dtype, device=device)
    ).requires_grad_()

    if no_dstate:
        do = torch.randn((b, n, h, e), dtype=dtype, device=device)
    else:
        do = torch.randn((), dtype=dtype, device=device)

    if use_initial_state:
        initial_state = torch.randn(
            (b, h, d, e), dtype=dtype, device=device
        ).requires_grad_()
    else:
        pass

    ##### Forward pass
    # Baseline implementation
    o_torch, s_torch = ladd_torch(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        ld=ld.clone(),
        beta=beta.clone(),
    )

    # Composed implementation
    o_comp, s_comp = ladd_composed_torch(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        ld=ld.clone(),
        beta=beta.clone(),
    )

    # inverse
    v_inv, s_inv = ilav_torch(
        q=k.clone(),
        k=k.clone() * beta.clone().unsqueeze(-1),
        o=v.clone(),
        ld=ld.clone(),
        normalize=False,
    )

    v_inv_by_delta, s_inv_by_delta = inverse_by_delta(
        k=k.clone(),
        v=v.clone(),
        ld=ld.clone(),
        beta=beta.clone(),
    )

    # atol, rtol = get_threshold(dtype)
    atol = 5e-3
    rtol = 5e-3

    ##### Check forward pass results
    score = torch.einsum("b h d, b h d -> b h", q[:, 0], k[:, 0])
    o_first = torch.einsum("b h, b h e -> b h e", score, v[:, 0])

    print(
        "o diff max (torch vs composed): ",
        torch.abs(o_torch - o_comp).max().item(),
    )
    print(
        "o diff norm (torch vs composed): ",
        torch.norm(o_torch - o_comp).item(),
    )
    assert_close(o_torch, o_comp, atol=atol, rtol=rtol)

    print(
        "v diff max (inv vs delta): ",
        torch.abs(v_inv - v_inv_by_delta).max().item(),
    )
    print(
        "v diff norm (inv vs delta): ",
        torch.norm(v_inv - v_inv_by_delta).item(),
    )
    assert_close(v_inv, v_inv_by_delta, atol=atol, rtol=rtol)
