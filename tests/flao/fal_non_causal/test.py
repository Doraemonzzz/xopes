import pytest
import torch

from xopes.ops.flao.fal_non_causal import (
    flao_al_non_causal_torch,
    flao_fal_non_causal_torch,
)


def get_params():
    shape = [
        (6, 8, 512, 256, 128, 64),
        (6, 8, 512, 512, 128, 128),
        (6, 8, 256, 256, 128, 128),
        (1, 1, 4, 4, 16, 16),
        (1, 1, 4, 4, 128, 128),
        (1, 1, 128, 128, 16, 16),
        (1, 1, 128, 128, 128, 128),
    ]

    return shape


@pytest.mark.parametrize("b, h, n, m, d, e", get_params())
# act
@pytest.mark.parametrize("q_act", ["silu"])
@pytest.mark.parametrize("q_act_dim", [None])
@pytest.mark.parametrize("k_act", ["silu"])
@pytest.mark.parametrize("k_act_dim", [None])
@pytest.mark.parametrize("v_act", ["none"])
@pytest.mark.parametrize("v_act_dim", [None])
@pytest.mark.parametrize("g_act", ["silu"])
@pytest.mark.parametrize("g_act_dim", [None])
# lrpe
@pytest.mark.parametrize("use_lrpe", [True, False])
@pytest.mark.parametrize("lrpe_type", ["cosine"])
@pytest.mark.parametrize("offset", [0])
@pytest.mark.parametrize("l", [0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(
    b,
    h,
    n,
    m,
    d,
    e,
    q_act,
    q_act_dim,
    k_act,
    k_act_dim,
    v_act,
    v_act_dim,
    g_act,
    g_act_dim,
    use_lrpe,
    lrpe_type,
    offset,
    l,
    dtype,
):
    torch.manual_seed(2024)
    device = torch.device("cuda")

    q = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((b, h, m, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, h, m, e), dtype=dtype, device=device).requires_grad_()
    g = (torch.randn((b, h, n, e), dtype=dtype, device=device) * 0.1).requires_grad_()
    do = torch.randn((b, h, n, e), dtype=dtype, device=device)

    if use_lrpe:
        theta = torch.randn((h, d), dtype=dtype, device=device)
        shape = None
    else:
        theta = None
        shape = None

    # forward
    o_flao_al_torch = flao_al_non_causal_torch(
        q,
        k,
        v,
        g,
        q_act,
        q_act_dim,
        k_act,
        k_act_dim,
        v_act,
        v_act_dim,
        g_act,
        g_act_dim,
        theta,
        shape,
        lrpe_type,
        offset,
        l,
    )
    o_flao_fal_torch = flao_fal_non_causal_torch(
        q,
        k,
        v,
        g,
        q_act,
        q_act_dim,
        k_act,
        k_act_dim,
        v_act,
        v_act_dim,
        g_act,
        g_act_dim,
        theta,
        shape,
        lrpe_type,
        offset,
        l,
    )

    # backward
    o_flao_al_torch.backward(do, retain_graph=True)
    dq_flao_al_torch, q.grad = q.grad.clone(), None
    dk_flao_al_torch, k.grad = k.grad.clone(), None
    dv_flao_al_torch, v.grad = v.grad.clone(), None
    dg_flao_al_torch, g.grad = g.grad.clone(), None

    o_flao_fal_torch.backward(do, retain_graph=True)
    dq_flao_fal_torch, q.grad = q.grad.clone(), None
    dk_flao_fal_torch, k.grad = k.grad.clone(), None
    dv_flao_fal_torch, v.grad = v.grad.clone(), None
    dg_flao_fal_torch, g.grad = g.grad.clone(), None

    atol, rtol = THRESHOLD_DICT = {
        torch.float32: [5e-2, 5e-2],
        torch.float16: [1e-1, 1e-1],
        torch.bfloat16: [1e-1, 1e-1],
    }[dtype]

    # forward
    assert torch.allclose(
        o_flao_al_torch, o_flao_fal_torch, atol=atol, rtol=rtol
    ), f"o diff: {torch.abs(o_flao_al_torch - o_flao_fal_torch).max().item()}"

    # backward
    assert torch.allclose(
        dq_flao_al_torch, dq_flao_fal_torch, atol=atol, rtol=rtol
    ), f"dq diff: {torch.abs(dq_flao_al_torch - dq_flao_fal_torch).max().item()}"
    assert torch.allclose(
        dk_flao_al_torch, dk_flao_fal_torch, atol=atol, rtol=rtol
    ), f"dk diff: {torch.abs(dk_flao_al_torch - dk_flao_fal_torch).max().item()}"
    assert torch.allclose(
        dv_flao_al_torch, dv_flao_fal_torch, atol=atol, rtol=rtol
    ), f"dv diff: {torch.abs(dv_flao_al_torch - dv_flao_fal_torch).max().item()}"
    assert torch.allclose(
        dg_flao_al_torch, dg_flao_fal_torch, atol=atol, rtol=rtol
    ), f"dg diff: {torch.abs(dg_flao_al_torch - dg_flao_fal_torch).max().item()}"
