import pytest
import torch

from xopes.ops.normalize import normalize_torch, normalize_triton
from xopes.utils import get_threshold


def get_params():
    shape = [(6, 128), (4, 8, 256)]

    return shape


##### gate tests
@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("num_groups", [1, 4])
@pytest.mark.parametrize("use_mean", [True, False])
@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("gate_act", ["sigmoid", "relu", "silu"])
@pytest.mark.parametrize("gate_pos", ["pre", "post"])
@pytest.mark.parametrize("c", [16])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gate(
    shape,
    num_groups,
    use_mean,
    use_weight,
    use_bias,
    gate_act,
    gate_pos,
    c,
    eps,
    dtype,
):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    d = shape[-1]
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    do = torch.randn(shape, dtype=dtype, device=device)
    gate = torch.randn(shape, dtype=dtype, device=device).requires_grad_()

    if use_weight:
        weight = torch.randn((d,), dtype=dtype, device=device).requires_grad_()
    else:
        weight = None

    if use_bias:
        bias = torch.randn((d,), dtype=dtype, device=device).requires_grad_()
    else:
        bias = None

    # forward
    o_torch = normalize_torch(
        x,
        weight=weight,
        bias=bias,
        residual=None,
        gate=gate,
        gate_act=gate_act,
        gate_pos=gate_pos,
        c=c,
        eps=eps,
        use_mean=use_mean,
        num_groups=num_groups,
        return_residual=False,
    )

    o_triton = normalize_triton(
        x,
        weight=weight,
        bias=bias,
        residual=None,
        gate=gate,
        gate_act=gate_act,
        gate_pos=gate_pos,
        c=c,
        eps=eps,
        use_mean=use_mean,
        num_groups=num_groups,
        return_residual=False,
    )

    # backward
    o_torch.backward(do, retain_graph=True)
    dx_torch, x.grad = x.grad.clone(), None
    dgate_torch, gate.grad = gate.grad.clone(), None

    if use_weight:
        dw_torch, weight.grad = weight.grad.clone(), None
    else:
        dw_torch = None

    if use_bias:
        db_torch, bias.grad = bias.grad.clone(), None
    else:
        db_torch = None

    o_triton.backward(do, retain_graph=True)
    dx_triton, x.grad = x.grad.clone(), None
    dgate_triton, gate.grad = gate.grad.clone(), None

    if use_weight:
        dw_triton, weight.grad = weight.grad.clone(), None
    else:
        dw_triton = None

    if use_bias:
        db_triton, bias.grad = bias.grad.clone(), None
    else:
        db_triton = None

    atol, rtol = get_threshold(dtype)

    ##### fwd
    print("o diff max: ", torch.abs(o_torch - o_triton).max().item())
    print("o diff norm: ", torch.norm(o_torch - o_triton).item())
    assert torch.allclose(o_torch, o_triton, atol=atol, rtol=rtol)

    ##### bwd
    print("dx diff max: ", torch.abs(dx_torch - dx_triton).max().item())
    print("dx diff norm: ", torch.norm(dx_torch - dx_triton).item())
    assert torch.allclose(dx_torch, dx_triton, atol=atol, rtol=rtol)

    print("dgate diff max: ", torch.abs(dgate_torch - dgate_triton).max().item())
    print("dgate diff norm: ", torch.norm(dgate_torch - dgate_triton).item())
    assert torch.allclose(dgate_torch, dgate_triton, atol=atol, rtol=rtol)

    if use_weight:
        print("dw diff max: ", torch.abs(dw_torch - dw_triton).max().item())
        print("dw diff norm: ", torch.norm(dw_torch - dw_triton).item())
        assert torch.allclose(dw_torch, dw_triton, atol=atol, rtol=rtol)

    if use_bias:
        print("db diff max: ", torch.abs(db_torch - db_triton).max().item())
        print("db diff norm: ", torch.norm(db_torch - db_triton).item())
        assert torch.allclose(db_torch, db_triton, atol=atol, rtol=rtol)
