import pytest
import torch

from xopes.ops.normalize import normalize_torch, normalize_triton
from xopes.utils import get_threshold


def f(x):
    return x + 1.0


def naive_prenorm(
    x, l=6, weight=None, bias=None, c=1.0, eps=1e-5, use_mean=False, num_groups=1
):
    dtype = x.dtype
    x = x.float()
    for i in range(l):
        r = x
        x_norm, _ = normalize_torch(
            x,
            residual=r,
            weight=weight,
            bias=bias,
            c=c,
            eps=eps,
            use_mean=use_mean,
            num_groups=num_groups,
            return_residual=False,
        )
        x = f(x_norm) + r

    return x.to(dtype)


def fuse_torch_prenorm(
    x, l=6, weight=None, bias=None, c=1.0, eps=1e-5, use_mean=False, num_groups=1
):
    dtype = x.dtype
    p = x
    r = None
    for i in range(l):
        q, r = normalize_torch(
            p,
            residual=r,
            weight=weight,
            bias=bias,
            c=c,
            eps=eps,
            use_mean=use_mean,
            num_groups=num_groups,
            return_residual=True,
        )
        p = f(q)

    o = p.float() + r.float()

    return o.to(dtype)


def fuse_triton_prenorm(
    x, l=6, weight=None, bias=None, c=1.0, eps=1e-5, use_mean=False, num_groups=1
):
    dtype = x.dtype
    p = x
    r = None
    for i in range(l):
        q, r = normalize_triton(
            p,
            residual=r,
            weight=weight,
            bias=bias,
            c=c,
            eps=eps,
            use_mean=use_mean,
            num_groups=num_groups,
            return_residual=True,
        )
        p = f(q)

    o = p.float() + r.float()

    return o.to(dtype)


def get_params():
    shape = [(6, 128), (4, 8, 256), (6, 2048, 768)]

    return shape


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("num_groups", [1, 4])
@pytest.mark.parametrize("use_mean", [True, False])
@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("c", [1, 16])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test(shape, num_groups, use_mean, use_weight, use_bias, c, eps, dtype, l=6):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    d = shape[-1]
    x = torch.randn(shape, dtype=dtype, device=device).requires_grad_()
    do = torch.randn(shape, dtype=dtype, device=device)

    if use_weight:
        weight = torch.randn((d,), dtype=dtype, device=device).requires_grad_()
    else:
        weight = None

    if use_bias:
        bias = torch.randn((d,), dtype=dtype, device=device).requires_grad_()
    else:
        bias = None

    # forward
    o_naive_prenorm = naive_prenorm(
        x,
        l=l,
        weight=weight,
        bias=bias,
        c=c,
        eps=eps,
        use_mean=use_mean,
        num_groups=num_groups,
    )

    o_fuse_torch_prenorm = fuse_torch_prenorm(
        x,
        l=l,
        weight=weight,
        bias=bias,
        c=c,
        eps=eps,
        use_mean=use_mean,
        num_groups=num_groups,
    )

    o_fuse_triton_prenorm = fuse_triton_prenorm(
        x,
        l=l,
        weight=weight,
        bias=bias,
        c=c,
        eps=eps,
        use_mean=use_mean,
        num_groups=num_groups,
    )

    # backward
    o_naive_prenorm.backward(do, retain_graph=True)
    dx_naive_prenorm, x.grad = x.grad.clone(), None
    if use_weight:
        dw_naive_prenorm, weight.grad = weight.grad.clone(), None
    else:
        dw_naive_prenorm = None

    if use_bias:
        db_naive_prenorm, bias.grad = bias.grad.clone(), None
    else:
        db_naive_prenorm = None

    o_fuse_torch_prenorm.backward(do, retain_graph=True)
    dx_fuse_torch_prenorm, x.grad = x.grad.clone(), None
    if use_weight:
        dw_fuse_torch_prenorm, weight.grad = weight.grad.clone(), None
    else:
        dw_fuse_torch_prenorm = None

    if use_bias:
        db_fuse_torch_prenorm, bias.grad = bias.grad.clone(), None
    else:
        db_fuse_torch_prenorm = None

    o_fuse_triton_prenorm.backward(do, retain_graph=True)
    dx_fuse_triton_prenorm, x.grad = x.grad.clone(), None
    if use_weight:
        dw_fuse_triton_prenorm, weight.grad = weight.grad.clone(), None
    else:
        dw_fuse_triton_prenorm = None

    if use_bias:
        db_fuse_triton_prenorm, bias.grad = bias.grad.clone(), None
    else:
        db_fuse_triton_prenorm = None

    atol, rtol = get_threshold(dtype)

    ##### fwd
    print(
        "o diff max (naive - fuse_torch): ",
        torch.abs(o_naive_prenorm - o_fuse_torch_prenorm).max().item(),
    )
    print(
        "o diff norm (naive - fuse_torch): ",
        torch.norm(o_naive_prenorm - o_fuse_torch_prenorm).item(),
    )
    assert torch.allclose(o_naive_prenorm, o_fuse_torch_prenorm, atol=atol, rtol=rtol)

    print(
        "o diff max (naive - fuse_triton): ",
        torch.abs(o_naive_prenorm - o_fuse_triton_prenorm).max().item(),
    )
    print(
        "o diff norm (naive - fuse_triton): ",
        torch.norm(o_naive_prenorm - o_fuse_triton_prenorm).item(),
    )
    assert torch.allclose(o_naive_prenorm, o_fuse_triton_prenorm, atol=atol, rtol=rtol)

    ##### bwd
    print(
        "dx diff max (naive - fuse_torch): ",
        torch.abs(dx_naive_prenorm - dx_fuse_torch_prenorm).max().item(),
    )
    print(
        "dx diff norm (naive - fuse_torch): ",
        torch.norm(dx_naive_prenorm - dx_fuse_torch_prenorm).item(),
    )
    assert torch.allclose(dx_naive_prenorm, dx_fuse_torch_prenorm, atol=atol, rtol=rtol)

    print(
        "dx diff max (naive - fuse_triton): ",
        torch.abs(dx_naive_prenorm - dx_fuse_triton_prenorm).max().item(),
    )
    print(
        "dx diff norm (naive - fuse_triton): ",
        torch.norm(dx_naive_prenorm - dx_fuse_triton_prenorm).item(),
    )
    assert torch.allclose(
        dx_naive_prenorm, dx_fuse_triton_prenorm, atol=atol, rtol=rtol
    )

    if use_weight:
        print(
            "dw diff max (naive - fuse_torch): ",
            torch.abs(dw_naive_prenorm - dw_fuse_torch_prenorm).max().item(),
        )
        print(
            "dw diff norm (naive - fuse_torch): ",
            torch.norm(dw_naive_prenorm - dw_fuse_torch_prenorm).item(),
        )
        assert torch.allclose(
            dw_naive_prenorm, dw_fuse_torch_prenorm, atol=atol, rtol=rtol
        )

        print(
            "dw diff max (naive - fuse_triton): ",
            torch.abs(dw_naive_prenorm - dw_fuse_triton_prenorm).max().item(),
        )
        print(
            "dw diff norm (naive - fuse_triton): ",
            torch.norm(dw_naive_prenorm - dw_fuse_triton_prenorm).item(),
        )
        assert torch.allclose(
            dw_naive_prenorm, dw_fuse_triton_prenorm, atol=atol, rtol=rtol
        )

    if use_bias:
        print(
            "db diff max (naive - fuse_torch): ",
            torch.abs(db_naive_prenorm - db_fuse_torch_prenorm).max().item(),
        )
        print(
            "db diff norm (naive - fuse_torch): ",
            torch.norm(db_naive_prenorm - db_fuse_torch_prenorm).item(),
        )
        assert torch.allclose(
            db_naive_prenorm, db_fuse_torch_prenorm, atol=atol, rtol=rtol
        )

        print(
            "db diff max (naive - fuse_triton): ",
            torch.abs(db_naive_prenorm - db_fuse_triton_prenorm).max().item(),
        )
        print(
            "db diff norm (naive - fuse_triton): ",
            torch.norm(db_naive_prenorm - db_fuse_triton_prenorm).item(),
        )
        assert torch.allclose(
            db_naive_prenorm, db_fuse_triton_prenorm, atol=atol, rtol=rtol
        )
