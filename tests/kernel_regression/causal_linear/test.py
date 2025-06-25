import pytest
import torch
import torch.nn.functional as F

from xopes.ops.kernel_regression.causal_linear.krcl_parallel_triton import (
    krcl_parallel_triton,
)
from xopes.ops.kernel_regression.causal_linear.krcl_recurrence_triton import (
    krcl_recurrence_triton,
)
from xopes.ops.kernel_regression.causal_linear.krcl_torch import krcl_torch
from xopes.utils import assert_close, get_threshold, print_diff


def get_params():
    """
    Generate test parameter combinations for different tensor shapes.
    Returns various shapes to test edge cases and typical usage scenarios.
    """
    shapes = [
        (2, 256, 12, 128, 64),
        (2, 1024, 8, 32, 16),
        (2, 257, 8, 64, 32),
        (2, 255, 8, 64, 32),
        (2, 65, 7, 33, 63),
        (2, 270, 8, 64, 32),
        (2, 270, 8, 97, 16),
        (2, 1125, 8, 107, 33),
        (8, 2048, 12, 128, 64),
        (2, 128, 12, 128, 64),
    ]

    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_q", [True])
@pytest.mark.parametrize("use_initial_state", [True, False])
@pytest.mark.parametrize("use_varlen", [False])
@pytest.mark.parametrize("no_dstate", [True])
@pytest.mark.parametrize("c", [0.1, 10])  # Scaling factor for log decay
@pytest.mark.parametrize("dtype", [torch.float32])
def test_krcl(shape, use_q, use_initial_state, use_varlen, no_dstate, c, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    scale = 0.01

    b, n, h, d, e = shape

    # Setup variable length sequences if requested
    if use_varlen:
        b = 1
        m = n // 5
        cu_seqlens = torch.tensor(
            [0, m - 2, 2 * m + 1, 3 * m - 1, 4 * m, n], dtype=torch.long, device=device
        )
    else:
        cu_seqlens = None

    # Generate input tensors
    if use_q:
        q = (
            F.normalize(torch.randn((b, n, h, d), dtype=dtype, device=device), dim=-1)
        ).requires_grad_()
    else:
        q = None
    k = (
        F.normalize(torch.randn((b, n, h, d), dtype=dtype, device=device), dim=-1)
    ).requires_grad_()
    v = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
    ld = F.logsigmoid(
        (1 + scale * torch.ones((b, n, h), dtype=dtype, device=device)) * c
    ).requires_grad_()
    # ld = torch.zeros_like(ld).requires_grad_()
    alpha = (
        torch.exp(
            F.logsigmoid(
                (1 + scale * torch.randn((b, n, h), dtype=dtype, device=device)) * c
            )
        )
    ).requires_grad_()
    beta = (
        torch.exp(
            F.logsigmoid(
                (1 + scale * torch.randn((b, n, h), dtype=dtype, device=device)) * c
            )
        )
    ).requires_grad_()
    BLOCK_N = 64

    # Setup gradient tensor for backward pass
    if no_dstate:
        do = torch.randn((b, n, h, e), dtype=dtype, device=device)
    else:
        do = torch.randn((), dtype=dtype, device=device)

    # Setup initial state if requested
    if use_initial_state:
        initial_state = torch.randn(
            (b, h, d, e), dtype=dtype, device=device
        ).requires_grad_()
    else:
        initial_state = None

    ##### Forward pass comparison
    # PyTorch reference implementation
    o_torch, s_torch = krcl_torch(
        q=q.clone() if use_q else None,
        k=k.clone(),
        v=v.clone(),
        ld=ld.clone(),
        alpha=alpha.clone(),
        beta=beta.clone(),
        initial_state=initial_state.clone() if initial_state is not None else None,
        cu_seqlens=cu_seqlens,
    )
    if no_dstate:
        output_torch = o_torch
    else:
        output_torch = o_torch.mean() + s_torch.mean()

    # Triton optimized implementation
    o_triton, s_triton = krcl_recurrence_triton(
        q=q.clone() if use_q else None,
        k=k.clone(),
        v=v.clone(),
        ld=ld.clone(),
        alpha=alpha.clone(),
        beta=beta.clone(),
        initial_state=initial_state.clone() if initial_state is not None else None,
        cu_seqlens=cu_seqlens,
    )
    if no_dstate:
        output_triton = o_triton
    else:
        output_triton = o_triton.mean() + s_triton.mean()

    # Triton parallel implementation
    o_triton_parallel, s_triton_parallel = krcl_parallel_triton(
        q=q.clone() if use_q else None,
        k=k.clone(),
        v=v.clone(),
        ld=ld.clone(),
        alpha=alpha.clone(),
        beta=beta.clone(),
        initial_state=initial_state.clone() if initial_state is not None else None,
        cu_seqlens=cu_seqlens,
        BLOCK_N=BLOCK_N,
    )
    if no_dstate:
        output_triton_parallel = o_triton_parallel
    else:
        output_triton_parallel = o_triton_parallel.mean() + s_triton_parallel.mean()

    ##### Backward pass comparison
    output_torch.backward(do, retain_graph=True)
    if use_q:
        dq_torch, q.grad = q.grad.clone(), None
    else:
        dq_torch = None
    dk_torch, k.grad = k.grad.clone(), None
    dv_torch, v.grad = v.grad.clone(), None
    dld_torch, ld.grad = ld.grad.clone(), None
    dalpha_torch, alpha.grad = alpha.grad.clone(), None
    dbeta_torch, beta.grad = beta.grad.clone(), None
    if use_initial_state:
        ds_torch, initial_state.grad = initial_state.grad.clone(), None

    # triton backward
    output_triton.backward(do, retain_graph=True)
    if use_q:
        dq_triton, q.grad = q.grad.clone(), None
    else:
        dq_triton = None
    dk_triton, k.grad = k.grad.clone(), None
    dv_triton, v.grad = v.grad.clone(), None
    dld_triton, ld.grad = ld.grad.clone(), None
    dalpha_triton, alpha.grad = alpha.grad.clone(), None
    dbeta_triton, beta.grad = beta.grad.clone(), None
    if use_initial_state:
        ds_triton, initial_state.grad = initial_state.grad.clone(), None

    # triton parallel backward
    output_triton_parallel.backward(do, retain_graph=True)
    if use_q:
        dq_triton_parallel, q.grad = q.grad.clone(), None
    else:
        pass
    dk_triton_parallel, k.grad = k.grad.clone(), None
    dv_triton_parallel, v.grad = v.grad.clone(), None
    dld_triton_parallel, ld.grad = ld.grad.clone(), None
    dalpha_triton_parallel, alpha.grad = alpha.grad.clone(), None
    dbeta_triton_parallel, beta.grad = beta.grad.clone(), None
    if use_initial_state:
        ds_triton_parallel, initial_state.grad = initial_state.grad.clone(), None

    # Set tolerance for numerical comparisons
    atol, rtol = get_threshold(dtype)
    ld_atol = 7e-2 if dtype == torch.bfloat16 else atol
    ld_rtol = 7e-2 if dtype == torch.bfloat16 else rtol

    ##### Forward pass validation
    # torch vs triton
    print("o diff max (torch vs triton): ", torch.abs(o_torch - o_triton).max().item())
    print("o diff norm (torch vs triton): ", torch.norm(o_torch - o_triton).item())
    print_diff(o_torch, o_triton, n)
    assert_close(o_torch, o_triton, atol=atol, rtol=rtol)

    print(
        "state diff max (torch vs triton): ", torch.abs(s_torch - s_triton).max().item()
    )
    print("state diff norm (torch vs triton): ", torch.norm(s_torch - s_triton).item())
    assert_close(s_torch, s_triton, atol=atol, rtol=rtol)

    # torch vs triton parallel
    print(
        "o diff max (torch vs triton parallel): ",
        torch.abs(o_torch - o_triton_parallel).max().item(),
    )
    print(
        "o diff norm (torch vs triton parallel): ",
        torch.norm(o_torch - o_triton_parallel).item(),
    )
    print_diff(o_torch, o_triton_parallel, n)
    assert_close(o_torch, o_triton_parallel, atol=atol, rtol=rtol)

    print(
        "s diff max (torch vs triton parallel): ",
        torch.abs(s_torch - s_triton_parallel).max().item(),
    )
    print(
        "s diff norm (torch vs triton parallel): ",
        torch.norm(s_torch - s_triton_parallel).item(),
    )
    assert_close(s_torch, s_triton_parallel, atol=atol, rtol=rtol)

    ##### Backward pass validation
    print(
        "dk diff max (torch vs triton): ", torch.abs(dk_torch - dk_triton).max().item()
    )
    print("dk diff norm (torch vs triton): ", torch.norm(dk_torch - dk_triton).item())
    print_diff(dk_torch, dk_triton, n)
    assert_close(dk_torch, dk_triton, atol=atol, rtol=rtol)

    print(
        "dv diff max (torch vs triton): ", torch.abs(dv_torch - dv_triton).max().item()
    )
    print("dv diff norm (torch vs triton): ", torch.norm(dv_torch - dv_triton).item())
    print_diff(dv_torch, dv_triton, n)
    assert_close(dv_torch, dv_triton, atol=atol, rtol=rtol)

    if use_q:
        print(
            "dq diff max (torch vs triton): ",
            torch.abs(dq_torch - dq_triton).max().item(),
        )
        print(
            "dq diff norm (torch vs triton): ", torch.norm(dq_torch - dq_triton).item()
        )
        print_diff(dq_torch, dq_triton, n)
        assert_close(dq_torch, dq_triton, atol=atol, rtol=rtol)

    print(
        "dld diff max (torch vs triton): ",
        torch.abs(dld_torch - dld_triton).max().item(),
    )
    print(
        "dld diff norm (torch vs triton): ", torch.norm(dld_torch - dld_triton).item()
    )
    assert_close(dld_torch, dld_triton, atol=ld_atol, rtol=ld_rtol)

    print(
        "dalpha diff max (torch vs triton): ",
        torch.abs(dalpha_torch - dalpha_triton).max().item(),
    )
    print(
        "dalpha diff norm (torch vs triton): ",
        torch.norm(dalpha_torch - dalpha_triton).item(),
    )
    assert_close(dalpha_torch, dalpha_triton, atol=atol, rtol=rtol)

    print(
        "dbeta diff max (torch vs triton): ",
        torch.abs(dbeta_torch - dbeta_triton).max().item(),
    )
    print(
        "dbeta diff norm (torch vs triton): ",
        torch.norm(dbeta_torch - dbeta_triton).item(),
    )
    assert_close(dbeta_torch, dbeta_triton, atol=atol, rtol=rtol)

    # Validate initial state gradients if applicable
    if use_initial_state:
        print(
            "ds diff max (torch vs triton): ",
            torch.abs(ds_torch - ds_triton).max().item(),
        )
        print(
            "ds diff norm (torch vs triton): ", torch.norm(ds_torch - ds_triton).item()
        )
        assert_close(ds_torch, ds_triton, atol=atol, rtol=rtol)

    # torch vs triton parallel
    print(
        "dq diff max (torch vs triton parallel): ",
        torch.abs(dq_torch - dq_triton_parallel).max().item(),
    )
    print(
        "dq diff norm (torch vs triton parallel): ",
        torch.norm((dq_torch - dq_triton_parallel)).item(),
    )
    print_diff(dq_torch, dq_triton_parallel, n)
    assert_close(dq_torch, dq_triton_parallel, atol=atol, rtol=rtol)

    print(
        "dk diff max (torch vs triton parallel): ",
        torch.abs(dk_torch - dk_triton_parallel).max().item(),
    )
    print(
        "dk diff norm (torch vs triton parallel): ",
        torch.norm(dk_torch - dk_triton_parallel).item(),
    )
    print_diff(dk_torch, dk_triton_parallel, n)
    assert_close(dk_torch, dk_triton_parallel, atol=atol, rtol=rtol)

    print(
        "dv diff max (torch vs triton parallel): ",
        torch.abs(dv_torch - dv_triton_parallel).max().item(),
    )
    print(
        "dv diff norm (torch vs triton parallel): ",
        torch.norm(dv_torch - dv_triton_parallel).item(),
    )
    print_diff(dv_torch, dv_triton_parallel, n)
    assert_close(dv_torch, dv_triton_parallel, atol=atol, rtol=rtol)

    print(
        "dld diff max (torch vs triton parallel): ",
        torch.abs(dld_torch - dld_triton_parallel).max().item(),
    )
    print(
        "dld diff norm (torch vs triton parallel): ",
        torch.norm(dld_torch - dld_triton_parallel).item(),
    )
    assert_close(dld_torch, dld_triton_parallel, atol=ld_atol, rtol=ld_rtol)

    print(
        "dalpha diff max (torch vs triton): ",
        torch.abs(dalpha_torch - dalpha_triton).max().item(),
    )
    print(
        "dalpha diff norm (torch vs triton): ",
        torch.norm(dalpha_torch - dalpha_triton).item(),
    )
    assert_close(dalpha_torch, dalpha_triton, atol=atol, rtol=rtol)

    print(
        "dbeta diff max (torch vs triton): ",
        torch.abs(dbeta_torch - dbeta_triton).max().item(),
    )
    print(
        "dbeta diff norm (torch vs triton): ",
        torch.norm(dbeta_torch - dbeta_triton).item(),
    )
    assert_close(dbeta_torch, dbeta_triton, atol=atol, rtol=rtol)
