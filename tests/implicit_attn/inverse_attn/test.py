import pytest
import torch
import torch.nn.functional as F

from xopes.ops.implicit_attn.inverse_attn.ilav_recurrence_triton import (
    ilav_recurrence_triton,
)
from xopes.ops.implicit_attn.inverse_attn.ilav_torch import ilav_torch
from xopes.utils import assert_close, print_diff


def get_params():
    """
    Generate test parameter combinations for different tensor shapes.
    Returns various shapes to test edge cases and typical usage scenarios.
    """
    shapes = [
        # Standard shapes
        (2, 256, 12, 128, 128),
        (2, 1024, 8, 32, 16),
        # BLOCK_N +- 1 (edge cases around block boundaries)
        (2, 257, 8, 64, 32),
        (2, 255, 8, 64, 32),
        (2, 65, 7, 33, 63),
        # BLOCK_N +- C (various offsets from block boundaries)
        (2, 270, 8, 64, 32),
        (2, 270, 8, 33, 16),
        (2, 1125, 8, 43, 33),
        # Training-like shape
        (8, 2048, 12, 64, 64),
        # debug
        (2, 128, 12, 128, 64),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_initial_state", [True, False])
@pytest.mark.parametrize("use_varlen", [False])  # Variable length sequences
@pytest.mark.parametrize(
    "no_dstate", [True, False]
)  # Whether to include state gradients
@pytest.mark.parametrize("c", [0.1, 10])  # Scaling factor for log decay
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("rms_norm", [False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])

# @pytest.mark.parametrize("shape", get_params())
# @pytest.mark.parametrize(
#     "use_initial_state",
#     [
#         False,
#     ],
# )
# @pytest.mark.parametrize("use_varlen", [False])  # Variable length sequences
# @pytest.mark.parametrize(
#     "no_dstate",
#     [
#         True,
#     ],
# )  # Whether to include state gradients
# @pytest.mark.parametrize("c", [10])  # Scaling factor for log decay
# @pytest.mark.parametrize("normalize", [False])
# @pytest.mark.parametrize("rms_norm", [False])
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_ilav(
    shape, use_initial_state, use_varlen, no_dstate, c, normalize, rms_norm, dtype
):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    scale = 0.01

    b, n, h, d, e = shape

    if not normalize and rms_norm:
        return

    # Setup variable length sequences if requested
    if use_varlen:
        b = 1
        m = n // 5
        cu_seqlens = torch.tensor(
            [0, m - 2, 2 * m + 1, 3 * m - 1, 4 * m, n], dtype=torch.long, device=device
        )
    else:
        cu_seqlens = None

    if rms_norm:
        c = d**0.5
    else:
        c = 1

    # Generate input tensors
    q = (
        F.normalize(torch.randn((b, n, h, d), dtype=dtype, device=device), dim=-1) * c
    ).requires_grad_()  # !!! important
    k = (
        F.normalize(torch.randn((b, n, h, d), dtype=dtype, device=device), dim=-1) * c
    ).requires_grad_()
    o = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
    ld = F.logsigmoid(
        (1 + scale * torch.randn((b, n, h), dtype=dtype, device=device)) * c
    ).requires_grad_()

    # Setup gradient tensor for backward pass
    if no_dstate:
        dv = torch.randn((b, n, h, e), dtype=dtype, device=device)
    else:
        dv = torch.randn((), dtype=dtype, device=device)

    # Setup initial state if requested
    if use_initial_state:
        initial_state = torch.randn(
            (b, h, d, e), dtype=dtype, device=device
        ).requires_grad_()
    else:
        initial_state = None

    ##### Forward pass comparison

    # PyTorch reference implementation
    v_torch, s_torch = ilav_torch(
        q=q.clone(),
        k=k.clone(),
        o=o.clone(),
        ld=ld.clone(),
        initial_state=initial_state.clone() if initial_state is not None else None,
        cu_seqlens=cu_seqlens,
        normalize=normalize,
        rms_norm=rms_norm,
    )
    if no_dstate:
        output_torch = v_torch
    else:
        output_torch = v_torch.mean() + s_torch.mean()

    # Triton optimized implementation
    v_triton, s_triton = ilav_recurrence_triton(
        q=q,
        k=k,
        o=o,
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        normalize=normalize,
        rms_norm=rms_norm,
    )
    if no_dstate:
        output_triton = v_triton
    else:
        output_triton = v_triton.mean() + s_triton.mean()

    ##### Backward pass comparison
    # PyTorch reference backward pass
    output_torch.backward(dv, retain_graph=True)
    dq_torch, q.grad = q.grad.clone(), None
    dk_torch, k.grad = k.grad.clone(), None
    do_torch, o.grad = o.grad.clone(), None
    dld_torch, ld.grad = ld.grad.clone(), None
    if use_initial_state:
        ds_torch, initial_state.grad = initial_state.grad.clone(), None

    # Triton optimized backward pass
    output_triton.backward(dv, retain_graph=True)
    dq_triton, q.grad = q.grad.clone(), None
    dk_triton, k.grad = k.grad.clone(), None
    do_triton, o.grad = o.grad.clone(), None
    dld_triton, ld.grad = ld.grad.clone(), None
    if use_initial_state:
        ds_triton, initial_state.grad = initial_state.grad.clone(), None

    # Set tolerance for numerical comparisons
    atol = 5e-3
    rtol = 5e-3
    ld_atol = 7e-2 if dtype == torch.bfloat16 else atol
    ld_rtol = 7e-2 if dtype == torch.bfloat16 else rtol

    ##### Forward pass validation
    print(
        "v diff max (torch vs triton): ",
        torch.abs(v_torch - v_triton).max().item(),
    )
    print(
        "v diff norm (torch vs triton): ",
        torch.norm(v_torch - v_triton).item(),
    )
    print_diff(v_torch, v_triton, n)
    assert_close(v_torch, v_triton, atol=atol, rtol=rtol)

    print(
        "state diff max (torch vs triton): ",
        torch.abs(s_torch - s_triton).max().item(),
    )
    print(
        "state diff norm (torch vs triton): ",
        torch.norm(s_torch - s_triton).item(),
    )
    assert_close(s_torch, s_triton, atol=atol, rtol=rtol)

    ##### Backward pass validation
    print(
        "dk diff max (torch vs triton): ",
        torch.abs(dk_torch - dk_triton).max().item(),
    )
    print("dk diff norm (torch vs triton): ", torch.norm(dk_torch - dk_triton).item())
    print_diff(dk_torch, dk_triton, n)
    assert_close(dk_torch, dk_triton, atol=atol, rtol=rtol)

    print(
        "do diff max (torch vs triton): ",
        torch.abs(do_torch - do_triton).max().item(),
    )
    print("do diff norm (torch vs triton): ", torch.norm(do_torch - do_triton).item())
    print_diff(do_torch, do_triton, n)
    assert_close(do_torch, do_triton, atol=atol, rtol=rtol)

    print(
        "dq diff max (torch vs triton): ",
        torch.abs(dq_torch - dq_triton).max().item(),
    )
    print("dq diff norm (torch vs triton): ", torch.norm(dq_torch - dq_triton).item())
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

    # Validate initial state gradients if applicable
    if use_initial_state:
        print(
            "ds diff max (torch vs triton): ",
            torch.abs(ds_torch - ds_triton).max().item(),
        )
        print(
            "ds diff norm (torch vs triton): ",
            torch.norm(ds_torch - ds_triton).item(),
        )
        assert_close(ds_torch, ds_triton, atol=atol, rtol=rtol)
