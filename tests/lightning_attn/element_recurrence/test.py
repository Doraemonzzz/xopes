import pytest
import torch
import torch.nn.functional as F

from xopes.ops.lightning_attn.element_recurrence import (
    laer_parallel_triton,
    laer_recurrence_triton,
    laer_torch,
)
from xopes.utils import assert_close, get_threshold, print_diff


def get_params():
    shapes = [
        # # standard shape
        (2, 256, 128),
        (2, 1024, 32),
        # BLOCK_N +- 1
        (2, 257, 64),
        (2, 255, 64),
        (2, 65, 33),
        # BLOCK_N +- C
        (2, 270, 64),
        (2, 270, 33),
        (2, 1125, 43),
        # LARGE D
        (2, 512, 255),
        (2, 128, 257),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("use_initial_state", [True, False])
@pytest.mark.parametrize("use_varlen", [False])
@pytest.mark.parametrize("no_dstate", [True, False])
@pytest.mark.parametrize("c", [-10, 1, 10])
@pytest.mark.parametrize("dtype", [torch.float32])

# @pytest.mark.parametrize("use_initial_state", [True, False])
# @pytest.mark.parametrize("use_varlen", [False])
# @pytest.mark.parametrize("no_dstate", [True, False])
# @pytest.mark.parametrize("c", [1])
# @pytest.mark.parametrize("dtype", [torch.float32])
def test_lasr(shape, use_initial_state, use_varlen, no_dstate, c, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    b, n, d = shape
    BLOCK = 16

    if use_varlen:
        b = 1
        m = n // 5
        cu_seqlens = torch.tensor(
            [0, m - 2, 2 * m + 1, 3 * m - 1, 4 * m, n], dtype=torch.long, device=device
        )
    else:
        cu_seqlens = None

    # Generate input tensors
    q = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()
    ld = (
        F.logsigmoid(torch.randn((b, n, d), dtype=dtype, device=device) * c)
    ).requires_grad_()
    if no_dstate:
        do = torch.randn((b, n, d), dtype=dtype, device=device)
    else:
        do = torch.randn((), dtype=dtype, device=device)

    if use_initial_state:
        initial_state = torch.randn((b, d), dtype=dtype, device=device).requires_grad_()
    else:
        initial_state = None

    ##### Forward pass
    # Baseline implementation
    o_torch, s_torch = laer_torch(
        q=q,
        k=k,
        v=v,
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )
    if no_dstate:
        output_torch = o_torch
    else:
        output_torch = o_torch.mean() + s_torch.mean()

    # Triton recurrence implementation
    o_triton, s_triton = laer_recurrence_triton(
        q=q,
        k=k,
        v=v,
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )
    if no_dstate:
        output_triton = o_triton
    else:
        output_triton = o_triton.mean() + s_triton.mean()

    # Triton parallel implementation
    o_parallel, s_parallel = laer_parallel_triton(
        q=q,
        k=k,
        v=v,
        ld=ld,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )
    if no_dstate:
        output_parallel = o_parallel
    else:
        output_parallel = o_parallel.mean() + s_parallel.mean()

    ##### Backward pass
    # Baseline implementation
    output_torch.backward(do, retain_graph=True)
    dq_torch, q.grad = q.grad.clone(), None
    dk_torch, k.grad = k.grad.clone(), None
    dv_torch, v.grad = v.grad.clone(), None
    dld_torch, ld.grad = ld.grad.clone(), None
    if use_initial_state:
        ds_torch, initial_state.grad = initial_state.grad.clone(), None

    # Triton recurrence implementation
    output_triton.backward(do, retain_graph=True)
    dq_triton, q.grad = q.grad.clone(), None
    dk_triton, k.grad = k.grad.clone(), None
    dv_triton, v.grad = v.grad.clone(), None
    dld_triton, ld.grad = ld.grad.clone(), None
    if use_initial_state:
        ds_triton, initial_state.grad = initial_state.grad.clone(), None

    # Triton parallel implementation
    output_parallel.backward(do, retain_graph=True)
    dq_parallel, q.grad = q.grad.clone(), None
    dk_parallel, k.grad = k.grad.clone(), None
    dv_parallel, v.grad = v.grad.clone(), None
    dld_parallel, ld.grad = ld.grad.clone(), None
    if use_initial_state:
        ds_parallel, initial_state.grad = initial_state.grad.clone(), None

    atol, rtol = get_threshold(dtype)
    dkv_atol = 0.05
    dkv_rtol = rtol

    ##### Check forward pass results
    print(
        "o diff max (torch vs triton): ",
        torch.abs(o_torch - o_triton).max().item(),
    )
    print(
        "o diff norm (torch vs triton): ",
        torch.norm(o_torch - o_triton).item(),
    )
    assert torch.allclose(o_torch, o_triton, atol=atol, rtol=rtol)

    print(
        "state diff max (torch vs triton): ",
        torch.abs(s_torch - s_triton).max().item(),
    )
    print(
        "state diff norm (torch vs triton): ",
        torch.norm(s_torch - s_triton).item(),
    )
    assert torch.allclose(s_torch, s_triton, atol=atol, rtol=rtol)

    # parallel start
    print(
        "o diff max (torch vs parallel): ",
        torch.abs(o_torch - o_parallel).max().item(),
    )
    print(
        "o diff norm (torch vs parallel): ",
        torch.norm(o_torch - o_parallel).item(),
    )
    print_diff(o_torch, o_parallel, n, BLOCK=BLOCK)
    if c == 1:
        assert torch.allclose(o_torch, o_parallel, atol=atol, rtol=rtol)
    else:
        assert_close(ref=o_torch, input=o_parallel, atol=atol, rtol=rtol)

    print(
        "state diff max (torch vs parallel): ",
        torch.abs(s_torch - s_parallel).max().item(),
    )
    print(
        "state diff norm (torch vs parallel): ",
        torch.norm(s_torch - s_parallel).item(),
    )
    print_diff(s_torch, s_parallel, n, BLOCK=BLOCK)
    assert torch.allclose(s_torch, s_parallel, atol=atol, rtol=rtol)

    ##### Check backward pass results
    print(
        "dq diff max (torch vs triton): ",
        torch.abs(dq_torch - dq_triton).max().item(),
    )
    print("dq diff norm (torch vs triton): ", torch.norm(dq_torch - dq_triton).item())
    assert torch.allclose(dq_torch, dq_triton, atol=atol, rtol=rtol)

    print(
        "dk diff max (torch vs triton): ",
        torch.abs(dk_torch - dk_triton).max().item(),
    )
    print("dk diff norm (torch vs triton): ", torch.norm(dk_torch - dk_triton).item())
    assert torch.allclose(dk_torch, dk_triton, atol=atol, rtol=rtol)

    print(
        "dv diff max (torch vs triton): ",
        torch.abs(dv_torch - dv_triton).max().item(),
    )
    print("dv diff norm (torch vs triton): ", torch.norm(dv_torch - dv_triton).item())
    assert torch.allclose(dv_torch, dv_triton, atol=atol, rtol=rtol)

    print(
        "dld diff max (torch vs triton): ",
        torch.abs(dld_torch - dld_triton).max().item(),
    )
    print(
        "dld diff norm (torch vs triton): ", torch.norm(dld_torch - dld_triton).item()
    )
    assert torch.allclose(dld_torch, dld_triton, atol=atol, rtol=rtol)

    if use_initial_state:
        print(
            "ds diff max (torch vs triton): ",
            torch.abs(ds_torch - ds_triton).max().item(),
        )
        print(
            "ds diff norm (torch vs triton): ",
            torch.norm(ds_torch - ds_triton).item(),
        )
        assert torch.allclose(ds_torch, ds_triton, atol=atol, rtol=rtol)

    # parallel start
    print(
        "dq diff max (torch vs parallel): ",
        torch.abs(dq_torch - dq_parallel).max().item(),
    )
    print(
        "dq diff norm (torch vs parallel): ",
        torch.norm(dq_torch - dq_parallel).item(),
    )
    print_diff(dq_torch, dq_parallel, n, BLOCK=BLOCK)
    if c == 1:
        assert torch.allclose(dq_torch, dq_parallel, atol=atol, rtol=rtol * 3)
    else:
        assert_close(ref=dq_torch, input=dq_parallel, atol=atol, rtol=rtol * 3)

    print(
        "dk diff max (torch vs parallel): ",
        torch.abs(dk_torch - dk_parallel).max().item(),
    )
    print(
        "dk diff norm (torch vs parallel): ", torch.norm(dk_torch - dk_parallel).item()
    )
    print_diff(dk_torch, dk_parallel, n, BLOCK=BLOCK)
    assert_close(ref=dk_torch, input=dk_parallel, atol=dkv_atol, rtol=dkv_rtol)

    print(
        "dv diff max (torch vs parallel): ",
        torch.abs(dv_torch - dv_parallel).max().item(),
    )
    print(
        "dv diff norm (torch vs parallel): ", torch.norm(dv_torch - dv_parallel).item()
    )
    assert_close(ref=dv_torch, input=dv_parallel, atol=dkv_atol, rtol=dkv_rtol)

    print(
        "dld diff max (torch vs parallel): ",
        torch.abs(dld_torch - dld_parallel).max().item(),
    )
    print(
        "dld diff norm (torch vs parallel): ",
        torch.norm(dld_torch - dld_parallel).item(),
    )
    print_diff(dld_torch, dld_parallel, n, BLOCK=BLOCK)
    for i in range(b):
        BLOCK_D = 64
        l = (d + BLOCK_D - 1) // BLOCK_D
        for j in range(l):
            start = j * BLOCK_D
            end = min(start + BLOCK_D, d)
            print(
                i,
                j,
                torch.norm(dld_torch[i, 0, start:end]),
                torch.norm(dld_parallel[i, 0, start:end]),
            )
    assert_close(ref=dld_torch, input=dld_parallel, atol=dkv_atol, rtol=dkv_rtol)

    if use_initial_state:
        print(
            "ds diff max (torch vs parallel): ",
            torch.abs(ds_torch - ds_parallel).max().item(),
        )
        print(
            "ds diff norm (torch vs parallel): ",
            torch.norm(ds_torch - ds_parallel).item(),
        )
        assert torch.allclose(ds_torch, ds_parallel, atol=atol, rtol=rtol)
