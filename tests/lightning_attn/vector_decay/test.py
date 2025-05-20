import pytest
import torch
import torch.nn.functional as F

from xopes.ops.lightning_attn.vector_decay import lavd_parallel_triton, lavd_torch
from xopes.utils import assert_close, print_diff


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
@pytest.mark.parametrize("use_initial_state", [True, False])
@pytest.mark.parametrize(
    "use_ldk",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "use_ldv",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "share_k",
    [True, False],
)
@pytest.mark.parametrize(
    "share_v",
    [True, False],
)
@pytest.mark.parametrize("use_varlen", [False])
@pytest.mark.parametrize("no_dstate", [True, False])
@pytest.mark.parametrize("use_chunk_loop", [True, False])
@pytest.mark.parametrize("c", [10])
@pytest.mark.parametrize("dtype", [torch.float32])
def test(
    shape,
    use_initial_state,
    use_ldk,
    use_ldv,
    share_k,
    share_v,
    use_varlen,
    no_dstate,
    use_chunk_loop,
    c,
    dtype,
):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    scale = 0.01

    b, n, h, d, e = shape

    if not use_ldk and not use_ldv:
        return

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

    if share_k:
        use_ldk = True
        ldk = F.logsigmoid(
            (1 + scale * torch.randn(b, n, h, d, dtype=dtype, device=device)) * c
        ).requires_grad_()
        k = None
    else:
        k = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()

        if use_ldk:
            ldk = F.logsigmoid(
                (1 + scale * torch.randn(b, n, h, d, dtype=dtype, device=device)) * c
            ).requires_grad_()
        else:
            ldk = None

    if share_v:
        use_ldv = True
        ldv = F.logsigmoid(
            (1 + scale * torch.randn(b, n, h, e, dtype=dtype, device=device)) * c
        ).requires_grad_()
        v = None
    else:
        v = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()

        if use_ldv:
            ldv = F.logsigmoid(
                (1 + scale * torch.randn(b, n, h, e, dtype=dtype, device=device)) * c
            ).requires_grad_()
        else:
            ldv = None

    if no_dstate:
        do = torch.randn((b, n, h, e), dtype=dtype, device=device)
    else:
        do = torch.randn((), dtype=dtype, device=device)

    if use_initial_state:
        initial_state = torch.randn(
            (b, h, d, e), dtype=dtype, device=device
        ).requires_grad_()
    else:
        initial_state = None

    ##### Forward pass
    # baseline
    o_torch, s_torch = lavd_torch(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
    )
    if no_dstate:
        output_torch = o_torch
    else:
        output_torch = o_torch.mean() + s_torch.mean()

    # triton parallel
    o_parallel_triton, s_parallel_triton = lavd_parallel_triton(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        use_chunk_loop=use_chunk_loop,
    )
    if no_dstate:
        output_parallel_triton = o_parallel_triton
    else:
        output_parallel_triton = o_parallel_triton.mean() + s_parallel_triton.mean()

    ##### Backward pass
    # baseline
    output_torch.backward(do, retain_graph=True)
    dq_torch, q.grad = q.grad.clone(), None
    if not share_k:
        dk_torch, k.grad = k.grad.clone(), None
    if not share_v:
        dv_torch, v.grad = v.grad.clone(), None
    if use_ldk:
        dldk_torch, ldk.grad = ldk.grad.clone(), None
    else:
        dldk_torch = None
    if use_ldv:
        dldv_torch, ldv.grad = ldv.grad.clone(), None
    else:
        dldv_torch = None
    if use_initial_state:
        ds_torch, initial_state.grad = initial_state.grad.clone(), None

    # triton parallel
    output_parallel_triton.backward(do, retain_graph=True)
    dq_parallel_triton, q.grad = q.grad.clone(), None
    if not share_k:
        dk_parallel_triton, k.grad = k.grad.clone(), None
    if not share_v:
        dv_parallel_triton, v.grad = v.grad.clone(), None
    if use_ldk:
        dldk_parallel_triton, ldk.grad = ldk.grad.clone(), None
    else:
        pass
    if use_ldv:
        dldv_parallel_triton, ldv.grad = ldv.grad.clone(), None
    else:
        pass
    if use_initial_state:
        ds_parallel_triton, initial_state.grad = initial_state.grad.clone(), None

    # atol, rtol = get_threshold(dtype)
    # ld_atol = 0.05
    # ld_rtol = rtol

    atol = 5e-3
    rtol = 5e-3
    ld_atol = 5e-3
    ld_rtol = 5e-3

    ##### Check forward pass results
    # triton parallel
    print(
        "o diff max (torch parallel Vs triton parallel): ",
        torch.abs(o_torch - o_parallel_triton).max().item(),
    )
    print(
        "o diff norm (torch parallel Vs triton parallel): ",
        torch.norm(o_torch - o_parallel_triton).item(),
    )
    print_diff(o_torch, o_parallel_triton, n)
    assert_close(o_torch, o_parallel_triton, atol=atol, rtol=rtol)

    print(
        "state diff max (torch parallel Vs triton parallel): ",
        torch.abs(s_torch - s_parallel_triton).max().item(),
    )
    print(
        "state diff norm (torch parallel Vs triton parallel): ",
        torch.norm(s_torch - s_parallel_triton).item(),
    )
    print(s_parallel_triton.shape)
    assert_close(s_torch, s_parallel_triton, atol=atol, rtol=rtol)

    ##### Check backward pass results
    # triton parallel
    print(
        "dq diff max (torch parallel Vs triton parallel): ",
        torch.abs(dq_torch - dq_parallel_triton).max().item(),
    )
    print(
        "dq diff norm (torch parallel Vs triton parallel): ",
        torch.norm(dq_torch - dq_parallel_triton).item(),
    )
    print_diff(dq_torch, dq_parallel_triton, n)
    assert_close(dq_torch, dq_parallel_triton, atol=atol, rtol=rtol)

    if not share_k:
        print(
            "dk diff max (torch parallel Vs triton parallel): ",
            torch.abs(dk_torch - dk_parallel_triton).max().item(),
        )
        print(
            "dk diff norm (torch parallel Vs triton parallel): ",
            torch.norm(dk_torch - dk_parallel_triton).item(),
        )
        print_diff(dk_torch, dk_parallel_triton, n)
        assert_close(dk_torch, dk_parallel_triton, atol=atol, rtol=rtol)

    if not share_v:
        print(
            "dv diff max (torch parallel Vs triton parallel): ",
            torch.abs(dv_torch - dv_parallel_triton).max().item(),
        )
        print(
            "dv diff norm (torch parallel Vs triton parallel): ",
            torch.norm(dv_torch - dv_parallel_triton).item(),
        )
        print_diff(dv_torch, dv_parallel_triton, n)
        assert_close(dv_torch, dv_parallel_triton, atol=atol, rtol=rtol)

    if use_ldk:
        print(
            "dldk diff max (torch parallel Vs triton parallel): ",
            torch.abs(dldk_torch - dldk_parallel_triton).max().item(),
        )
        print(
            "dldk diff norm (torch parallel Vs triton parallel): ",
            torch.norm(dldk_torch - dldk_parallel_triton).item(),
        )
        print_diff(dldk_torch, dldk_parallel_triton, n)
        assert_close(
            ref=dldk_torch, input=dldk_parallel_triton, atol=ld_atol, rtol=ld_rtol
        )

    if use_ldv:
        print(
            "dldv diff max (torch parallel Vs triton parallel): ",
            torch.abs(dldv_torch - dldv_parallel_triton).max().item(),
        )
        print(
            "dldv diff norm (torch parallel Vs triton parallel): ",
            torch.norm(dldv_torch - dldv_parallel_triton).item(),
        )
        print_diff(dldv_torch, dldv_parallel_triton, n)
        assert_close(
            ref=dldv_torch, input=dldv_parallel_triton, atol=ld_atol, rtol=ld_rtol
        )

    if use_initial_state:
        print(
            "ds diff max (torch parallel Vs triton parallel): ",
            torch.abs(ds_torch - ds_parallel_triton).max().item(),
        )
        print(
            "ds diff norm (torch parallel Vs triton parallel): ",
            torch.norm(ds_torch - ds_parallel_triton).item(),
        )
        print_diff(ds_torch, ds_parallel_triton, n)
        assert_close(ds_torch, ds_parallel_triton, atol=atol, rtol=rtol)
