from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from xopes.utils import contiguous


def rev_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.flip(torch.cumsum(torch.flip(x, dims=[dim]), dim=dim), dims=[dim])


class LavdChunkParallelFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        ldk=None,
        ldv=None,
        use_ldk=True,
        use_ldv=False,
        initial_state=None,
        chunk_size=128,
    ):
        compute_dldk = use_ldk and ldk is not None
        compute_dldv = use_ldv and ldv is not None
        dtype = q.dtype
        b, n, h, d = q.shape
        e = v.shape[-1]
        c = chunk_size

        q = q.float()
        k = k.float()
        v = v.float()
        if use_ldk and ldk is None:
            ldk = torch.log(1 - k)
        if use_ldv and ldv is None:
            ldv = torch.log(1 - v)

        if initial_state is None:
            state = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)
        else:
            state = initial_state.float()

        static_state = initial_state is not None and len(initial_state.shape) == 3

        if static_state:
            # h d e -> b h d e
            state = repeat(state, "h d e -> b h d e")

        # reshape
        l = (c - n % c) % c
        m = (n + c - 1) // c
        n = q.shape[1]
        q = F.pad(q, (0, 0, 0, 0, 0, l))
        ldk = F.pad(ldk, (0, 0, 0, 0, 0, l))
        ldv = F.pad(ldv, (0, 0, 0, 0, 0, l))
        q, ldk, ldv = map(
            lambda x: rearrange(x, "b (m c) h d -> b m c h d", c=c), (q, ldk, ldv)
        )
        if k is not None:
            k = F.pad(k, (0, 0, 0, 0, 0, l))
            k = rearrange(k, "b (m c) h d -> b m c h d", m=m)
        if v is not None:
            v = F.pad(v, (0, 0, 0, 0, 0, l))
            v = rearrange(v, "b (m c) h d -> b m c h d", m=m)
        if use_ldk:
            log_pi = torch.cumsum(ldk, dim=2)
        if use_ldv:
            log_rho = torch.cumsum(ldv, dim=2)

        # prepare
        o = torch.zeros((b, m, c, h, e), dtype=torch.float32, device=q.device)
        state_requires_grad = state.requires_grad
        # b 1 h d e
        states = [state.unsqueeze(1)]
        # !!! important, the initial state is 0 for intra
        state = torch.cat(
            [torch.zeros((b, m, h, d, e), dtype=torch.float32, device=q.device)], dim=1
        )
        array = torch.arange(c, device=q.device, dtype=torch.int32)
        mask = torch.where(array[:, None] - array[None, :] >= 0, 1, 0)

        ##### compute intra
        for i in range(c):
            qi = q[:, :, i]
            ki = k[:, :, i]
            vi = v[:, :, i]
            ldk_i = ldk[:, :, i] if use_ldk else None
            ldv_i = ldv[:, :, i] if use_ldv else None

            if use_ldk:
                # b m h d -> b m h d 1
                lambda_i = torch.exp(ldk_i).unsqueeze(-1)
                state = lambda_i * state
            if use_ldv:
                # b m h e -> b m h 1 e
                gamma_i = torch.exp(ldv_i).unsqueeze(-2)
                state = state * gamma_i
            # b m h d e
            state = state + ki.unsqueeze(-1) * vi.unsqueeze(-2)
            # b m h d e
            oi_intra = torch.einsum("... h d, ... h d e -> ... h e", qi, state)
            o[:, :, i] = oi_intra

        states.append(state)
        # b (m + 1) h d e
        states = torch.concat(states, dim=1)
        state = states[:, 0]

        ##### update state
        for i in range(m):
            qi = q[:, i]
            ki = k[:, i]
            vi = v[:, i]
            # b c h d
            ldk_i = ldk[:, i] if use_ldk else None
            # b c h e
            ldv_i = ldv[:, i] if use_ldv else None

            state_ = states[:, i + 1]

            # preprocess
            if use_ldk:
                log_pi_ = log_pi[:, i, -1]
                pi_ = torch.exp(log_pi_).unsqueeze(-1)
                state = pi_ * state

            if use_ldv:
                log_rho_ = log_rho[:, i, -1]
                rho_ = torch.exp(log_rho_).unsqueeze(-2)
                state = state * rho_

            # update
            state = state + state_
            states[:, i + 1] = state

        ##### compute inter
        if use_ldk:
            pi = torch.exp(log_pi)
            # update
            q_ = q * pi
        else:
            q_ = q

        if use_ldv:
            rho = torch.exp(log_rho)

        o_inter = torch.einsum("b m c h d, b m h d e -> b m c h e", q_, states[:, :m])
        if use_ldv:
            o_inter = o_inter * rho
        o += o_inter

        # Save inputs for backward pass
        ctx.save_for_backward(q, ldk, ldv, k, v, state, o, states)
        ctx.chunk_size = chunk_size
        ctx.static_state = static_state
        ctx.state_requires_grad = state_requires_grad
        ctx.use_ldk = use_ldk
        ctx.use_ldv = use_ldv
        ctx.compute_dldk = compute_dldk
        ctx.compute_dldv = compute_dldv
        ctx.dtype = dtype
        ctx.n = n

        o = rearrange(o, "b m c h e -> b (m c) h e")[:, :n]

        if not use_ldk:
            log_pi = None
        else:
            log_pi = rearrange(log_pi, "b m c h d -> b (m c) h d")[:, :n]

        if not use_ldv:
            log_rho = None
        else:
            log_rho = rearrange(log_rho, "b m c h e -> b (m c) h e")[:, :n]

        return o.to(dtype), state.to(dtype)  # , states, log_pi, log_rho

    @staticmethod
    @contiguous
    def backward(ctx, do, dstate):
        q, ldk, ldv, k, v, state, o, states = ctx.saved_tensors
        ctx.chunk_size
        static_state = ctx.static_state
        state_requires_grad = ctx.state_requires_grad
        use_ldk = ctx.use_ldk
        use_ldv = ctx.use_ldv
        compute_dldk = ctx.compute_dldk
        compute_dldv = ctx.compute_dldv

        dtype = q.dtype
        n = ctx.n
        k is None
        v is None

        q = q.float()
        k = k.float()
        v = v.float()
        if use_ldk and ldk is None:
            ldk = torch.log(1 - k)
        if use_ldv and ldv is None:
            ldv = torch.log(1 - v)

        b, m, c, h, d = q.shape
        e = v.shape[-1]
        n = do.shape[1]
        l = (c - n % c) % c
        do = F.pad(do, (0, 0, 0, 0, 0, l))
        do = rearrange(do, "b (m c) h d -> b m c h d", m=m)

        # Initialize gradient tensors
        dq = torch.empty_like(q, dtype=torch.float32)
        dldk = torch.zeros_like(k, dtype=torch.float32)
        dldv = torch.zeros_like(v, dtype=torch.float32)
        dk = torch.empty_like(k, dtype=torch.float32)
        dv = torch.empty_like(v, dtype=torch.float32)
        if dstate is None:
            dstate = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)
            dstate_clone = torch.zeros_like(dstate)
            state_clone = torch.zeros_like(state)
        else:
            # for compute dldk
            dstate_clone = dstate.clone()
            state_clone = state.clone()

        if use_ldk:
            log_pi = torch.cumsum(ldk, dim=2)
        if use_ldv:
            log_rho = torch.cumsum(ldv, dim=2)

        array = torch.arange(c, device=q.device, dtype=torch.int32)
        mask = torch.where(array[:, None] - array[None, :] >= 0, 1, 0)
        dstates = [dstate.unsqueeze(1)]

        ##### compute intra
        # !!! important, the initial state is 0 for intra
        state = torch.cat(
            [torch.zeros((b, m, h, d, e), dtype=torch.float32, device=q.device)], dim=1
        )
        for i in range(c):
            qi = q[:, :, i]
            ki = k[:, :, i]
            vi = v[:, :, i]
            doi = do[:, :, i]
            ldk_i = ldk[:, :, i] if use_ldk else None
            ldv_i = ldv[:, :, i] if use_ldv else None

            if use_ldk:
                # b m h d -> b m h d 1
                lambda_i = torch.exp(ldk_i).unsqueeze(-1)
                state = lambda_i * state
            if use_ldv:
                # b m h e -> b m h 1 e
                gamma_i = torch.exp(ldv_i).unsqueeze(-2)
                state = state * gamma_i
            # b m h d e
            state = state + ki.unsqueeze(-1) * vi.unsqueeze(-2)

            # compute
            dqi_intra = torch.einsum("... h e, ... h d e -> ... h d", doi, state)
            dq[:, :, i] = dqi_intra

        # !!! important, the initial dstate is 0 for intra
        dstate = torch.cat(
            [torch.zeros((b, m, h, d, e), dtype=torch.float32, device=q.device)]
        )
        for i in range(c - 1, -1, -1):
            qi = q[:, :, i]
            ki = k[:, :, i]
            vi = v[:, :, i]
            doi = do[:, :, i]
            if use_ldk:
                ldk_i = ldk[:, :, i + 1] if i != c - 1 else torch.zeros_like(ldk_i)
            if use_ldv:
                ldv_i = ldv[:, :, i + 1] if i != c - 1 else torch.zeros_like(ldv_i)

            if use_ldk:
                # b m h d -> b m h d 1
                lambda_i = torch.exp(ldk_i).unsqueeze(-1)
                dstate = lambda_i * dstate

            if use_ldv:
                # b m h e -> b m h 1 e
                gamma_i = torch.exp(ldv_i).unsqueeze(-2)
                dstate = dstate * gamma_i

            dstate = dstate + qi.unsqueeze(-1) * doi.unsqueeze(-2)

            # compute
            dki_intra = torch.einsum("... h e, ... h d e -> ... h d", vi, dstate)
            dvi_intra = torch.einsum("... h d, ... h d e -> ... h e", ki, dstate)
            dk[:, :, i] = dki_intra
            dv[:, :, i] = dvi_intra

            # !!! important
            if i == 0:
                if use_ldk:
                    ldk_i = ldk[:, :, i]
                    # b m h d -> b m h d 1
                    lambda_i = torch.exp(ldk_i).unsqueeze(-1)
                    dstate = lambda_i * dstate

                if use_ldv:
                    ldv_i = ldv[:, :, i]
                    # b m h e -> b m h 1 e
                    gamma_i = torch.exp(ldv_i).unsqueeze(-2)
                    dstate = dstate * gamma_i

        dstates.insert(0, dstate)
        # b (m + 1) h d e
        dstates = torch.concat(dstates, dim=1)

        # update dstate
        dstate = dstates[:, -1]
        for i in range(m - 1, -1, -1):
            if use_ldk:
                # b c h d
                ldk_i = ldk[:, i]
            if use_ldv:
                # b c h e
                ldv_i = ldv[:, i]

            dstate_ = dstates[:, i]

            # preprocess
            if use_ldk:
                log_pi_ = log_pi[:, i, -1]
                pi_ = torch.exp(log_pi_).unsqueeze(-1)
                dstate = pi_ * dstate
            if use_ldv:
                log_rho_ = log_rho[:, i, -1]
                rho_ = torch.exp(log_rho_).unsqueeze(-2)
                dstate = dstate * rho_

            # b m h d e
            dstate = dstate + dstate_
            dstates[:, i] = dstate

        ##### compute inter
        if use_ldk:
            log_theta = log_pi[:, :, -1:] - log_pi
            pi = torch.exp(log_pi)
            theta = torch.exp(log_theta)

        if use_ldv:
            log_phi = log_rho[:, :, -1:] - log_rho
            rho = torch.exp(log_rho)
            phi = torch.exp(log_phi)

        if use_ldk:
            # update
            k_ = k * theta
        else:
            k_ = k

        if use_ldv:
            v_ = v * phi
            do_ = do * rho
        else:
            v_ = v
            do_ = do

        dq_inter = torch.einsum("b m c h e, b m h d e -> b m c h d", do_, states[:, :m])
        if use_ldk:
            dq_inter = dq_inter * pi
        dq += dq_inter

        dk_inter = torch.einsum("b m c h e, b m h d e -> b m c h d", v_, dstates[:, 1:])
        if use_ldk:
            dk_inter = dk_inter * theta
        dk += dk_inter

        dv_inter = torch.einsum("b m c h d, b m h d e -> b m c h e", k_, dstates[:, 1:])
        if use_ldv:
            dv_inter = dv_inter * phi
        dv += dv_inter

        if state is not None and state_requires_grad:
            # The following is the correct formula, but it is not used because it is slower
            # log_gamma = torch.cumsum(ldk, dim=1)
            # log_delta = torch.cumsum(ldv, dim=1)
            # ds_ = torch.einsum(
            #     "b n h d, b n h e -> b h d e", q * torch.exp(log_gamma), do * torch.exp(log_delta)
            # ) + log_gamma[:, -1].unsqueeze(-1) * dstate * log_delta[:, -1].unsqueeze(-2)
            ds = dstate
            if static_state:
                ds = ds.sum(dim=0)
            ds = ds.to(dtype)
        else:
            ds = None

        dldk_ = q * dq - k * dk
        dldv_ = o * do - v * dv

        # reshape
        dq = rearrange(dq, "b m c h d -> b (m c) h d")[:, :n]
        dldk_ = rearrange(dldk_, "b m c h d -> b (m c) h d")[:, :n]
        dldv_ = rearrange(dldv_, "b m c h d -> b (m c) h d")[:, :n]
        dk = (
            rearrange(dk, "b m c h d -> b (m c) h d")[:, :n] if dk is not None else None
        )
        dv = (
            rearrange(dv, "b m c h d -> b (m c) h d")[:, :n] if dv is not None else None
        )
        ldk = rearrange(ldk, "b m c h d -> b (m c) h d")[:, :n]
        ldv = rearrange(ldv, "b m c h d -> b (m c) h d")[:, :n]
        k = rearrange(k, "b m c h d -> b (m c) h d")[:, :n]
        v = rearrange(v, "b m c h d -> b (m c) h d")[:, :n]

        dldk = rev_cumsum(dldk_, dim=1)
        dldv = rev_cumsum(dldv_, dim=1)

        # b h d e -> b h d -> b 1 h d
        dldk += (dstate_clone * state_clone).sum(dim=-1).unsqueeze(1)
        # b h d e -> b h e -> b 1 h e
        dldv += (dstate_clone * state_clone).sum(dim=-2).unsqueeze(1)

        # ldk = log(1 - k)
        if not compute_dldk:
            dk = dk - dldk / (1 - k)
            dldk = None

        # ldv = log(1 - v)
        if not compute_dldv:
            dv = dv - dldv / (1 - v)
            dldv = None

        dq = dq.to(dtype)
        dk = dk.to(dtype)
        dv = dv.to(dtype)
        dldk = dldk.to(dtype) if compute_dldk else None
        dldv = dldv.to(dtype) if compute_dldv else None
        ds = ds.to(dtype) if ds is not None else None

        return dq, dk, dv, dldk, dldv, None, None, ds, None


def lavd_chunk_parallel_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    chunk_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements Lightning Attention with Vector Decay with chunking parallel in PyTorch with autograd support.

    Args:
        q: Query tensor, shape (B, N, H, D)
        k: Key tensor, shape (B, N, H, D)
        v: Value tensor, shape (B, N, H, E)
        ldk: Log Decay vector for key, shape (B, N, H, D), if not provided uses log(1 - exp(k))
        ldv: Log Decay vector for value, shape (B, N, H, E), if not provided uses log(1 - exp(v))
        use_ldk: Whether to use log decay for key
        use_ldv: Whether to use log decay for value
        initial_state: Initial state tensor, shape (B, H, D, E) or (H, D, E)

    Returns:
        Output tensor, shape (B, N, H, E)
        State tensor, shape (B, H, D, E)
    """
    if ldk is not None:
        use_ldk = True
    if ldv is not None:
        use_ldv = True

    assert use_ldk or use_ldv, "At least one of ldk or ldv must be used"

    return LavdChunkParallelFunction.apply(
        q, k, v, ldk, ldv, use_ldk, use_ldv, initial_state, chunk_size
    )


if __name__ == "__main__":
    b, n, h, d = 2, 129, 12, 128
    e = 64
    dtype = torch.float32
    device = "cuda"

    q = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
    ldk = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    ldv = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
    initial_state = torch.randn(
        (b, h, d, e), dtype=dtype, device=device
    ).requires_grad_()

    o, state = lavd_chunk_parallel_torch(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        initial_state=initial_state,
    )
    (o.sum() + state.sum()).backward()
    print(o.shape)
