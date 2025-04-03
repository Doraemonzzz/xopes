from typing import Optional, Tuple

import torch
from einops import repeat

from xopes.utils import contiguous


def rev_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.flip(torch.cumsum(torch.flip(x, dims=[dim]), dim=dim), dims=[dim])


class LavdChunkFunction(torch.autograd.Function):
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
        dtype = q.dtype
        compute_dldk = use_ldk and ldk is not None
        compute_dldv = use_ldv and ldv is not None
        q = q.float()
        k = k.float()
        v = v.float()
        if use_ldk and ldk is None:
            ldk = torch.log(1 - k)
        if use_ldv and ldv is None:
            ldv = torch.log(1 - v)

        b, n, h, d = q.shape
        e = v.shape[-1]
        c = chunk_size
        if initial_state is None:
            state = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)
        else:
            state = initial_state.float()
        static_state = initial_state is not None and len(initial_state.shape) == 3
        if static_state:
            # h d e -> b h d e
            state = repeat(state, "h d e -> b h d e")

        m = (n + c - 1) // c

        o = torch.zeros((b, n, h, e), dtype=torch.float32, device=q.device)
        state_requires_grad = state.requires_grad
        array = torch.arange(c, device=q.device, dtype=torch.int32)
        mask = torch.where(array[:, None] - array[None, :] >= 0, 1, 0)
        states = []

        for i in range(m):
            start = i * c
            end = min(start + c, n)
            l = end - start
            qi = q[:, start:end]
            ki = k[:, start:end]
            vi = v[:, start:end]
            ldk_i = ldk[:, start:end] if use_ldk else None
            ldv_i = ldv[:, start:end] if use_ldv else None

            states.append(state)

            # preprocess
            if use_ldk:
                log_pi = torch.cumsum(ldk_i, dim=1)
                log_pi_ = log_pi[:, -1:, :, :]
                log_theta = log_pi_ - log_pi
                pi = torch.exp(log_pi)
                theta = torch.exp(log_theta)
                pi_ = torch.exp(log_pi_).squeeze(1)
                # update
                qi_ = qi * pi
                ki_ = ki / pi
                ki__ = ki * theta
            else:
                qi_ = qi
                ki_ = ki
                ki__ = ki

            if use_ldv:
                log_rho = torch.cumsum(ldv_i, dim=1)
                log_rho_ = log_rho[:, -1:, :, :]
                log_phi = log_rho_ - log_rho
                rho = torch.exp(log_rho)
                phi = torch.exp(log_phi)
                rho_ = torch.exp(log_rho_).squeeze(1)
                # update
                vi_ = vi / rho
                vi__ = vi * phi
            else:
                vi_ = vi
                vi__ = vi

            # intra
            energy = (
                torch.einsum("b c h d, b n h d -> b h c n", qi_, ki_) * mask[:l, :l]
            )
            oi_intra = torch.einsum("b h c n, b n h e -> b c h e", energy, vi_)
            # inter
            oi_inter = torch.einsum("b c h d, b h d e -> b c h e", qi_, state)

            oi = oi_intra + oi_inter
            if use_ldv:
                oi = oi * rho
            o[:, start:end] = oi

            # update
            state_ = torch.einsum("b c h d, b c h e -> b h d e", ki__, vi__)

            if use_ldk:
                state = pi_.unsqueeze(-1) * state
            if use_ldv:
                state = state * rho_.unsqueeze(-2)

            state = state + state_

        states.append(state)
        states = torch.stack(states, dim=0)
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

        return o.to(dtype), state.to(dtype)

    @staticmethod
    @contiguous
    def backward(ctx, do, dstate):
        q, ldk, ldv, k, v, state, o, states = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        static_state = ctx.static_state
        state_requires_grad = ctx.state_requires_grad
        use_ldk = ctx.use_ldk
        use_ldv = ctx.use_ldv
        compute_dldk = ctx.compute_dldk
        compute_dldv = ctx.compute_dldv

        dtype = q.dtype
        q = q.float()
        k = k.float()
        v = v.float()
        if use_ldk and ldk is None:
            ldk = torch.log(1 - k)
        if use_ldv and ldv is None:
            ldv = torch.log(1 - v)

        b, n, h, d = q.shape
        e = v.shape[-1]
        c = chunk_size
        m = (n + c - 1) // c

        # Initialize gradient tensors
        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.empty_like(k, dtype=torch.float32)
        dv = torch.empty_like(v, dtype=torch.float32)
        dldk = torch.zeros_like(k, dtype=torch.float32)
        dldv = torch.zeros_like(v, dtype=torch.float32)
        if dstate is None:
            dstate = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)
            dstate_clone = torch.zeros_like(dstate)
            state_clone = torch.zeros_like(state)
        else:
            # for compute dldk
            dstate_clone = dstate.clone()
            state_clone = state.clone()

        array = torch.arange(c, device=q.device, dtype=torch.int32)
        mask = torch.where(array[:, None] - array[None, :] >= 0, 1, 0)
        dstates = []

        # first pass, compute dk, dv intra and dq
        for i in range(m):
            start = i * c
            end = min(start + c, n)
            l = end - start
            qi = q[:, start:end]
            ki = k[:, start:end]
            vi = v[:, start:end]
            ldk_i = ldk[:, start:end] if use_ldk else None
            ldv_i = ldv[:, start:end] if use_ldv else None
            doi = do[:, start:end]
            state = states[i]

            # preprocess
            if use_ldk:
                log_pi = torch.cumsum(ldk_i, dim=1)
                log_pi_ = log_pi[:, -1:, :, :]
                log_theta = log_pi_ - log_pi
                pi = torch.exp(log_pi)
                theta = torch.exp(log_theta)
                # update
                qi_ = qi * pi
                ki_ = ki / pi
            else:
                qi_ = qi
                ki_ = ki

            if use_ldv:
                log_rho = torch.cumsum(ldv_i, dim=1)
                log_rho_ = log_rho[:, -1:, :, :]
                log_phi = log_rho_ - log_rho
                rho = torch.exp(log_rho)
                phi = torch.exp(log_phi)
                # update
                vi_ = vi / rho
                doi_ = doi * rho
            else:
                vi_ = vi
                doi_ = doi

            # intra
            energy_qk = (
                torch.einsum("b c h d, b n h d -> b h c n", doi_, vi_) * mask[:l, :l]
            )
            energy_v = (
                torch.einsum("b c h d, b n h d -> b h c n", qi_, ki_) * mask[:l, :l]
            )
            dqi_intra = torch.einsum("b h c n, b n h d -> b c h d", energy_qk, ki_)
            if use_ldk:
                dqi_intra = dqi_intra * pi
            dki_intra = torch.einsum("b h c n, b c h d -> b n h d", energy_qk, qi_)
            if use_ldk:
                dki_intra = dki_intra / pi
            dvi_intra = torch.einsum("b h c n, b c h d -> b n h d", energy_v, doi_)
            if use_ldv:
                dvi_intra = dvi_intra / rho

            # inter
            dqi_inter = torch.einsum("b c h e, b h d e -> b c h d", doi_, state)
            if use_ldk:
                dqi_inter = dqi_inter * pi

            # local dstate
            dstate_ = torch.einsum("b c h d, b c h e -> b h d e", qi_, doi_)
            dstates.append(dstate_)

            # save
            dq[:, start:end] = dqi_intra + dqi_inter
            dk[:, start:end] = dki_intra
            dv[:, start:end] = dvi_intra

        # second pass, reduce and compute intra
        for i in range(m - 1, -1, -1):
            start = i * c
            end = min(start + c, n)
            l = end - start
            qi = q[:, start:end]
            ki = k[:, start:end]
            vi = v[:, start:end]
            ldk_i = ldk[:, start:end] if use_ldk else None
            ldv_i = ldv[:, start:end] if use_ldv else None
            doi = do[:, start:end]

            dstate_ = dstates[i]
            if use_ldk:
                log_pi = torch.cumsum(ldk_i, dim=1)
                log_pi_ = log_pi[:, -1:, :, :]
                log_theta = log_pi_ - log_pi
                pi = torch.exp(log_pi)
                theta = torch.exp(log_theta)
                pi_ = torch.exp(log_pi_).squeeze(1)
                ki__ = ki * theta
            else:
                ki__ = ki
            if use_ldv:
                log_rho = torch.cumsum(ldv_i, dim=1)
                log_rho_ = log_rho[:, -1:, :, :]
                log_phi = log_rho_ - log_rho
                rho = torch.exp(log_rho)
                phi = torch.exp(log_phi)
                rho_ = torch.exp(log_rho_).squeeze(1)
                vi__ = vi * phi
            else:
                vi__ = vi

            # update
            dki_inter = torch.einsum("b c h e, b h d e -> b c h d", vi__, dstate)
            if use_ldk:
                dki_inter = dki_inter * theta
            dvi_inter = torch.einsum("b c h d, b h d e -> b c h e", ki__, dstate)
            if use_ldv:
                dvi_inter = dvi_inter * phi

            if use_ldk:
                dstate = pi_.unsqueeze(-1) * dstate

            if use_ldv:
                dstate = dstate * rho_.unsqueeze(-2)

            dstate = dstate + dstate_

            # save
            dk[:, start:end] = dk[:, start:end] + dki_inter
            dv[:, start:end] = dv[:, start:end] + dvi_inter

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
        dldk = rev_cumsum(dldk_, dim=1)

        dldv_ = o * do - v * dv
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


def lavd_chunk_torch(
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
    Implements Lightning Attention with Vector Decay with chunking in PyTorch with autograd support.

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

    return LavdChunkFunction.apply(
        q, k, v, ldk, ldv, use_ldk, use_ldv, initial_state, chunk_size
    )


if __name__ == "__main__":
    b, n, h, d = 2, 8, 12, 128
    e = 64
    dtype = torch.bfloat16
    device = "cuda"

    q = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
    ldk = torch.randn((b, n, h, d), dtype=dtype, device=device).requires_grad_()
    ldv = torch.randn((b, n, h, e), dtype=dtype, device=device).requires_grad_()
    initial_state = torch.randn(
        (b, h, d, e), dtype=dtype, device=device
    ).requires_grad_()

    o, state = lavd_chunk_torch(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        initial_state=initial_state,
    )
    (o.sum() + state.sum()).backward()
    print(o.shape)
