from typing import Optional, Tuple

import torch
from einops import repeat

from xopes.utils import contiguous


def rev_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.flip(torch.cumsum(torch.flip(x, dims=[dim]), dim=dim), dims=[dim])


class LavdChunkFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, q, ldk, ldv, k=None, v=None, state=None, chunk_size=128):
        dtype = q.dtype
        q = q.float()
        ldk = ldk.float()
        ldv = ldv.float()
        if k is not None:
            k = k.float()
        if v is not None:
            v = v.float()

        b, n, h, d = q.shape
        e = ldv.shape[-1]
        c = chunk_size
        if state is not None:
            state = state.float()
        static_state = state is not None and len(state.shape) == 3
        if static_state:
            # h d e -> b h d e
            state = repeat(state, "h d e -> b h d e")

        m = (n + c - 1) // c

        o = torch.zeros((b, n, h, e), dtype=torch.float32, device=q.device)
        if state is None:
            state = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)
        state_requires_grad = state.requires_grad
        array = torch.arange(c, device=q.device, dtype=torch.int32)
        mask = torch.where(array[:, None] - array[None, :] >= 0, 1, 0)
        states = []

        for i in range(m):
            start = i * c
            end = min(start + c, n)
            l = end - start
            qi = q[:, start:end]
            ldk_i = ldk[:, start:end]
            ldv_i = ldv[:, start:end]
            if k is not None:
                ki = k[:, start:end]
            else:
                ki = 1 - torch.exp(ldk_i)
            if v is not None:
                vi = v[:, start:end]
            else:
                vi = 1 - torch.exp(ldv_i)

            states.append(state)

            # preprocess
            log_pi = torch.cumsum(ldk_i, dim=1)
            log_rho = torch.cumsum(ldv_i, dim=1)
            log_pi_ = log_pi[:, -1:, :, :]
            log_rho_ = log_rho[:, -1:, :, :]
            log_theta = log_pi_ - log_pi
            log_phi = log_rho_ - log_rho
            pi = torch.exp(log_pi)
            rho = torch.exp(log_rho)
            theta = torch.exp(log_theta)
            phi = torch.exp(log_phi)
            pi_ = torch.exp(log_pi_).squeeze(1)
            rho_ = torch.exp(log_rho_).squeeze(1)
            # update
            qi_ = qi * pi
            ki_ = ki / pi
            vi_ = vi / rho
            ki__ = ki * theta
            vi__ = vi * phi

            # intra
            energy = (
                torch.einsum("b c h d, b n h d -> b h c n", qi_, ki_) * mask[:l, :l]
            )
            oi_intra = torch.einsum("b h c n, b n h e -> b c h e", energy, vi_)
            # inter
            oi_inter = torch.einsum("b c h d, b h d e -> b c h e", qi_, state)

            oi = (oi_intra + oi_inter) * rho
            o[:, start:end] = oi

            # update
            state = pi_.unsqueeze(-1) * state * rho_.unsqueeze(-2) + torch.einsum(
                "b c h d, b c h e -> b h d e", ki__, vi__
            )

        states.append(state)
        states = torch.stack(states, dim=0)
        # Save inputs for backward pass
        ctx.save_for_backward(q, ldk, ldv, k, v, state, o, states)
        ctx.chunk_size = chunk_size
        ctx.static_state = static_state
        ctx.state_requires_grad = state_requires_grad
        ctx.dtype = dtype

        return o.to(dtype), state.to(dtype)

    @staticmethod
    @contiguous
    def backward(ctx, do, dstate):
        q, ldk, ldv, k, v, state, o, states = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        static_state = ctx.static_state
        state_requires_grad = ctx.state_requires_grad
        dtype = q.dtype
        q = q.float()
        ldk = ldk.float()
        ldv = ldv.float()
        if k is not None:
            k = k.float()
        if v is not None:
            v = v.float()

        b, n, h, d = q.shape
        e = ldv.shape[-1]
        c = chunk_size
        m = (n + c - 1) // c

        # Initialize gradient tensors
        dq = torch.empty_like(q, dtype=torch.float32)
        dldk = torch.zeros_like(ldk, dtype=torch.float32)
        dldv = torch.zeros_like(ldv, dtype=torch.float32)
        dk = torch.empty_like(dldk, dtype=torch.float32)
        dv = torch.empty_like(dldv, dtype=torch.float32)
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
            ldk_i = ldk[:, start:end]
            ldv_i = ldv[:, start:end]
            if k is not None:
                ki = k[:, start:end]
            else:
                ki = 1 - torch.exp(ldk_i)
            if v is not None:
                vi = v[:, start:end]
            else:
                vi = 1 - torch.exp(ldv_i)
            doi = do[:, start:end]
            state = states[i]

            # preprocess
            log_pi = torch.cumsum(ldk_i, dim=1)
            log_rho = torch.cumsum(ldv_i, dim=1)
            log_theta = log_pi[:, -1:, :, :] - log_pi
            log_phi = log_rho[:, -1:, :, :] - log_rho
            pi = torch.exp(log_pi)
            rho = torch.exp(log_rho)
            theta = torch.exp(log_theta)
            phi = torch.exp(log_phi)
            # update
            qi_ = qi * pi
            ki_ = ki / pi
            vi_ = vi / rho
            doi_ = doi * rho

            # intra
            energy_qk = (
                torch.einsum("b c h d, b n h d -> b h c n", doi_, vi_) * mask[:l, :l]
            )
            energy_v = (
                torch.einsum("b c h d, b n h d -> b h c n", qi_, ki_) * mask[:l, :l]
            )
            dqi_intra = torch.einsum("b h c n, b n h d -> b c h d", energy_qk, ki_) * pi
            dki_intra = torch.einsum("b h c n, b c h d -> b n h d", energy_qk, qi_) / pi
            dvi_intra = (
                torch.einsum("b h c n, b c h d -> b n h d", energy_v, doi_) / rho
            )

            # inter
            dqi_inter = torch.einsum("b c h e, b h d e -> b c h d", doi_, state) * pi

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
            ldk_i = ldk[:, start:end]
            ldv_i = ldv[:, start:end]
            if k is not None:
                ki = k[:, start:end]
            else:
                ki = 1 - torch.exp(ldk_i)
            if v is not None:
                vi = v[:, start:end]
            else:
                vi = 1 - torch.exp(ldv_i)

            dstate_ = dstates[i]
            log_pi = torch.cumsum(ldk_i, dim=1)
            log_rho = torch.cumsum(ldv_i, dim=1)
            log_pi_ = log_pi[:, -1:, :, :]
            log_rho_ = log_rho[:, -1:, :, :]
            log_theta = log_pi_ - log_pi
            log_phi = log_rho_ - log_rho
            pi = torch.exp(log_pi)
            rho = torch.exp(log_rho)
            theta = torch.exp(log_theta)
            phi = torch.exp(log_phi)
            pi_ = torch.exp(log_pi_).squeeze(1)
            rho_ = torch.exp(log_rho_).squeeze(1)
            ki__ = ki * theta
            vi__ = vi * phi

            # update
            dki_inter = (
                torch.einsum("b c h e, b h d e -> b c h d", vi__, dstate) * theta
            )
            dvi_inter = torch.einsum("b c h d, b h d e -> b c h e", ki__, dstate) * phi
            dstate = pi_.unsqueeze(-1) * dstate * rho_.unsqueeze(-2) + dstate_

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

        if k is not None:
            dldk_ = q * dq - k * dk
        else:
            dldk_ = q * dq - (1 - torch.exp(ldk)) * dk

        if v is not None:
            dldv_ = o * do - v * dv
        else:
            dldv_ = o * do - (1 - torch.exp(ldv)) * dv

        dldk = rev_cumsum(dldk_, dim=1)
        dldv = rev_cumsum(dldv_, dim=1)

        # k = 1 - exp(ldk)
        if k is None:
            dldk = dldk - torch.exp(ldk) * dk
            dk = None

        # v = 1 - exp(ldv)
        if v is None:
            dldv = dldv - torch.exp(ldv) * dv
            dv = None

        # b h d e -> b h d -> b 1 h d
        dldk += (dstate_clone * state_clone).sum(dim=-1).unsqueeze(1)
        # b h d e -> b h e -> b 1 h e
        dldv += (dstate_clone * state_clone).sum(dim=-2).unsqueeze(1)

        dq = dq.to(dtype)
        dldk = dldk.to(dtype)
        dldv = dldv.to(dtype)
        dk = dk.to(dtype) if dk is not None else None
        dv = dv.to(dtype) if dv is not None else None
        ds = ds.to(dtype) if ds is not None else None

        return dq, dldk, dldv, dk, dv, ds, None


def lavd_chunk_torch(
    q: torch.Tensor,
    ldk: torch.Tensor,
    ldv: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
    chunk_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements Lightning Attention with Vector Decay with chunking in PyTorch with autograd support.

    Args:
        q: Query tensor, shape (B, N, H, D)
        ldk: Log Decay vector for key, shape (B, N, H, D)
        ldv: Log Decay vector for value, shape (B, N, H, E)
        k: Key tensor, if not provided uses 1 - exp(ldk), shape (B, N, H, D)
        v: Value tensor, if not provided uses 1 - exp(ldv), shape (B, N, H, E)
        state: State tensor, shape (B, H, D, E) or (H, D, E)

    Returns:
        Output tensor, shape (B, N, H, E)
        State tensor, shape (B, H, D, E)
    """
    return LavdChunkFunction.apply(q, ldk, ldv, k, v, state, chunk_size)


if __name__ == "__main__":
    b, n, h, d = 2, 129, 12, 128
    e = 64
    dtype = torch.bfloat16
    q = torch.randn((b, n, h, d), dtype=dtype).cuda().requires_grad_()
    ldk = torch.randn((b, n, h, d), dtype=dtype).cuda().requires_grad_()
    ldv = torch.randn((b, n, h, e), dtype=dtype).cuda().requires_grad_()
    k = torch.randn((b, n, h, d), dtype=dtype).cuda().requires_grad_()
    v = torch.randn((b, n, h, e), dtype=dtype).cuda().requires_grad_()
    state = torch.randn((b, h, d, e), dtype=dtype).cuda().requires_grad_()
    o, state = lavd_chunk_torch(q, ldk, ldv, k, v, state)
    (o.sum() + state.sum()).backward()
    print(o.shape)
