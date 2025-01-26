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
    def forward(ctx, q, ldk, ldv, k=None, v=None, state=None, chunk_size=128):
        dtype = q.dtype
        b, n, h, d = q.shape
        e = ldv.shape[-1]
        c = chunk_size

        q = q.float()
        ldk = ldk.float()
        ldv = ldv.float()
        if k is not None:
            k = k.float()
        if v is not None:
            v = v.float()

        if state is not None:
            state = state.float()

        static_state = state is not None and len(state.shape) == 3

        if static_state:
            # h d e -> b h d e
            state = repeat(state, "h d e -> b h d e")

        if state is None:
            state = torch.zeros((b, h, d, e), dtype=torch.float32, device=q.device)

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
        log_pi = torch.cumsum(ldk, dim=2)
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
            ldk_i = ldk[:, :, i]
            ldv_i = ldv[:, :, i]
            if k is not None:
                ki = k[:, :, i]
            else:
                ki = 1 - torch.exp(ldk_i)
            if v is not None:
                vi = v[:, :, i]
            else:
                vi = 1 - torch.exp(ldv_i)

            # b m h d -> b m h d 1
            lambda_i = torch.exp(ldk_i).unsqueeze(-1)
            # b m h e -> b m h 1 e
            gamma_i = torch.exp(ldv_i).unsqueeze(-2)
            # b m h d e
            state = lambda_i * state * gamma_i + ki.unsqueeze(-1) * vi.unsqueeze(-2)
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
            # b c h d
            ldk_i = ldk[
                :,
                i,
            ]
            # b c h e
            ldv_i = ldv[
                :,
                i,
            ]
            if k is not None:
                ki = k[:, i]
            else:
                ki = 1 - torch.exp(ldk_i)
            if v is not None:
                vi = v[:, i]
            else:
                vi = 1 - torch.exp(ldv_i)

            state_ = states[:, i + 1]

            # preprocess
            log_pi_ = log_pi[:, i, -1]
            log_rho_ = log_rho[:, i, -1]
            pi_ = torch.exp(log_pi_).unsqueeze(-1)
            rho_ = torch.exp(log_rho_).unsqueeze(-2)

            # update
            state = pi_ * state * rho_ + state_
            states[:, i + 1] = state

        ##### compute inter
        pi = torch.exp(log_pi)
        rho = torch.exp(log_rho)
        # update
        q_ = q * pi
        o_inter = (
            torch.einsum("b m c h d, b m h d e -> b m c h e", q_, states[:, :m]) * rho
        )
        o += o_inter

        # Save inputs for backward pass
        ctx.save_for_backward(q, ldk, ldv, k, v, state, o, states)
        ctx.chunk_size = chunk_size
        ctx.static_state = static_state
        ctx.state_requires_grad = state_requires_grad
        ctx.dtype = dtype
        ctx.n = n

        o = rearrange(o, "b m c h e -> b (m c) h e")[:, :n]

        return o.to(dtype), state.to(dtype)

    @staticmethod
    @contiguous
    def backward(ctx, do, dstate):
        q, ldk, ldv, k, v, state, o, states = ctx.saved_tensors
        ctx.chunk_size
        static_state = ctx.static_state
        state_requires_grad = ctx.state_requires_grad
        dtype = q.dtype
        n = ctx.n
        k_is_none = k is None
        v_is_none = v is None

        q = q.float()
        ldk = ldk.float()
        ldv = ldv.float()
        if k is not None:
            k = k.float()
        else:
            k = 1 - torch.exp(ldk)
        if v is not None:
            v = v.float()
        else:
            v = 1 - torch.exp(ldv)

        b, m, c, h, d = q.shape
        e = ldv.shape[-1]
        n = do.shape[1]
        l = (c - n % c) % c
        do = F.pad(do, (0, 0, 0, 0, 0, l))
        do = rearrange(do, "b (m c) h d -> b m c h d", m=m)

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
        log_pi = torch.cumsum(ldk, dim=2)
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
            ldk_i = ldk[:, :, i]
            ldv_i = ldv[:, :, i]
            ki = k[:, :, i]
            vi = v[:, :, i]
            doi = do[:, :, i]

            # b m h d -> b m h d 1
            lambda_i = torch.exp(ldk_i).unsqueeze(-1)
            # b m h e -> b m h 1 e
            gamma_i = torch.exp(ldv_i).unsqueeze(-2)
            # b m h d e
            state = lambda_i * state * gamma_i + ki.unsqueeze(-1) * vi.unsqueeze(-2)

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
            ldk_i = ldk[:, :, i + 1] if i != c - 1 else torch.zeros_like(ldk_i)
            ldv_i = ldv[:, :, i + 1] if i != c - 1 else torch.zeros_like(ldv_i)

            # b m h d -> b m h d 1
            lambda_i = torch.exp(ldk_i).unsqueeze(-1)
            # b m h e -> b m h 1 e
            gamma_i = torch.exp(ldv_i).unsqueeze(-2)
            dstate = lambda_i * dstate * gamma_i + qi.unsqueeze(-1) * doi.unsqueeze(-2)

            # compute
            dki_intra = torch.einsum("... h e, ... h d e -> ... h d", vi, dstate)
            dvi_intra = torch.einsum("... h d, ... h d e -> ... h e", ki, dstate)
            dk[:, :, i] = dki_intra
            dv[:, :, i] = dvi_intra

            # !!! important
            if i == 0:
                ldk_i = ldk[:, :, i]
                ldv_i = ldv[:, :, i]
                # b m h d -> b m h d 1
                lambda_i = torch.exp(ldk_i).unsqueeze(-1)
                # b m h e -> b m h 1 e
                gamma_i = torch.exp(ldv_i).unsqueeze(-2)
                dstate = lambda_i * dstate * gamma_i

        dstates.insert(0, dstate)
        # b (m + 1) h d e
        dstates = torch.concat(dstates, dim=1)

        # update dstate
        dstate = dstates[:, -1]
        for i in range(m - 1, -1, -1):
            # b c h d
            ldk_i = ldk[:, i]
            # b c h e
            ldv_i = ldv[:, i]

            dstate_ = dstates[:, i]

            # preprocess
            log_pi_ = log_pi[:, i, -1]
            log_rho_ = log_rho[:, i, -1]
            pi_ = torch.exp(log_pi_).unsqueeze(-1)
            rho_ = torch.exp(log_rho_).unsqueeze(-2)

            # b m h d e
            dstate = pi_ * dstate * rho_ + dstate_
            dstates[:, i] = dstate

        ##### compute inter
        log_theta = log_pi[:, :, -1:] - log_pi
        log_phi = (
            log_rho[
                :,
                :,
                -1:,
            ]
            - log_rho
        )
        pi = torch.exp(log_pi)
        rho = torch.exp(log_rho)
        theta = torch.exp(log_theta)
        phi = torch.exp(log_phi)
        # update
        k_ = k * theta
        v_ = v * phi
        do_ = do * rho
        dq += torch.einsum("b m c h e, b m h d e -> b m c h d", do_, states[:, :m]) * pi
        dk += (
            torch.einsum("b m c h e, b m h d e -> b m c h d", v_, dstates[:, 1:])
            * theta
        )
        dv += (
            torch.einsum("b m c h d, b m h d e -> b m c h e", k_, dstates[:, 1:]) * phi
        )

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

        dldk = rev_cumsum(dldk_, dim=1)
        dldv = rev_cumsum(dldv_, dim=1)

        # k = 1 - exp(ldk)
        if k_is_none:
            dldk = dldk - torch.exp(ldk) * dk
            dk = None

        # v = 1 - exp(ldv)
        if v_is_none:
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


def lavd_chunk_parallel_torch(
    q: torch.Tensor,
    ldk: torch.Tensor,
    ldv: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
    chunk_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements Lightning Attention with Vector Decay with chunking parallel in PyTorch with autograd support.

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
    return LavdChunkParallelFunction.apply(q, ldk, ldv, k, v, state, chunk_size)


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
    o, state = lavd_chunk_parallel_torch(q, ldk, ldv, k, v, state)
    (o.sum() + state.sum()).backward()
    print(o.shape)
