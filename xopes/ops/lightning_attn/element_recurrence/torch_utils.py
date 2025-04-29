from typing import Optional

import torch

from xopes.utils import contiguous


########## pytorch implementation reference ##########
@contiguous
def laer_intra_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    BLOCK_N: int = 256,
):
    def _laer_intra(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ld: torch.Tensor,
        reverse: bool = False,
        BLOCK_N: int = 256,
    ):
        b, n, h, d = k.shape
        e = v.shape[-1]
        dtype = q.dtype

        q = q.float()
        k = k.float()
        v = v.float()
        ld = ld.float()

        o = []
        l = (n + BLOCK_N - 1) // BLOCK_N
        for i in range(l):
            start = i * BLOCK_N
            end = min(start + BLOCK_N, n)
            m = end - start
            q_ = q[
                :,
                start:end,
            ]
            k_ = k[
                :,
                start:end,
            ]
            v_ = v[
                :,
                start:end,
            ]
            ld_ = ld[:, start:end]
            state = torch.zeros(b, h, d, e, dtype=torch.float32, device=q.device)
            o_array = []
            if reverse:
                array = range(m - 1, -1, -1)
            else:
                array = range(m)

            for j in array:
                if reverse:
                    if j == m - 1:
                        decay = 1  # does not affect the result
                    else:
                        decay = torch.exp(ld_[:, j + 1]).unsqueeze(-1).unsqueeze(-1)
                else:
                    decay = torch.exp(ld_[:, j]).unsqueeze(-1).unsqueeze(-1)
                state_ = k_[:, j] * v_[:, j]
                state = decay * state + state_
                o_ = q_[:, j] * state
                o_ = o_.unsqueeze(1)
                o_array.append(o_)
            o_array = torch.cat(o_array, dim=1)
            if reverse:
                o_array = torch.flip(o_array, dims=[1])
            o.append(o_array)
        o = torch.cat(o, dim=1)

        return o.to(dtype)

    o = _laer_intra(q, k, v, ld, reverse, BLOCK_N)

    return o


@contiguous
def compute_states(
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    BLOCK_N: int = 256,
    reverse: bool = False,
):
    def _compute_states(
        k: torch.Tensor,
        v: torch.Tensor,
        ld: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        BLOCK_N: int = 256,
        reverse: bool = False,
    ):
        b, n, d = k.shape

        # local state
        l = (n + BLOCK_N - 1) // BLOCK_N
        states = []
        for i in range(l):
            start = i * BLOCK_N
            end = min(start + BLOCK_N, n)
            m = end - start
            k_i = k[:, start:end]
            v_i = v[:, start:end]
            ld_i = ld[:, start:end]
            state_i = torch.zeros(b, d, dtype=torch.float32, device=k.device)
            array = range(m - 1, -1, -1) if reverse else range(m)
            local_states = []
            for j in array:
                if reverse:
                    if j == m - 1:
                        decay = 1
                    else:
                        decay = torch.exp(ld_i[:, j + 1])
                else:
                    decay = torch.exp(ld_i[:, j])
                state_ = k_i[:, j] * v_i[:, j]
                state_i = decay * state_i + state_
                local_states.append(state_i.unsqueeze(1))
            local_states = torch.cat(local_states, dim=1)
            if reverse:
                local_states = torch.flip(local_states, dims=[1])
            states.append(local_states)
        states = torch.cat(states, dim=1)

        return states

    states = _compute_states(
        k=k,
        v=v,
        ld=ld,
        initial_state=initial_state,
        BLOCK_N=BLOCK_N,
        reverse=reverse,
    )

    return states


@contiguous
def laer_inter_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    BLOCK_N: int = 256,
):
    b, n, h, d = k.shape
    e = v.shape[-1]

    m = (n + BLOCK_N - 1) // BLOCK_N
    o_intra = []
    for i in range(m):
        start = i * BLOCK_N
        end = min(start + BLOCK_N, n)
        qi = q[
            :,
            start:end,
        ]
        ki = k[
            :,
            start:end,
        ]
        vi = v[
            :,
            start:end,
        ]
        ld_i = ld[:, start:end]
        o_intra_i = laer_intra_torch(q=qi, k=ki, v=vi, ld=ld_i, BLOCK_N=BLOCK_N)
        o_intra.append(o_intra_i)
    o_intra = torch.cat(o_intra, dim=1)

    o = []
    # global state
    if initial_state is None:
        state = torch.zeros(b, h, d, e, dtype=torch.float32, device=k.device)
    else:
        state = initial_state.float()

    for i in range(n):
        decay = torch.exp(ld[:, i]).unsqueeze(-1).unsqueeze(-1)
        state_ = (
            k[
                :,
                i,
            ]
            * v[
                :,
                i,
            ]
        )
        state = decay * state + state_
        o_ = (
            q[
                :,
                i,
            ]
            * state
        )
        o_ = o_.unsqueeze(1)
        o.append(o_)

    o = torch.cat(o, dim=1)

    o_inter = o - o_intra

    return o_inter
