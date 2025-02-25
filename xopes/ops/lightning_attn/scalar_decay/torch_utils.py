from typing import Optional

import torch


########## pytorch implementation reference ##########
def lasd_intra_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
):
    def _lasd_intra(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ld: Optional[torch.Tensor] = None,
    ):
        b, n, h, d = k.shape
        e = v.shape[-1]
        dtype = q.dtype

        q = q.float()
        k = k.float()
        v = v.float()

        if ld is None:
            decay = (
                torch.ones((), dtype=torch.float32, device=q.device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
        else:
            decay = torch.exp(ld.float()).unsqueeze(-1).unsqueeze(-1)

        state = torch.zeros(b, h, d, e, dtype=torch.float32, device=q.device)
        o = []
        for i in range(n):
            state_ = torch.einsum(
                "b h d, b h e -> b h d e", k[:, i, :, :], v[:, i, :, :]
            )
            state = decay * state + state_
            o_ = torch.einsum(
                "b h d, b h d e -> b h e", q[:, i, :, :], state
            ).unsqueeze(1)
            o.append(o_)
        o = torch.cat(o, dim=1)

        return o.to(dtype)

    if reverse:
        q = torch.flip(q, dims=[1])
        k = torch.flip(k, dims=[1])
        v = torch.flip(v, dims=[1])

    o = _lasd_intra(q, k, v, ld)

    if reverse:
        o = torch.flip(o, dims=[1])

    return o


def compute_states(
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    BLOCK_N: int = 256,
    reverse: bool = False,
):
    def _compute_states(
        k: torch.Tensor,
        v: torch.Tensor,
        ld: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        BLOCK_N: int = 256,
        reverse: bool = False,
    ):
        b, n, h, d = k.shape
        e = v.shape[-1]

        if ld is None:
            decay = (
                torch.ones((), dtype=torch.float32, device=k.device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
        else:
            decay = torch.exp(ld.float()).unsqueeze(-1).unsqueeze(-1)

        # local state
        l = (n + BLOCK_N - 1) // BLOCK_N
        local_states = []
        for i in range(l):
            start = i * BLOCK_N
            end = min(start + BLOCK_N, n)
            m = end - start
            k_i = k[:, start:end, :, :]
            v_i = v[:, start:end, :, :]
            state_i = torch.zeros(b, h, d, e, dtype=torch.float32, device=k.device)
            array = range(m - 1, -1, -1) if reverse else range(m)
            for j in array:
                state_ = torch.einsum(
                    "b h d, b h e -> b h d e", k_i[:, j, :, :], v_i[:, j, :, :]
                )
                state_i = decay * state_i + state_
            local_states.append(state_i.unsqueeze(2))
        local_states = torch.cat(local_states, dim=2)

        # global state
        if initial_state is None:
            state = torch.zeros(b, h, d, e, dtype=torch.float32, device=k.device)
        else:
            state = initial_state.float()

        if reverse:
            k = torch.flip(k, dims=[1])
            v = torch.flip(v, dims=[1])

        global_states = [state.unsqueeze(2)]

        if reverse:
            c = n % BLOCK_N
        else:
            c = 0

        for i in range(n):
            state_ = torch.einsum(
                "b h d, b h e -> b h d e", k[:, i, :, :], v[:, i, :, :]
            )
            if reverse and i == 0:
                state = state + state_
            else:
                state = decay * state + state_

            # !!! important
            if reverse and i == n - 1:
                state = decay * state

            if (i + 1 - c) % BLOCK_N == 0 or (i == n - 1):
                global_states.append(state.unsqueeze(2))

        global_states = torch.cat(global_states, dim=2)

        if reverse:
            global_states = torch.cat(
                [
                    torch.flip(global_states[:, :, :-1], dims=[2]),
                    global_states[:, :, -1:],
                ],
                dim=2,
            )

        return local_states, global_states

    local_states, global_states = _compute_states(
        k, v, ld, initial_state, BLOCK_N, reverse
    )

    return local_states, global_states


def lasd_inter_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ld: Optional[torch.Tensor] = None,
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
        qi = q[:, start:end, :, :]
        ki = k[:, start:end, :, :]
        vi = v[:, start:end, :, :]
        o_intra_i = lasd_intra_torch(qi, ki, vi, ld)
        o_intra.append(o_intra_i)
    o_intra = torch.cat(o_intra, dim=1)

    if ld is None:
        decay = (
            torch.ones((), dtype=torch.float32, device=k.device)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
    else:
        decay = torch.exp(ld.float()).unsqueeze(-1).unsqueeze(-1)

    o = []
    # global state
    if initial_state is None:
        state = torch.zeros(b, h, d, e, dtype=torch.float32, device=k.device)
    else:
        state = initial_state.float()

    for i in range(n):
        state_ = torch.einsum("b h d, b h e -> b h d e", k[:, i, :, :], v[:, i, :, :])
        state = decay * state + state_
        o_ = torch.einsum("b h d, b h d e -> b h e", q[:, i, :, :], state).unsqueeze(1)
        o.append(o_)

    o = torch.cat(o, dim=1)

    o_inter = o - o_intra

    return o_inter
