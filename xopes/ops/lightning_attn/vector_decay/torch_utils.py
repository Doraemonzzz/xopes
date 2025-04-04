from typing import Optional

import torch

from xopes.utils import contiguous


########## pytorch implementation reference ##########
@contiguous
def lavd_intra_torch(
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    BLOCK_N: int = 256,
):
    def _lavd_intra(
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        ldk: Optional[torch.Tensor] = None,
        ldv: Optional[torch.Tensor] = None,
        use_ldk: bool = True,
        use_ldv: bool = False,
        reverse: bool = False,
        BLOCK_N: int = 256,
    ):
        dtype = q.dtype

        q = q.float()
        if use_ldk and k is None:
            ldk = ldk.float()
            k = 1 - torch.exp(ldk)
        else:
            k = k.float()

        if use_ldv and v is None:
            ldv = ldv.float()
            v = 1 - torch.exp(ldv)
        else:
            v = v.float()

        b, n, h, d = k.shape
        e = v.shape[-1]

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

            if use_ldk:
                ldk_ = ldk[:, start:end]

            if use_ldv:
                ldv_ = ldv[:, start:end]

            state = torch.zeros(b, h, d, e, dtype=torch.float32, device=q.device)
            o_array = []
            if reverse:
                array = range(m - 1, -1, -1)
            else:
                array = range(m)

            for j in array:
                if reverse:
                    if j == m - 1:
                        dk_ = 1
                        dv_ = 1  # does not affect the result
                    else:
                        if use_ldk:
                            dk_ = torch.exp(ldk_[:, j + 1]).unsqueeze(-1)
                        else:
                            dk_ = 1

                        if use_ldv:
                            dv_ = torch.exp(ldv_[:, j + 1]).unsqueeze(-2)
                        else:
                            dv_ = 1
                else:
                    if use_ldk:
                        dk_ = torch.exp(ldk_[:, j]).unsqueeze(-1)
                    else:
                        dk_ = 1

                    if use_ldv:
                        dv_ = torch.exp(ldv_[:, j]).unsqueeze(-2)
                    else:
                        dv_ = 1

                state_ = torch.einsum(
                    "b h d, b h e -> b h d e",
                    k_[
                        :,
                        j,
                    ],
                    v_[
                        :,
                        j,
                    ],
                )
                state = dk_ * state * dv_ + state_
                o_ = torch.einsum(
                    "b h d, b h d e -> b h e",
                    q_[
                        :,
                        j,
                    ],
                    state,
                ).unsqueeze(1)
                o_array.append(o_)
            o_array = torch.cat(o_array, dim=1)
            if reverse:
                o_array = torch.flip(o_array, dims=[1])
            o.append(o_array)
        o = torch.cat(o, dim=1)

        return o.to(dtype)

    o = _lavd_intra(
        q=q,
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        reverse=reverse,
        BLOCK_N=BLOCK_N,
    )

    return o


@contiguous
def compute_states(
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    reverse: bool = False,
    BLOCK_N: int = 256,
):
    def _compute_states(
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        ldk: Optional[torch.Tensor] = None,
        ldv: Optional[torch.Tensor] = None,
        use_ldk: bool = True,
        use_ldv: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        reverse: bool = False,
        BLOCK_N: int = 256,
    ):
        if use_ldk and k is None:
            ldk = ldk.float()
            k = 1 - torch.exp(ldk)
        else:
            k = k.float()

        if use_ldv and v is None:
            ldv = ldv.float()
            v = 1 - torch.exp(ldv)
        else:
            v = v.float()

        b, n, h, d = k.shape
        e = v.shape[-1]

        # local state
        l = (n + BLOCK_N - 1) // BLOCK_N
        local_states = []
        for i in range(l):
            start = i * BLOCK_N
            end = min(start + BLOCK_N, n)
            m = end - start
            k_i = k[
                :,
                start:end,
            ]
            v_i = v[
                :,
                start:end,
            ]
            if use_ldk:
                ldk_i = ldk[:, start:end]

            if use_ldv:
                ldv_i = ldv[:, start:end]

            state_i = torch.zeros(b, h, d, e, dtype=torch.float32, device=k.device)
            array = range(m - 1, -1, -1) if reverse else range(m)
            for j in array:
                if reverse:
                    if j == m - 1:
                        dk_ = 1
                        dv_ = 1
                    else:
                        if use_ldk:
                            dk_ = torch.exp(ldk_i[:, j + 1]).unsqueeze(-1)
                        else:
                            dk_ = 1

                        if use_ldv:
                            dv_ = torch.exp(ldv_i[:, j + 1]).unsqueeze(-2)
                        else:
                            dv_ = 1
                else:
                    if use_ldk:
                        dk_ = torch.exp(ldk_i[:, j]).unsqueeze(-1)
                    else:
                        dk_ = 1

                    if use_ldv:
                        dv_ = torch.exp(ldv_i[:, j]).unsqueeze(-2)
                    else:
                        dv_ = 1
                state_ = torch.einsum(
                    "b h d, b h e -> b h d e", k_i[:, j, :, :], v_i[:, j, :, :]
                )
                state_i = dk_ * state_i * dv_ + state_
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
            if use_ldk:
                ldk = torch.flip(ldk, dims=[1])
            if use_ldv:
                ldv = torch.flip(ldv, dims=[1])

        global_states = [state.unsqueeze(2)]

        if reverse:
            c = n % BLOCK_N
        else:
            c = 0

        for i in range(n):
            if reverse:
                if i == 0:
                    dk_ = 1
                    dv_ = 1  # does not affect the result
                else:
                    if use_ldk:
                        dk_ = torch.exp(ldk[:, i - 1]).unsqueeze(-1)
                    else:
                        dk_ = 1

                    if use_ldv:
                        dv_ = torch.exp(ldv[:, i - 1]).unsqueeze(-2)
                    else:
                        dv_ = 1
            else:
                if use_ldk:
                    dk_ = torch.exp(ldk[:, i]).unsqueeze(-1)
                else:
                    dk_ = 1

                if use_ldv:
                    dv_ = torch.exp(ldv[:, i]).unsqueeze(-2)
                else:
                    dv_ = 1

            state_ = torch.einsum(
                "b h d, b h e -> b h d e", k[:, i, :, :], v[:, i, :, :]
            )
            if reverse and i == 0:
                state = state + state_
            else:
                state = dk_ * state * dv_ + state_

            # !!! important
            if reverse and i == n - 1:
                if use_ldk:
                    dk_ = torch.exp(ldk[:, i]).unsqueeze(-1)
                else:
                    dk_ = 1

                if use_ldv:
                    dv_ = torch.exp(ldv[:, i]).unsqueeze(-2)
                else:
                    dv_ = 1
                state = dk_ * state * dv_

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
        k=k,
        v=v,
        ldk=ldk,
        ldv=ldv,
        use_ldk=use_ldk,
        use_ldv=use_ldv,
        cu_seqlens=cu_seqlens,
        reverse=reverse,
        BLOCK_N=BLOCK_N,
    )

    return local_states, global_states


@contiguous
def lavd_inter_torch(
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    ldk: Optional[torch.Tensor] = None,
    ldv: Optional[torch.Tensor] = None,
    use_ldk: bool = True,
    use_ldv: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    BLOCK_N: int = 256,
):
    q = q.float()
    if use_ldk and k is None:
        ldk = ldk.float()
        k = 1 - torch.exp(ldk)
    else:
        k = k.float()

    if use_ldv and v is None:
        ldv = ldv.float()
        v = 1 - torch.exp(ldv)
    else:
        v = v.float()

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
        ldk_i = ldk[:, start:end] if use_ldk else None
        ldv_i = ldv[:, start:end] if use_ldv else None
        o_intra_i = lavd_intra_torch(
            q=qi,
            k=ki,
            v=vi,
            ldk=ldk_i,
            ldv=ldv_i,
            use_ldk=use_ldk,
            use_ldv=use_ldv,
            cu_seqlens=cu_seqlens,
            BLOCK_N=BLOCK_N,
        )
        o_intra.append(o_intra_i)
    o_intra = torch.cat(o_intra, dim=1)

    o = []
    # global state
    if initial_state is None:
        state = torch.zeros(b, h, d, e, dtype=torch.float32, device=k.device)
    else:
        state = initial_state.float()

    for i in range(n):
        qi = q[:, i]
        ki = k[:, i]
        vi = v[:, i]
        if use_ldk:
            dk_i = torch.exp(ldk[:, i]).unsqueeze(-1)
        else:
            dk_i = 1

        if use_ldv:
            dv_i = torch.exp(ldv[:, i]).unsqueeze(-2)
        else:
            dv_i = 1

        state_ = torch.einsum("b h d, b h e -> b h d e", ki, vi)

        state = dk_i * state * dv_i + state_
        oi = torch.einsum("b h d, b h d e -> b h e", qi, state).unsqueeze(1)
        o.append(oi)

    o = torch.cat(o, dim=1)

    o_inter = o - o_intra

    return o_inter
