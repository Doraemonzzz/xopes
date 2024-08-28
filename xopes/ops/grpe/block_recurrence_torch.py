import torch
import torch.nn.functional as F
from einops import rearrange


def pad(x, BLOCK_SIZE=64, dim=-2):
    if dim != -2:
        x = x.transpose(dim, -2)
    n = x.shape[-2]
    final_len = (n + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    if final_len != n:
        x = F.pad(x, (0, 0, 0, final_len - n))

    if dim != -2:
        x = x.transpose(dim, -2)

    return x


def cumsum_and_revcumsum(x, dim=-1):
    if dim != -1:
        x = x.transpose(-1, dim)
    x.shape[-1]
    x_cumsum = x.cumsum(dim=-1)
    x_rev_cumsum = x_cumsum[..., -1, None] - x_cumsum + x

    if dim != -1:
        x_cumsum = x_cumsum.transpose(-1, dim)
        x_rev_cumsum = x_rev_cumsum.transpose(-1, dim)

    return x_cumsum.contiguous(), x_rev_cumsum.contiguous()


def grpe_block_recurrence_torch(
    q,
    k,
    v,
    alpha,
    beta,
    gamma,
    initial_state=None,
    output_final_state=False,
    BLOCK_SIZE=64,
):
    b, h, n, d = q.shape
    e = v.shape[-1]

    q, k, v, alpha, gamma = map(lambda x: pad(x, BLOCK_SIZE), [q, k, v, alpha, gamma])
    beta = pad(beta, BLOCK_SIZE, dim=-1)
    n_pad = q.shape[-2]
    q, k, v, alpha, gamma = map(
        lambda x: rearrange(x, "b h (n B) d -> b h n B d", B=BLOCK_SIZE),
        [q, k, v, alpha, gamma],
    )
    beta = rearrange(beta, "b h (n B) -> b h n B", B=BLOCK_SIZE)

    if initial_state is not None:
        s = initial_state
    else:
        s = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)

    # compute cumsum
    alpha_cumsum, alpha_revcumsum = cumsum_and_revcumsum(alpha, dim=-2)
    # b h l B d d

    gamma_out_product = gamma.unsqueeze(-1) * gamma.unsqueeze(-2)
    gamma_out_product = beta.unsqueeze(-1).unsqueeze(-1) * gamma_out_product
    gamma_out_product_cumsum, gamma_out_product_revcumsum = cumsum_and_revcumsum(
        gamma_out_product, dim=-3
    )
    # compute matrix
    identity = torch.eye(d, device=torch.cuda.current_device())
    log_m_cumsum = alpha_cumsum.unsqueeze(-1) * identity + gamma_out_product_cumsum

    l = n_pad // BLOCK_SIZE
    o = []

    # compute intra in parallel
    s_intra = torch.zeros(b, h, l, d, e).to(torch.float32).to(q.device)
    identity = torch.eye(d, device=torch.cuda.current_device())
    o_intra = []

    for i in range(BLOCK_SIZE):
        # b h l d
        qi = q[:, :, :, i : i + 1]
        ki = k[:, :, :, i : i + 1]
        vi = v[:, :, :, i : i + 1]
        alpha_i = alpha[:, :, :, i]
        beta_i = beta[:, :, :, i]
        gamma_i = gamma[:, :, :, i]
        # b h l d d
        gamma_i_out_product = (
            beta_i.unsqueeze(-1).unsqueeze(-1)
            * gamma_i.unsqueeze(-1)
            * gamma_i.unsqueeze(-2)
        )
        log_mi = alpha_i.unsqueeze(-1) * identity + gamma_i_out_product
        mi = torch.matrix_exp(log_mi)

        s_intra = torch.matmul(mi, s_intra) + ki.transpose(-1, -2) * vi

        oi_intra = torch.matmul(qi, s_intra.to(qi.dtype))
        o_intra.append(oi_intra)

    o_intra = torch.cat(o_intra, dim=-2)

    # compute inter
    o_inter = []

    # s: b h d e
    s_inter_list = []
    for i in range(l):
        s_inter_list.append(s)

        s = torch.matrix_exp(log_m_cumsum[:, :, i, -1]) @ s + s_intra[:, :, i]

    # b h l d d
    s_inter = torch.stack(s_inter_list, dim=-3)
    # b h l B d d
    m_cumprod = torch.matrix_exp(log_m_cumsum.contiguous())
    # b h l B d
    q_cumprod = torch.einsum("... d, ... d e-> ... e", q, m_cumprod.to(q.dtype))
    o_inter = torch.einsum(
        "... B d, ... d e -> ... B e", q_cumprod, s_inter.to(q.dtype)
    )

    o = o_intra + o_inter
    o = rearrange(o, "b h n B d -> b h (n B) d")[:, :, :n, :]

    final_state = None
    if output_final_state:
        final_state = s

    return o, final_state


if __name__ == "__main__":
    import torch.nn.functional as F

    b, h, n, d, e = 2, 8, 128, 128, 32
    dtype = torch.float32
    device = torch.cuda.current_device()
    q = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    k = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    v = (torch.randn((b, h, n, e), dtype=dtype, device=device)).requires_grad_()
    lower_bound = 0.95
    alpha = torch.log(
        lower_bound
        + (1 - lower_bound)
        * F.sigmoid(torch.randn((b, h, n, d), dtype=dtype, device=device))
    ).requires_grad_()
    beta = torch.log(
        lower_bound
        + (1 - lower_bound)
        * F.sigmoid(torch.randn((b, h, n), dtype=dtype, device=device))
    ).requires_grad_()
    gamma = F.normalize(
        torch.randn((b, h, n, d), dtype=dtype, device=device)
    ).requires_grad_()
    initial_state = None
    output_final_state = False

    (
        o_block_recurrence_torch,
        final_state_block_recurrence_torch,
    ) = grpe_block_recurrence_torch(
        q, k, v, alpha, beta, gamma, initial_state, output_final_state
    )
