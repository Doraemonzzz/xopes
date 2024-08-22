import torch


def grpe_recurrence_torch(
    q, k, v, alpha, beta, gamma, initial_state=None, output_final_state=False
):
    b, h, n, d = q.shape
    e = v.shape[-1]

    # m = exp(alpha + beta * gamma * gamma ^ T)
    identity = torch.eye(d, device=torch.cuda.current_device())
    order_one_term = alpha.unsqueeze(-1) * identity
    order_two_term = (
        beta.unsqueeze(-1).unsqueeze(-1) * gamma.unsqueeze(-1) * gamma.unsqueeze(-2)
    )
    log_m = order_one_term + order_two_term
    m = torch.matrix_exp(log_m)

    if initial_state is not None:
        s = initial_state
    else:
        s = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)

    o = []
    for i in range(n):
        qi = q[:, :, i : i + 1].contiguous().to(torch.float32)
        ki = k[:, :, i : i + 1].contiguous().to(torch.float32)
        vi = v[:, :, i : i + 1].contiguous().to(torch.float32)
        mi = m[:, :, i].contiguous().to(torch.float32)

        s = torch.matmul(mi, s) + ki.transpose(-1, -2) * vi

        oi = torch.matmul(qi, s)

        o.append(oi)

    o = torch.cat(o, dim=-2).to(q.dtype)

    final_state = None
    if output_final_state:
        final_state = s

    return o, final_state


if __name__ == "__main__":
    import torch.nn.functional as F

    b, h, n, d, e = 2, 8, 128, 64, 32
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
    gamma = (torch.randn((b, h, n, d), dtype=dtype, device=device)).requires_grad_()
    initial_state = None
    output_final_state = False

    o_recurrence_torch, final_state_recurrence_torch = grpe_recurrence_torch(
        q, k, v, alpha, beta, gamma, initial_state, output_final_state
    )
