from ..normalize import normalize_triton


def layernorm_triton(x, weight, bias, dim, eps=1e-6, residual=None):
    return normalize_triton(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        c=dim**0.5,
        eps=eps,
        use_mean=True,
        num_groups=1,
    )
