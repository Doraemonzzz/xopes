from .normalize_triton import normalize_triton


def rmsnorm_triton(x, weight, dim, eps=1e-6, residual=None):
    return normalize_triton(
        x=x,
        weight=weight,
        bias=None,
        residual=residual,
        c=dim**0.5,
        eps=eps,
        use_mean=False,
        num_groups=1,
    )
