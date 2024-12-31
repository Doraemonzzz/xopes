from .normalize_triton import normalize_triton


def srmsnorm_triton(x, dim, eps=1e-6, residual=None):
    return normalize_triton(
        x=x,
        weight=None,
        bias=None,
        residual=residual,
        c=dim**0.5,
        eps=eps,
        use_mean=False,
        num_groups=1,
    )
