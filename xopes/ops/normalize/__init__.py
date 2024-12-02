from .srmsnorm import srmsnorm_triton


def srmsnorm_fn(x, eps=1e-8, scale=False):
    return srmsnorm_triton(x, eps, scale)
