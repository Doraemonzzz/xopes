import triton
import triton.language as tl


@triton.jit
def _normalization_fwd(X, USE_NORM: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
    if USE_NORM:
        sigma = tl.sqrt(tl.sum(X * X, axis=-1) / D + EPS)
        O = (1 / D**0.5) * X / sigma
    else:
        O = X

    return O


@triton.jit
def _normalization_bwd(
    X, DX, USE_NORM: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr
):
    if USE_NORM:
        sigma = tl.sqrt(tl.sum(X * X, axis=-1) / D + EPS)
        R = X / sigma
        DR = DX * (1 / D**0.5)
        DX = 1 / sigma * (DR - R * tl.sum(R * DR, axis=-1) / D)

    return DX
