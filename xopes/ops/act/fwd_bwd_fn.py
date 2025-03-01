import triton
import triton.language as tl


@triton.jit
def _activation_fwd(X, ACT: tl.constexpr):
    if ACT != "none":
        if ACT == "relu":
            X = tl.where(X >= 0, X, 0)
        elif ACT == "sigmoid":
            X = tl.sigmoid(X)
        elif ACT == "silu":
            X = X * tl.sigmoid(X)
        elif ACT == "softmax":
            X_max = tl.max(X, axis=-1)
            X_minus_max = X - X_max
            # softmax
            numerator = tl.exp(X_minus_max)
            denominator = tl.sum(numerator, axis=-1, keep_dims=True)
            X = numerator / denominator
    return X


@triton.jit
def _activation_bwd(X, DX, ACT: tl.constexpr):
    if ACT == "relu":
        DX = tl.where(X >= 0, DX, 0)
    elif ACT == "sigmoid":
        sigmoid = tl.sigmoid(X)
        DX = DX * sigmoid * (1 - sigmoid)
    elif ACT == "silu":
        sigmoid = tl.sigmoid(X)
        DX = DX * sigmoid * (1 + X * (1 - sigmoid))
    elif ACT == "softmax":
        X_max = tl.max(X, axis=-1)
        # for stable
        X_minus_max = X - X_max
        # softmax
        numerator = tl.exp(X_minus_max)
        denominator = tl.sum(numerator, axis=-1, keep_dims=True)
        O = numerator / denominator
        # scalar
        c = tl.sum(O * DX, axis=-1, keep_dims=True)
        DX = O * DX - c * O

    return DX
