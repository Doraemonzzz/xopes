try:
    from fla.modules.activations import swiglu_linear
except ImportError:
    swiglu_linear = None


def fla_gate_linear_wrapper(x1, x2, weight, bias=None, residual=None, act="none"):
    o = swiglu_linear(x1, x2, weight, bias)
    return o
