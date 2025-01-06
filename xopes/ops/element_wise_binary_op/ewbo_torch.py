from xopes.utils import is_dim_valid, is_op_valid


def ewbo_torch(x: torch.Tensor, y: torch.Tensor, op="add") -> torch.Tensor:
    """
    Element-wise binary operation.

    Args:
        x: Input tensor, (n1, ... , nk, n(k+1), ... , n(k+m), m >= 0)
        y: Input tensor, (n1, ... , nk)
        op: Binary operation to apply

    Returns:
        Result of the binary operation
    """
    is_op_valid(op)
    x_shape = x.shape
    y_shape = y.shape
    is_dim_valid(x_shape, y_shape)

    n = len(x_shape) - len(y_shape)
    for i in range(n):
        y = y.unsqueeze(-1)

    if op == "add":
        o = x + y
    elif op == "mul":
        o = x * y
    elif op == "sub":
        o = x - y
    elif op == "div":
        o = x / y

    return o
