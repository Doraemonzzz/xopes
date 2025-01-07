import torch


def oplr_no_decay_torch(
    xk: torch.Tensor,  # b n d
    xv: torch.Tensor,  # b n e
) -> torch.Tensor:
    """
    Applies Out Product Linear Recurrence without decay.

    Args:
        xk: Expansion vector
        xv: Input tensor

    Returns:
        Output tensor
    """
    b, n, d = xk.shape
    xv.shape[-1]

    xkv = torch.einsum("b n d, b n e -> b n d e", xk, xv)
    o = torch.cumsum(xkv, dim=1)

    return o
