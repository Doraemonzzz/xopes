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


if __name__ == "__main__":
    b, n, d, e = 2, 512, 128, 128
    dtype = torch.bfloat16
    xv = torch.randn((b, n, e), dtype=dtype).cuda()
    xk = torch.randn((b, n, d), dtype=dtype).cuda()
    o = oplr_no_decay_torch(xk, xv)
    print(o.shape)
