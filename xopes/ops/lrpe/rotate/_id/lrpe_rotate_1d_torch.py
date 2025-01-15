import torch
import torch.nn.functional as F

from xopes.ops.act.act_torch import act_torch


def lrpe_rotate_1d_torch(
    x: torch.Tensor,
    theta: torch.Tensor,
    offset: int = 0,
    act: str = "none",
    dim: int = None,
) -> torch.Tensor:
    """
    Apply Lrpe Rotate (i.e. RoPE) 1d on the last dimension of x.

    Args:
        x: Input tensor of shape (B, N, H, D) or (B, N, D)
        theta: Tensor of shape (H, E) or (1, E), E <= D / 2
        offset: Offset for the index
        act: Activation function before apply lrpe cosine
        dim: Dimension to apply the operation on, choose from [None, -1, 1]

    Returns:
        output: Tensor of shape (B, N, H, D)
    """
    dtype = x.dtype
    assert dim in [None, -1, 1], "dim must in [None, -1, 1]"
    has_head = len(x.shape) != 3
    if not has_head:  # b n d -> b n h d
        assert theta.shape[0] == 1, "theta must be (1, E)"
        x = x.unsqueeze(-2)
    b, n, h, d = x.shape
    h_t, h_d = theta.shape
    index = offset + torch.arange(n, device=x.device, dtype=torch.int64)
    theta = torch.einsum("h d, n -> n h d", theta.float(), index)

    # When h_d != d // 2, we need to pad the theta with zeros, this makes the kernel much simpler
    if h_d != 1 and h_d != d // 2:
        theta = F.pad(theta, (0, 0, 0, d // 2 - h_d))

    x = act_torch(x, act, dim)

    theta = torch.polar(torch.ones_like(theta), theta)
    x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    output = torch.view_as_real(x * theta).flatten(3)

    if not has_head:
        output = output.squeeze(-2)
    return output.to(dtype)


if __name__ == "__main__":
    b, n, h, d = 2, 16, 12, 64
    x = torch.randn(b, n, h, d)
    theta = torch.randn(h, d // 2)
    o = lrpe_rotate_1d_torch(x, theta)
    print(o.shape)
