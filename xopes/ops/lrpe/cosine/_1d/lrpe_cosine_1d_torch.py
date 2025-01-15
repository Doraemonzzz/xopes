import torch

from xopes.ops.act.act_torch import act_torch


def lrpe_cosine_1d_torch(
    x: torch.Tensor,
    theta: torch.Tensor,
    offset: int = 0,
    act: str = "none",
    dim: int = None,
) -> torch.Tensor:
    """
    Apply Lrpe Cosine 1d on the last dimension of x.

    Args:
        x: Input tensor of shape (B, N, H, D) or (B, N, D)
        theta: Tensor of shape (H, E) or (H, 1) or (1, E)
        offset: Offset for the index
        act: Activation function before apply lrpe cosine
        dim: Dimension to apply the operation on, choose from [None, -1, 1]

    Returns:
        output: Tensor of shape (B, N, H, 2 * D)
    """
    assert dim in [None, -1, 1], "dim must in [None, -1, 1]"
    has_head = len(x.shape) != 3
    if not has_head:  # b n d -> b n h d
        assert theta.shape[0] == 1, "theta must be (1, E)"
        x = x.unsqueeze(-2)
    b, n, h, d = x.shape
    h_t, h_d = theta.shape
    index = offset + torch.arange(
        n, device=torch.cuda.current_device(), dtype=torch.int64
    )
    theta = torch.einsum("h d, n -> n h d", theta.float(), index)
    cos = theta.cos()
    sin = theta.sin()

    # When h_d != d, we need to pad the theta with zeros, this makes the kernel much simpler
    if h_d != 1 and h_d != d:
        theta = F.pad(theta, (0, 0, 0, d - h_d))

    x = act_torch(x, act, dim)

    output = torch.cat([x * cos, x * sin], dim=-1)

    if not has_head:
        output = output.squeeze(-2)
    return output.to(x.dtype)


if __name__ == "__main__":
    b, n, h, d = 2, 16, 12, 64
    x = torch.randn(b, n, h, d)
    theta = torch.randn(h, d)
    o = lrpe_cosine_1d_torch(x, theta)
    print(o.shape)
