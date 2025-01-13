import torch

from xopes.ops.act.act_torch import act_torch


def lrpe_cosine_1d_torch(
    x: torch.Tensor,
    theta: torch.Tensor,
    offset: int = 0,
    start_dim: int = 0,
    act: str = "none",
    dim: int = None,
) -> torch.Tensor:
    """
    Apply Lrpe Cosine 1d on the last dimension of x.

    Args:
        x: Input tensor of shape (B, N, H, D)
        theta: Tensor of shape (H, D) or (H) or (D)
        offset: Offset for the index
        start_dim: Start dimension to apply the operation on
        act: Activation function before apply lrpe cosine
        dim: Dimension to apply the operation on

    Returns:
        output: Tensor of shape (B, N, H, start_dim + 2 * (D - start_dim))

    Examples:
        [:start_dim], [start_dim:d] -> [:start_dim], [start_dim:d] * cos, [start_dim:d] * sin
    """
    b, n, h, d = x.shape
    index = offset + torch.arange(
        n, device=torch.cuda.current_device(), dtype=torch.int64
    )
    if len(theta.shape) == 1:
        if theta.shape[0] == h:  # h -> h 1
            theta = theta.unsqueeze(-1)
        elif theta.shape[0] == d:  # d -> 1 d
            theta = theta.unsqueeze(0)
    theta = torch.einsum("h d, n -> h n d", theta.float(), index)
    cos = theta.cos()
    sin = theta.sin()

    x = act_torch(x, act, dim)

    output_identity = x[..., :start_dim]
    output_lrpe = torch.cat(
        [x[..., start_dim:] * cos, x[..., start_dim:] * sin], dim=-1
    )
    output = torch.cat([output_identity, output_lrpe], dim=start_dim)

    return output.to(x.dtype)
