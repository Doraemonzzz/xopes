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
        x: Input tensor of shape (B, N, H, D)
        theta: Tensor of shape (H, E) or (H, 1) or (1, E)
        offset: Offset for the index
        e: Number of elements to apply the operation on
        act: Activation function before apply lrpe cosine
        dim: Dimension to apply the operation on

    Returns:
        output: Tensor of shape (B, N, H, 2 * D)
    """
    b, n, h, d = x.shape
    index = offset + torch.arange(
        n, device=torch.cuda.current_device(), dtype=torch.int64
    )
    theta = torch.einsum("h d, n -> n h d", theta.float(), index)
    cos = theta.cos()
    sin = theta.sin()

    # When e != d, we need to pad the theta with zeros, this makes the kernel much simpler
    if e != d:
        theta = torch.cat(
            [theta[..., :e], torch.zeros(h, d - e, device=theta.device)], dim=-1
        )

    x = act_torch(x, act, dim)

    output = torch.cat([x * cos, x * sin], dim=-1)

    return output.to(x.dtype)


if __name__ == "__main__":
    b, n, h, d = 2, 16, 12, 64
    x = torch.randn(b, n, h, d)
    theta = torch.randn(h, d)
    o = lrpe_cosine_1d_torch(x, theta)
    print(o.shape)
