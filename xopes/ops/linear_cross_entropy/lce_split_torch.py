from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from xopes.utils import contiguous


class LinearCrossEntropySplitFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        x: torch.Tensor,  # (n d)
        y: torch.Tensor,  # (n)
        W: torch.Tensor,  # (e d)
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        n, d = x.shape
        e = W.shape[0]
        n_chunks = (e + chunk_size - 1) // chunk_size
        # n 1
        m = torch.full((n, 1), float("-inf"), dtype=x.dtype, device=x.device)
        # n 1
        sse = torch.full((n, 1), 0, dtype=torch.float32, device=x.device)
        # n 1
        s = torch.full((n, 1), 0, dtype=torch.float32, device=x.device)
        # n 1
        z_y = torch.full((n, 1), 0, dtype=x.dtype, device=x.device)

        # gradient
        # n d
        dx1 = -(1 - label_smoothing) * W[y] - label_smoothing / e * W.sum(
            dim=0, keepdim=True
        )
        # n d
        dx2 = torch.zeros((n, d), dtype=torch.float32, device=x.device)
        # e d
        dW1 = torch.zeros(e, d, device=x.device)
        dW1.scatter_add_(0, y.unsqueeze(-1).expand(-1, d), x)

        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            Wi = W[start:end]
            # n c
            logits_i = F.linear(x, Wi)
            # n 1
            mi = torch.max(logits_i, dim=-1, keepdim=True).values
            m_ = torch.maximum(m, mi)
            # n 1
            sse_i = torch.sum(torch.exp(logits_i - mi), dim=-1, keepdim=True)
            # n 1
            sse = torch.exp(m - m_) * sse + torch.exp(mi - m_) * sse_i
            m = m_
            s = s + torch.sum(logits_i, dim=-1, keepdim=True)
            # Find which target labels fall into current chunk
            mask = (y >= start) & (y < end)
            # Get relative positions of y within current chunk
            y_local = y[mask] - start
            # Update z_y for samples whose targets are in current chunk
            z_y[mask] = logits_i[mask, y_local].unsqueeze(-1)

            # gradient
            # n 1
            lse_i = mi + torch.log(sse_i)
            lse = m + torch.log(sse)
            # n c
            pi = torch.exp(logits_i - lse_i)
            # n 1
            lambda_i = torch.exp(lse_i - lse)

            # "n c, c d -> n d"
            dx2_i = torch.matmul(pi, Wi)
            dx2 = lambda_i * dx2 + (1 - lambda_i) * dx2_i

        # e d
        dW2 = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            Wi = W[start:end]
            # n c
            logits_i = F.linear(x, Wi)
            # n c
            pi = torch.exp(logits_i - lse)
            dW2_i = torch.einsum("n d, n c -> c d", x, pi)
            dW2.append(dW2_i)

        loss = -(1 - label_smoothing) * z_y + lse - label_smoothing / e * s

        # gradient
        dx = dx1 + dx2
        dW2 = torch.cat(dW2, dim=0)
        dW = dW1 + dW2

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

        ctx.save_for_backward(dx, dW)

        return loss

    @staticmethod
    @contiguous
    def backward(
        ctx, do: torch.Tensor
    ) -> Tuple[torch.Tensor, None, torch.Tensor, None, None, None, None, None]:
        dx, dW = ctx.saved_tensors
        dx = do * dx
        dW = do * dW

        return dx, None, dW, None, None, None, None, None


def linear_cross_entropy_split_torch(
    x: torch.Tensor,  # (n d)
    y: torch.Tensor,  # (n)
    W: torch.Tensor,  # (e d)
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Split version of linear cross entropy using custom autograd function.

    Args:
        x: Input tensor of shape (n, d)
        y: Target tensor of shape (n,)
        W: Weight matrix of shape (e, d)
        weight: Optional weight tensor for class weights
        ignore_index: Target value to be ignored
        reduction: Reduction method ('none', 'mean', 'sum')
        label_smoothing: Label smoothing factor
        chunk_size: Maximum chunk size for splitting the computation

    Returns:
        Loss tensor based on specified reduction
    """
    return LinearCrossEntropySplitFunction.apply(
        x, y, W, weight, ignore_index, reduction, label_smoothing, chunk_size
    )


if __name__ == "__main__":
    # Test code
    n, d, e = 2048, 512, 1024
    chunk_size = 256
    x = torch.randn((n, d), requires_grad=True).cuda()
    y = torch.randint(0, e, (n,)).cuda()
    W = torch.randn((e, d), requires_grad=True).cuda()

    loss = linear_cross_entropy_split_torch(x, y, W, chunk_size=chunk_size)
    print(f"Loss shape: {loss.shape}")

    # Test backward pass
    loss.backward()
    print(f"X grad shape: {x.grad.shape}")
    print(f"W grad shape: {W.grad.shape}")
