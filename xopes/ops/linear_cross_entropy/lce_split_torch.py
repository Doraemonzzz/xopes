from typing import Optional

import torch
import torch.nn.functional as F

from xopes.utils import contiguous


class LinearCrossEntropySplitFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(
        ctx,
        x,
        y,
        W,
        weight,
        ignore_index,
        reduction,
        label_smoothing,
        chunk_size,
    ):
        b, d = x.shape
        v = W.shape[0]
        if reduction == "mean":
            n = y.ne(ignore_index).sum().item()
        elif reduction == "sum":
            n = 1
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

        nv_chunks = (v + chunk_size - 1) // chunk_size
        # b 1
        m = torch.full((b, 1), float("-inf"), dtype=x.dtype, device=x.device)
        # b 1
        sse = torch.full((b, 1), 0, dtype=torch.float32, device=x.device)
        # b 1
        s = torch.full((b, 1), 0, dtype=torch.float32, device=x.device)
        # b 1
        z_y = torch.full((b, 1), 0, dtype=x.dtype, device=x.device)

        # gradient
        # b d
        dx1 = -(1 - label_smoothing) * W[y] - label_smoothing / v * W.sum(
            dim=0, keepdim=True
        )
        # b d
        dx2 = torch.zeros((b, d), dtype=torch.float32, device=x.device)

        for i in range(nv_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, v)
            Wi = W[start:end]
            # b c
            logits_i = F.linear(x, Wi)
            # b 1
            mi = torch.max(logits_i, dim=-1, keepdim=True).values
            m_ = torch.maximum(m, mi)
            # b 1
            sse_i = torch.sum(torch.exp(logits_i - mi), dim=-1, keepdim=True)
            # b 1
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
            # b 1
            lse_i = mi + torch.log(sse_i)
            lse = m + torch.log(sse)
            # b c
            pi = torch.exp((logits_i - lse_i).float())
            # b 1
            lambda_i = torch.exp(lse_i - lse)
            # "n c, c d -> n d"
            dx2_i = torch.matmul(pi, Wi.float())
            dx2 = (1 - lambda_i) * dx2 + lambda_i * dx2_i

        loss = -(1 - label_smoothing) * z_y + lse - label_smoothing / v * s

        # ##### naive version, for reference
        # # b v
        # dz1 = -(1 - label_smoothing) * F.one_hot(y, v) - label_smoothing / v
        # # v d
        # dW1 = torch.einsum("b v, b d -> v d", dz1, x.float()).to(x.dtype)

        dW1 = torch.zeros(v, d, dtype=x.dtype, device=x.device)
        dW1.scatter_add_(0, y.unsqueeze(-1).expand(-1, d), x)
        dW1 = -(1 - label_smoothing) * dW1 - label_smoothing / v * x.sum(
            dim=0, keepdim=True
        )
        dW2 = torch.zeros(v, d, dtype=torch.float32, device=x.device)
        # split over batch
        nb_chunks = (b + chunk_size - 1) // chunk_size
        for i in range(nb_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, b)
            # c d
            xi = x[start:end]
            lse_i = lse[start:end]
            # c v
            logits_i = F.linear(xi, W)
            # c v
            pi = torch.exp(logits_i - lse_i)
            dW2 += torch.einsum("c d, c v -> v d", xi, pi.to(xi.dtype))

        logits = F.linear(x, W)
        p = torch.exp(logits - lse)
        dW2 = torch.einsum("b d, b v -> v d", x, p.to(x.dtype)).to(x.dtype)

        # ##### naive version, for reference
        # # gradient
        # logits = F.linear(x, W)
        # p = torch.exp(logits - lse)
        # dx2 = torch.einsum("b v, v d -> b d", p, W.float()).to(x.dtype)

        dx = (dx1 + dx2) / n
        dW = (dW1 + dW2) / n
        loss = loss.sum() / n

        dx = dx.to(x.dtype)
        dW = dW.to(x.dtype)
        loss = loss.to(x.dtype)

        ctx.save_for_backward(dx, dW)

        return loss

    @staticmethod
    @contiguous
    def backward(ctx, do):
        dx, dW = ctx.saved_tensors
        dx = do * dx
        dW = do * dW

        return dx, None, dW, None, None, None, None, None


def linear_cross_entropy_split_torch(
    x: torch.Tensor,  # (b d)
    y: torch.Tensor,  # (b)
    W: torch.Tensor,  # (v d)
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Split version of linear cross entropy using custom autograd function.

    Args:
        x: Input tensor of shape (b, d)
        y: Target tensor of shape (b,)
        W: Weight matrix of shape (v, d)
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
    b, d, v = 2048, 512, 1024
    chunk_size = 256
    x = torch.randn((b, d), requires_grad=True).cuda()
    y = torch.randint(0, v, (b,)).cuda()
    W = torch.randn((v, d), requires_grad=True).cuda()

    loss = linear_cross_entropy_split_torch(x, y, W, chunk_size=chunk_size)
    print(f"Loss shape: {loss.shape}")

    # Test backward pass
    loss.backward()
    print(f"X grad shape: {x.grad.shape}")
    print(f"W grad shape: {W.grad.shape}")
