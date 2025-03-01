from typing import Optional, Tuple

import torch
from einops import rearrange

from xopes.ops.act import act_torch


def normalize_torch(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    gate_act: str = "sigmoid",
    gate_pos: str = "pre",
    c: float = 1.0,
    eps: float = 1e-6,
    use_mean: bool = False,
    num_groups: int = 1,
    return_residual: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply normalization to the input tensor x.

    Args:
        x: Input tensor
        weight: Weight tensor
        bias: Bias tensor
        residual: Residual tensor
        gate: Gate tensor
        gate_act: Activation function for gate
        gate_pos: Position of gate
        c: Normalization constant
        eps: Epsilon value for numerical stability
        use_mean: Whether to use mean normalization
        num_groups: Number of groups to normalize across

    Returns:
        Normalized tensor, Updated residual tensor
    """
    assert (
        x.shape[-1] % num_groups == 0
    ), "The last dimension of x must be divisible by num_groups"
    dtype = x.dtype
    x = x.float()

    if weight is not None:
        weight = weight.float()
    if bias is not None:
        bias = bias.float()

    use_residual = residual is not None
    use_gate = gate is not None
    assert not (
        use_residual and use_gate
    ), "gate and residual cannot be used at the same time"

    if use_residual:
        residual = residual.float()
        x = x + residual
        # update residual
        residual = x

    if use_gate:
        gate = act_torch(gate, gate_act)
        if gate_pos == "pre":
            x = x * gate

    x_ = rearrange(x, "... (g e) -> ... g e", g=num_groups)

    if use_mean:
        x_ = x_ - x_.mean(dim=-1, keepdim=True)

    sigma = torch.sqrt(torch.sum(x_ * x_, dim=-1, keepdim=True) + eps)
    o = c * x_ / sigma

    if weight is not None:
        weight = rearrange(weight, "... (g e) -> ... g e", g=num_groups)
        o = o * weight
    if bias is not None:
        bias = rearrange(bias, "... (g e) -> ... g e", g=num_groups)
        o = o + bias

    o = o.reshape_as(x).to(dtype)
    # if residual is not None, update it; if residual is None and return_residual is True, set residual to x
    if residual is not None:
        residual = residual.to(dtype)
    else:
        if return_residual:
            residual = x.to(dtype)

    if use_gate and gate_pos == "post":
        o = o * gate

    if use_residual or return_residual:
        return o, residual
    else:
        return o
