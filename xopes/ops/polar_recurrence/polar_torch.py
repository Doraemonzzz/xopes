from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def polar_torch(
    q: torch.Tensor,
    alpha: torch.Tensor,
    r: torch.Tensor,
    norm_style: str = 'none',
    beta: Optional[torch.Tensor] = None,
    s: Optional[torch.Tensor] = None,
    gamma: Optional[torch.Tensor] = None,
    log_decay: Optional[torch.Tensor] = None,
    u_state: Optional[torch.Tensor] = None,
    p_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies Polar Recurrence (Sequential Recurrence) in PyTorch.

    Args:
        q: Query tensor (B, N, H, D)
        alpha: Alpha tensor (B, N, H, D)
        r: R tensor (B, N, H, D)
        norm_style: Normalization style, choose from ['none', 'alpha', 'r', 'alpha_r'], 
                    if 'none', no normalization is applied;
                    if 'alpha', alpha and beta are normalized;
                    if 'r', r and s are normalized;
                    if 'alpha_r', alpha, beta, r and s are normalized;
        beta: Beta tensor (B, N, H, D), if not provided, alpha is used
        s: S tensor (B, N, H, E), if not provided, r is used, and E = D
        gamma: Gamma scaling tensor (B, N, H, D) or (B, N, H)
        log_decay: Log decay tensor (B, N, H, D) or (B, N, H)
        u_state: Initial U state tensor (B, H, D, E), optional
        p_state: Initial P state tensor (B, H, D, E), optional

    Returns:
        Output tensor (B, N, H, E)
        Final U state tensor (B, H, D, D)
        Final P state tensor (B, H, D, E)
    """
    dtype = q.dtype
    device = q.device
    b, n, h, d = q.shape
    e = s.shape[-1] if s is not None else d

    # Convert to float32 for better numerical stability
    q = q.float()
    alpha = alpha.float()
    beta = beta.float()
    r = r.float()
    s = s.float()

    # Initialize states if not provided
    if u_state is None:
        u_state = torch.eye(d, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)
    if p_state is None:
        p_state = torch.zeros((b, h, d, e), dtype=torch.float32, device=device)
    
    # Initialize output tensor
    o = torch.zeros((b, n, h, e), dtype=torch.float32, device=device)

    # Optional tensors handling
    if gamma is not None:
        gamma = gamma.float()
    else:
        gamma = torch.ones((b, n, h, d), dtype=torch.float32, device=device)
        
    if len(gamma.shape) == 3:
        gamma = gamma.unsqueeze(-1)
    
    if log_decay is not None:
        log_decay = log_decay.float()
    else:
        log_decay = torch.zeros((b, n, h, d), dtype=torch.float32, device=device)
        
    if len(log_decay.shape) == 3:
        log_decay = log_decay.unsqueeze(-1)
        
    # normalize vectors
    if norm_style == 'alpha':
        alpha = F.normalize(alpha, dim=-1)
        if beta is not None:
            beta = F.normalize(beta, dim=-1)
        else:
            beta = alpha
    elif norm_style == 'r':
        r = F.normalize(r, dim=-1)
        if s is not None:
            s = F.normalize(s, dim=-1)
        else:
            s = r
    elif norm_style == 'alpha_r':
        alpha = F.normalize(alpha, dim=-1)
        r = F.normalize(r, dim=-1)
        if beta is not None:
            beta = F.normalize(beta, dim=-1)
        else:
            beta = alpha
        if s is not None:
            s = F.normalize(s, dim=-1)
        else:
            s = r

    for i in range(n):
        # Get current step tensors
        alpha_i = alpha[:, i] * gamma[:, i] # b h d
        beta_i = beta[:, i] # b h d
        r_i = r[:, i] # b h d
        s_i = s[:, i] # b h e
        decay_i = torch.exp(log_decay[:, i]) # b h d
        q_i = q[:, i] # b h d

        # Update u state
        eta_i = torch.einsum("... d e, ... d -> ... e", u_state, beta_i) # b h d
        u_state += torch.einsum("... d, ... e -> ... d e", alpha_i, eta_i) # b h d e

        # Update p state
        p_state = decay_i.unsqueeze(-1) * p_state + torch.einsum("... d, ... e -> ... d e", r_i, s_i)

        # Compute output
        h_i = torch.einsum("... d e, ... d -> ... e", u_state, q_i) # b h d
        o_i = torch.einsum("... d e, ... d -> ... e", p_state, h_i) # b h e
        o[:, i] = o_i

    return o.to(dtype), u_state.to(dtype), p_state.to(dtype)


if __name__ == "__main__":
    # Test code
    b, h, n = 2, 8, 128
    d = 128
    e = 64
    dtype = torch.float32
    
    q = torch.randn((b, n, h, d), dtype=dtype).cuda()
    alpha = torch.randn((b, n, h, d), dtype=dtype).cuda()
    beta = torch.randn((b, n, h, d), dtype=dtype).cuda()
    r = torch.randn((b, n, h, d), dtype=dtype).cuda()
    s = torch.randn((b, n, h, e), dtype=dtype).cuda()
    gamma = torch.randn((b, n, h, d), dtype=dtype).cuda()
    log_decay = torch.randn((b, n, h, d), dtype=dtype).cuda()
    u_state = torch.randn((b, h, d, d), dtype=dtype).cuda()
    p_state = torch.randn((b, h, d, e), dtype=dtype).cuda()
    
    o, u_state, p_state = polar_torch(
        q=q, alpha=alpha, beta=beta, r=r, s=s,
        norm_style='alpha_r',
        gamma=gamma,
        log_decay=log_decay,
        u_state=u_state,
        p_state=p_state
    )
    print(f"Output shape: {o.shape}")
    print(f"U state shape: {u_state.shape}")
    print(f"P state shape: {p_state.shape}")