from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from xopes.utils import contiguous
from xopes.ops.act import act_torch

class PolarAgTorchFunction(torch.autograd.Function):
    @staticmethod
    @contiguous
    def forward(ctx, q, alpha, r, norm_style="none", beta=None, s=None, 
                gamma=None, log_decay=None, u_state=None, p_state=None):
        dtype = q.dtype
        device = q.device
        b, n, h, d = q.shape
        e = s.shape[-1] if s is not None else d

        # Initialize states
        u_init_state = u_state
        p_init_state = p_state
        if u_state is None:
            u_state = (torch.eye(d, dtype=torch.float32, device=device)
                      .unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1))
        u_state = u_state.float()
        
        if p_state is None:
            p_state = torch.zeros((b, h, d, e), dtype=torch.float32, device=device)
        p_state = p_state.float()

        # Optional tensors
        if gamma is None:
            gamma = torch.ones((b, n, h, d), dtype=torch.float32, device=device)
        else:
            gamma = gamma.float()
        if len(gamma.shape) == 3:
            gamma = gamma.unsqueeze(-1)

        if log_decay is None:
            log_decay = torch.zeros((b, n, h, d), dtype=torch.float32, device=device)
        else:
            log_decay = log_decay.float()
        if len(log_decay.shape) == 3:
            log_decay = log_decay.unsqueeze(-1)

        # Initialize output
        o = torch.zeros((b, n, h, e), dtype=torch.float32, device=device)
        eta = torch.zeros((b, n, h, d), dtype=torch.float32, device=device)
        h = torch.zeros((b, n, h, d), dtype=torch.float32, device=device)

        for i in range(n):
            # Unitary update
            alpha_i = alpha[:, i] * gamma[:, i]
            beta_i = beta[:, i]
            q_i = q[:, i]
            eta_i = torch.einsum("... d e, ... d -> ... e", u_state, beta_i)
            u_state = u_state + torch.einsum("... d, ... e -> ... d e", alpha_i, eta_i)
            h_i = torch.einsum("... d e, ... d -> ... e", u_state, q_i)
            
            # Spectral update
            r_i = r[:, i]
            s_i = s[:, i]
            decay_i = torch.exp(log_decay[:, i])
            p_state = decay_i.unsqueeze(-1) * p_state + torch.einsum(
                "... d, ... e -> ... d e", r_i, s_i
            )
            o_i = torch.einsum("... d e, ... d -> ... e", p_state, h_i)
            o[:, i] = o_i

        # Save for backward
        ctx.save_for_backward(q, alpha, r, beta, s, gamma, log_decay, u_init_state, p_init_state, o)
        ctx.norm_style = norm_style
        ctx.dtype = dtype
        
        return o.to(dtype), u_state.to(dtype), p_state.to(dtype)

    @staticmethod
    @contiguous
    def backward(ctx, do, du_state, dp_state):
        q, alpha, r, beta, s, gamma, log_decay, u_state, p_state, o = ctx.saved_tensors
        norm_style = ctx.norm_style
        dtype = ctx.dtype
        b, n, h, d = q.shape
        e = s.shape[-1] if s is not None else d
        
        # Initialize gradients
        dq = torch.zeros_like(q, dtype=torch.float32)
        dalpha = torch.zeros_like(alpha, dtype=torch.float32)
        dr = torch.zeros_like(r, dtype=torch.float32)
        dbeta = torch.zeros_like(beta, dtype=torch.float32)
        ds = torch.zeros_like(s, dtype=torch.float32)
        dgamma = torch.zeros_like(gamma, dtype=torch.float32)
        dlog_decay = torch.zeros_like(log_decay, dtype=torch.float32)

        # Initialize states for backward pass
        if du_state is None:
            du_state = torch.zeros((b, h, d, d), dtype=torch.float32, device=device)
        du = du_state.float()
            
        if dp_state is None:
            dp_state = torch.zeros((b, h, d, e), dtype=torch.float32, device=device)
        dp = dp_state.float()
        # Reconstruct states and intermediate values for backward pass
        eta = []
        h = []
        p_states = []
        u_states = []
        
        # Forward pass to cache intermediate values
        for i in range(n):
            alpha_i = alpha[:, i] * gamma[:, i]
            beta_i = beta[:, i]
            r_i = r[:, i]
            s_i = s[:, i]
            decay_i = torch.exp(log_decay[:, i])
            q_i = q[:, i]

            eta_i = torch.einsum("... d e, ... d -> ... e", u_state, beta_i)
            
            u_states.append(u_state.unsqueeze(1))
            p_states.append(p_state.unsqueeze(1))
            eta.append(eta_i.unsqueeze(1))
            
            u_state = u_state + torch.einsum("... d, ... e -> ... d e", alpha_i, eta_i)
            p_state = decay_i.unsqueeze(-1) * p_state + torch.einsum(
                "... d, ... e -> ... d e", r_i, s_i
            )
            h_i = torch.einsum("... d e, ... d -> ... e", u_state, q_i)
            
            h.append(h_i.unsqueeze(1))
        
        u_states.append(u_state.unsqueeze(1))
        p_states.append(p_state.unsqueeze(1))
        
        p_states = torch.cat(p_states, dim=1)
        u_states = torch.cat(u_states, dim=1)
        eta = torch.cat(eta, dim=1)
        h = torch.cat(h, dim=1)
        
        # Compute dh
        dh = torch.einsum("... d e, ... e -> ... d", p_states[:, :n], do)

        for i in range(n-1, -1, -1):
            # Spectral update
            h_i = h[:, i]
            do_i = do[:, i]
            s_i = s[:, i]
            r_i = r[:, i]
            if i < n - 1:
                decay_i_1 = torch.exp(log_decay[:, i + 1])
                dp = decay_i_1 * dp
                
            dp += torch.einsum("... d, ... e -> ... d e", h_i, do_i)
            dr[:, i] = torch.einsum("... d e, ... e -> ... d", dp, s_i)
            ds[:, i] = torch.einsum("... d e, ... d -> ... e", dp, r_i)
            
            # Unitary update
            eta_i = eta[:, i]
            alpha_i_ = alpha[:, i]
            gamma_i = gamma[:, i]
            alpha_i = alpha_i_ * gamma_i
            beta_i = beta[:, i]
            dh_i = dh[:, i]
            # TODO: check if this is correct
            u_i_1 = u_state[:, i]
            u_i = u_state[:, i+1]
            
            du += torch.einsum("... d, ... e -> ... d e", q_i, dh_i)
            dalpha_i = torch.einsum("... d e, ... e -> ... d", du, eta_i)
            deta_i = torch.einsum("... d e, ... d -> ... e", du, alpha_i)
            dbeta[:, i] = torch.einsum("... d e, ... d -> ... e", u_i_1, deta_i)
            dq[:, i] = torch.einsum("... d e, ... e -> ... d", u_i, dh_i)
            dgamma[:, i] = dalpha_i * alpha_i_
            dalpha[:, i] = dalpha_i * gamma_i

        return (dq.to(dtype), dalpha.to(dtype), dr.to(dtype), dbeta.to(dtype), 
                ds.to(dtype), dgamma.to(dtype), dlog_decay.to(dtype), None, None, None)

def polar_ag_torch(
    q: torch.Tensor,
    alpha: torch.Tensor,
    r: torch.Tensor,
    norm_style: str = "none",
    beta: Optional[torch.Tensor] = None,
    s: Optional[torch.Tensor] = None,
    gamma: Optional[torch.Tensor] = None,
    log_decay: Optional[torch.Tensor] = None,
    u_state: Optional[torch.Tensor] = None,
    p_state: Optional[torch.Tensor] = None,
    alpha_act: str = "none",
    r_act: str = "none",
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
        alpha_act: Activation function for alpha, choose from ['none', 'relu', 'sigmoid', 'silu']
        r_act: Activation function for r, choose from ['none', 'relu', 'sigmoid', 'silu']

    Returns:
        Output tensor (B, N, H, E)
        Final U state tensor (B, H, D, D)
        Final P state tensor (B, H, D, E)
    """
    assert alpha_act in ["none", "relu", "sigmoid", "silu"], "Invalid alpha activation function"
    assert r_act in ["none", "relu", "sigmoid", "silu"], "Invalid r activation function"
    
    dtype = q.dtype
    # Convert to float32 for better numerical stability
    q = q.float()
    alpha = alpha.float()
    if beta is not None:
        beta = beta.float()
    r = r.float()
    if s is not None:
        s = s.float()
        
    q = act_torch(q, alpha_act)
    alpha = act_torch(alpha, alpha_act)
    if beta is not None:
        beta = act_torch(beta, alpha_act)
    r = act_torch(r, r_act)
    if s is not None:
        s = act_torch(s, r_act)
    
    # normalize vectors
    if norm_style == "alpha":
        alpha = F.normalize(alpha, dim=-1)
        if beta is not None:
            beta = F.normalize(beta, dim=-1)
    elif norm_style == "r":
        r = F.normalize(r, dim=-1)
        if s is not None:
            s = F.normalize(s, dim=-1)
    elif norm_style == "alpha_r":
        alpha = F.normalize(alpha, dim=-1)
        r = F.normalize(r, dim=-1)
        if beta is not None:
            beta = F.normalize(beta, dim=-1)
        if s is not None:
            s = F.normalize(s, dim=-1)
            
    if beta is None:
        beta = alpha
    
    if s is None:
        s = r
            
    return PolarAgTorchFunction.apply(q, alpha, r, norm_style, beta, s, 
                                  gamma, log_decay, u_state, p_state)
    
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

    o, u_state, p_state = polar_ag_torch(
        q=q,
        alpha=alpha,
        beta=beta,
        r=r,
        s=s,
        norm_style="alpha_r",
        gamma=gamma,
        log_decay=log_decay,
        u_state=u_state,
        p_state=p_state,
    )
    print(f"Output shape: {o.shape}")
    print(f"U state shape: {u_state.shape}")
    print(f"P state shape: {p_state.shape}")