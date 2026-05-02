"""
Diffusion process utilities for LoMDM.

Implements the forward (noising) and reverse (denoising) processes
with position-dependent learnable schedulers.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def normalized_sigmoid(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute normalized sigmoid: NormSig(v)_i = σ(v_i) - Σ_j σ(v_j) / L
    
    This ensures the scheduler priorities are zero-mean across positions,
    so the overall denoising velocity is controlled by c1 while c2 only
    modulates the relative ordering.
    
    Args:
        x: Input tensor of shape (..., L) where L is sequence length
        dim: Dimension along which to normalize
        
    Returns:
        Normalized sigmoid values of same shape as input
    """
    sigmoid_x = torch.sigmoid(x)
    mean_sigmoid = sigmoid_x.mean(dim=dim, keepdim=True)
    return sigmoid_x - mean_sigmoid


def compute_alpha(
    scheduler_logits: torch.Tensor,
    t: torch.Tensor,
    c1: float,
    c2: float,
) -> torch.Tensor:
    """
    Compute the position-dependent noise schedule α(x,t) or α(z_t,t).
    
    From Equation (4) and (6) in the paper:
    α^(i)(x,t) = 1 - t^{c1 + c2 * [NormSig(g(f(x)))]_i}
    
    Args:
        scheduler_logits: Output from scheduler network g_φ or g_ψ, shape (B, L)
        t: Time values, shape (B,) or (B, 1)
        c1: Base velocity parameter (controls overall speed)
        c2: Variance parameter (controls spread of generation priorities)
        
    Returns:
        Alpha values α(x,t) of shape (B, L)
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)  # (B,) -> (B, 1)
    
    # Compute normalized sigmoid for position-dependent exponents
    norm_sig = normalized_sigmoid(scheduler_logits, dim=-1)  # (B, L)
    
    # Compute exponents: c1 + c2 * NormSig(...)
    exponents = c1 + c2 * norm_sig  # (B, L)
    
    # Clamp t to avoid numerical issues at t=0
    t_clamped = t.clamp(min=1e-6)
    
    # Compute α = 1 - t^exponent
    # Using exp(exponent * log(t)) for numerical stability
    alpha = 1.0 - torch.exp(exponents * torch.log(t_clamped))
    
    return alpha


def compute_velocity(
    scheduler_logits: torch.Tensor,
    t: torch.Tensor,
    c1: float,
    c2: float,
) -> torch.Tensor:
    """
    Compute the denoising velocity A(x,t) or Â(z_t,t).
    
    From Equation (5) and (7) in the paper:
    A^(i)(x,t) = (c1 + c2 * [NormSig(g(f(x)))]_i) / t
    
    The velocity represents how fast each position should be denoised.
    Higher velocity = earlier generation (lower mask probability over time).
    
    Args:
        scheduler_logits: Output from scheduler network g_φ or g_ψ, shape (B, L)
        t: Time values, shape (B,) or (B, 1)
        c1: Base velocity parameter
        c2: Variance parameter
        
    Returns:
        Velocity values A(x,t) of shape (B, L)
    """
    if t.dim() == 1:
        t = t.unsqueeze(-1)  # (B,) -> (B, 1)
    
    # Compute normalized sigmoid
    norm_sig = normalized_sigmoid(scheduler_logits, dim=-1)  # (B, L)
    
    # Compute velocity = (c1 + c2 * NormSig(...)) / t
    velocity = (c1 + c2 * norm_sig) / t.clamp(min=1e-6)
    
    return velocity


def sample_forward_process(
    x: torch.Tensor,
    alpha: torch.Tensor,
    mask_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample from the forward masking process q(z_t | x) given position-dependent α.
    
    For masked diffusion, each position i is independently masked with probability (1 - α^(i)).
    Once a position is masked, it stays masked (absorbing state property).
    
    Args:
        x: Clean token sequence, shape (B, L)
        alpha: Position-dependent noise schedule values α(x,t), shape (B, L)
        mask_token_id: Token ID for [MASK]
        
    Returns:
        z_t: Noisy (masked) sequence, shape (B, L)
        mask: Boolean mask indicating which positions are masked, shape (B, L)
    """
    # Sample Bernoulli mask: position i is unmasked with probability α^(i)
    # i.e., masked with probability (1 - α^(i))
    uniform_noise = torch.rand_like(alpha)
    mask = uniform_noise > alpha  # True where position should be masked
    
    # Create masked sequence
    z_t = torch.where(mask, mask_token_id, x)
    
    return z_t, mask


def compute_forward_log_prob(
    x: torch.Tensor,
    z_t: torch.Tensor,
    alpha: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """
    Compute log probability log q_α(z_t | x) for the forward process.
    
    This is needed for the RLOO gradient estimator.
    
    q(z_t^(i) | x^(i)) = α^(i) if z_t^(i) = x^(i) (unmasked)
                       = 1 - α^(i) if z_t^(i) = [MASK] (masked)
    
    Args:
        x: Clean sequence, shape (B, L)
        z_t: Masked sequence, shape (B, L)
        alpha: Noise schedule values, shape (B, L)
        mask_token_id: Token ID for [MASK]
        
    Returns:
        Log probability, shape (B,)
    """
    is_masked = (z_t == mask_token_id)
    
    # Clamp alpha for numerical stability
    alpha_clamped = alpha.clamp(min=1e-6, max=1.0 - 1e-6)
    
    # Log prob for each position
    log_prob_unmasked = torch.log(alpha_clamped)
    log_prob_masked = torch.log(1.0 - alpha_clamped)
    
    log_prob_per_position = torch.where(is_masked, log_prob_masked, log_prob_unmasked)
    
    # Sum over positions
    log_prob = log_prob_per_position.sum(dim=-1)
    
    return log_prob


def get_reverse_transition_probs(
    logits: torch.Tensor,
    z_t: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_s: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """
    Compute reverse transition probabilities p_θ(z_s | z_t).
    
    For masked diffusion (SUBS parameterization from MDLM):
    - If z_t^(i) ≠ [MASK]: z_s^(i) = z_t^(i) (carry-over unmasking)
    - If z_t^(i) = [MASK]: 
        p(z_s^(i) | z_t) = Cat((1-α_s)*m + (α_s-α_t)*x_θ^(i) / (1-α_t))
    
    Args:
        logits: Model predictions, shape (B, L, V+1)
        z_t: Current noisy sequence, shape (B, L)
        alpha_t: Current alpha values, shape (B, L)
        alpha_s: Target alpha values (s < t), shape (B, L)
        mask_token_id: Token ID for [MASK]
        
    Returns:
        Transition probabilities, shape (B, L, V+1)
    """
    B, L, V = logits.shape

    # SUBS zero-masking: set mask token logit to -inf before softmax
    # so that <x_θ^(i), m> = 0 (Appendix C)
    logits = logits.clone()
    logits[:, :, mask_token_id] = float('-inf')

    # Softmax to get predicted token probabilities
    x_theta = F.softmax(logits, dim=-1)  # (B, L, V)
    
    is_masked = (z_t == mask_token_id).unsqueeze(-1)  # (B, L, 1)
    
    # For masked positions: compute mixture
    # (1 - α_s) * one_hot([MASK]) + (α_s - α_t) / (1 - α_t) * x_θ
    alpha_t_expanded = alpha_t.unsqueeze(-1)  # (B, L, 1)
    alpha_s_expanded = alpha_s.unsqueeze(-1)  # (B, L, 1)
    
    # One-hot for mask token
    mask_one_hot = F.one_hot(
        torch.tensor(mask_token_id, device=z_t.device),
        num_classes=V
    ).float().unsqueeze(0).unsqueeze(0)  # (1, 1, V)
    
    # Denoising ratio
    denom = (1.0 - alpha_t_expanded).clamp(min=1e-6)
    denoising_ratio = (alpha_s_expanded - alpha_t_expanded) / denom
    
    # Mixture distribution for masked positions
    p_masked = (1.0 - alpha_s_expanded) * mask_one_hot + denoising_ratio * x_theta
    
    # For unmasked positions: deterministic copy (one-hot of current token)
    p_unmasked = F.one_hot(z_t, num_classes=V).float()  # (B, L, V)
    
    # Combine
    p_zs = torch.where(is_masked, p_masked, p_unmasked)
    
    return p_zs


def sample_reverse_step(
    logits: torch.Tensor,
    z_t: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_s: torch.Tensor,
    mask_token_id: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample from the reverse process p_θ(z_s | z_t).
    
    Args:
        logits: Model predictions, shape (B, L, V+1)
        z_t: Current noisy sequence, shape (B, L)
        alpha_t: Current alpha values, shape (B, L)
        alpha_s: Target alpha values (s < t), shape (B, L)
        mask_token_id: Token ID for [MASK]
        temperature: Sampling temperature
        
    Returns:
        Sampled z_s, shape (B, L)
    """
    # Apply temperature to logits before computing transition probabilities
    if temperature != 1.0:
        logits = logits / temperature

    # Get transition probabilities (SUBS zero-masking applied inside)
    probs = get_reverse_transition_probs(
        logits, z_t, alpha_t, alpha_s, mask_token_id
    )  # (B, L, V)

    # Sample from categorical distribution
    B, L, V = probs.shape
    probs_flat = probs.view(-1, V)  # (B*L, V)
    samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)  # (B*L,)
    z_s = samples.view(B, L)
    
    return z_s
