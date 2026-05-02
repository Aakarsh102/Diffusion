"""
Loss computations for LoMDM.

Implements the NELBO objective from Equation (3) in the paper:
L_LoMDM = ∫ E_qαφ [ Σ_i <z_t^(i), m> * (L_main^(i) + L_velocity^(i)) ] dt

Where:
- L_main = -A_φ^(i) * log<x_θ^(i)(z_t, t), x^(i)>
- L_velocity = A_φ^(i) * (log A_φ^(i) - log Â_ψ^(i)) - (A_φ^(i) - Â_ψ^(i))

Also implements the RLOO (Leave-One-Out) gradient estimator from Equation (8).
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict


def compute_main_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    velocity_phi: torch.Tensor,
    mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """
    Compute the main reconstruction loss L_main for masked positions.

    L_main^(i) = -A_φ^(i) * log<x_θ^(i)(z_t, t), x^(i)>

    Per Eq 3, this is a sum over masked positions (not averaged by num_masked).
    We normalize by sequence length L (a constant) for scale stability.

    SUBS parametrization requires the mask token logit to be set to -inf
    before softmax so that <x_θ, m> = 0 (Appendix C).

    Returns per-sample losses (B,) needed for the RLOO estimator (Eq 8).

    Args:
        logits: Model output logits, shape (B, L, V)
        targets: Ground truth token IDs, shape (B, L)
        velocity_phi: Forward scheduler velocity A_φ, shape (B, L)
        mask: Boolean mask indicating masked positions, shape (B, L)
        mask_token_id: Token ID for [MASK], used for SUBS zero-masking

    Returns:
        loss_per_sample: Per-sample loss values, shape (B,)
    """
    B, L, V = logits.shape

    # SUBS zero-masking: set mask token logit to -inf before softmax
    # so that <x_θ^(i), m> = 0 (Appendix C)
    logits = logits.clone()
    logits[:, :, mask_token_id] = float('-inf')

    # Cross-entropy loss per position
    log_probs = F.log_softmax(logits, dim=-1)  # (B, L, V)

    # Gather log probs for target tokens
    target_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)  # (B, L)

    # Weight by velocity and mask
    # Only compute loss on masked positions (where mask is True)
    weighted_loss = -velocity_phi * target_log_probs * mask.float()  # (B, L)

    # Sum over positions per Eq 3 (not divided by num_masked).
    # Divide by L (constant) to match MDLM baseline scaling (.mean() over B*L).
    loss_per_sample = weighted_loss.sum(dim=-1) / L  # (B,)

    return loss_per_sample


def compute_velocity_loss(
    velocity_phi: torch.Tensor,
    velocity_psi: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the velocity matching loss L_velocity for masked positions.

    L_velocity^(i) = A_φ^(i) * (log A_φ^(i) - log Â_ψ^(i)) - (A_φ^(i) - Â_ψ^(i))

    Per Eq 3, this is a sum over masked positions (not averaged by num_masked).
    We normalize by sequence length L (a constant) for scale stability.

    Returns per-sample losses (B,) needed for the RLOO estimator (Eq 8).

    Args:
        velocity_phi: Forward scheduler velocity A_φ, shape (B, L)
        velocity_psi: Reverse scheduler velocity Â_ψ, shape (B, L)
        mask: Boolean mask indicating masked positions, shape (B, L)

    Returns:
        loss_per_sample: Per-sample loss values, shape (B,)
    """
    L = velocity_phi.shape[-1]
    eps = 1e-6

    # Clamp velocities and their ratio for numerical stability (per author)
    A_phi_s = velocity_phi.clamp(min=eps)
    A_psi_s = velocity_psi.clamp(min=eps)
    ratio = (A_phi_s / A_psi_s).clamp(min=eps)

    # L_velocity = A_φ * log(A_φ / Â_ψ) - (A_φ - Â_ψ), only at masked positions
    mask_bool = mask.bool()
    loss_per_position = torch.zeros_like(A_phi_s)
    loss_per_position[mask_bool] = (
        A_phi_s[mask_bool] * torch.log(ratio[mask_bool])
        - (A_phi_s[mask_bool] - A_psi_s[mask_bool])
    )

    # Sum over positions per Eq 3, divide by L to match MDLM baseline scaling
    loss_per_sample = loss_per_position.sum(dim=-1) / L  # (B,)

    return loss_per_sample


def compute_lomdm_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    velocity_phi: torch.Tensor,
    velocity_psi: torch.Tensor,
    mask: torch.Tensor,
    mask_token_id: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the full LoMDM loss: L_main + L_velocity.

    Returns per-sample loss (B,) for use in the RLOO estimator (Eq 8),
    plus a dict with batch-mean scalars for logging.

    Args:
        logits: Model output logits, shape (B, L, V)
        targets: Ground truth token IDs, shape (B, L)
        velocity_phi: Forward scheduler velocity A_φ, shape (B, L)
        velocity_psi: Reverse scheduler velocity Â_ψ, shape (B, L)
        mask: Boolean mask indicating masked positions, shape (B, L)
        mask_token_id: Token ID for [MASK], used for SUBS zero-masking

    Returns:
        loss_per_sample: Per-sample total loss, shape (B,)
        loss_dict: Dictionary with batch-mean loss components for logging
    """
    main_loss = compute_main_loss(logits, targets, velocity_phi, mask, mask_token_id)
    velocity_loss = compute_velocity_loss(velocity_phi, velocity_psi, mask)

    loss_per_sample = main_loss + velocity_loss  # (B,)

    loss_dict = {
        "loss": loss_per_sample.mean(),
        "main_loss": main_loss.mean(),
        "velocity_loss": velocity_loss.mean(),
    }

    return loss_per_sample, loss_dict


def compute_rloo_loss(
    z1_t: torch.Tensor,
    z2_t: torch.Tensor,
    alpha_phi: torch.Tensor,
    mask_token_id: int,
    loss_z1: torch.Tensor,
    loss_z2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the RLOO (Reinforce Leave-One-Out) gradient estimator loss.

    From Equation (8) in the paper:
    L_rloo = 1/2 * log(q_αφ(z1_t|x) / q_αφ(z2_t|x)) * Sgd(L_z1 - L_z2)

    This provides a low-variance gradient estimator for the scheduler φ
    through the discrete sampling operation. Per Appendix E.1, each
    sample's log-ratio is multiplied by that same sample's loss difference.

    Args:
        z1_t: First sampled masked sequence, shape (B, L)
        z2_t: Second sampled masked sequence, shape (B, L)
        alpha_phi: Forward scheduler α values, shape (B, L)
        mask_token_id: Token ID for [MASK]
        loss_z1: Per-sample loss on z1_t, shape (B,)
        loss_z2: Per-sample loss on z2_t, shape (B,)

    Returns:
        rloo_loss: Scalar RLOO loss (mean over batch)
    """
    z1_masked = (z1_t == mask_token_id)
    z2_masked = (z2_t == mask_token_id)

    # Clamp alpha for numerical stability
    alpha_clamped = alpha_phi.clamp(min=1e-6, max=1.0 - 1e-6)

    # Log prob for each masking state
    log_prob_unmasked = torch.log(alpha_clamped)
    log_prob_masked = torch.log(1.0 - alpha_clamped)

    # Compute log q(z1|x)
    log_q_z1 = torch.where(z1_masked, log_prob_masked, log_prob_unmasked)
    log_q_z1 = log_q_z1.sum(dim=-1)  # (B,)

    # Compute log q(z2|x)
    log_q_z2 = torch.where(z2_masked, log_prob_masked, log_prob_unmasked)
    log_q_z2 = log_q_z2.sum(dim=-1)  # (B,)

    # Log ratio: log(q(z1|x) / q(z2|x))
    log_ratio = log_q_z1 - log_q_z2  # (B,)

    # Per-sample loss difference (stop-gradiented) -- Eq 8
    loss_diff = loss_z1.detach() - loss_z2.detach()  # (B,)

    # RLOO loss: per-sample product, then average over batch
    rloo_loss = 0.5 * log_ratio * loss_diff  # (B,)

    return rloo_loss.mean()

