"""
Main LoMDM (Learnable-Order Masked Diffusion Model) class.

This brings together all components:
- Diffusion transformer backbone
- Forward scheduler α_φ(x, t)
- Reverse scheduler α_ψ(z_t, t)
- Training with RLOO gradient estimator
- Sampling with learned generation order
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .config import LoMDMConfig
from .backbone import DiffusionTransformer
from .scheduler import ForwardScheduler, ReverseScheduler
from .diffusion import sample_forward_process
from .losses import compute_lomdm_loss, compute_rloo_loss


class LoMDM(nn.Module):
    """
    Learnable-Order Masked Diffusion Model.
    
    LoMDM extends standard masked diffusion models (like MDLM) by learning
    position-dependent noise schedules that determine the generation order.
    
    Key components:
    1. Backbone transformer θ: Predicts token distributions given masked input
    2. Forward scheduler φ: Learns which positions to mask first during training
    3. Reverse scheduler ψ: Learns which positions to unmask first during generation
    
    The model is trained end-to-end by minimizing the NELBO (Eq. 3):
    L = L_main + L_velocity
    
    Where:
    - L_main encourages accurate reconstruction
    - L_velocity aligns forward and reverse schedulers
    
    During generation, the learned scheduler guides which positions are unmasked
    first, leading to improved sample quality compared to random ordering.
    """
    
    def __init__(self, config: LoMDMConfig):
        super().__init__()
        self.config = config
        
        # Main backbone transformer
        self.backbone = DiffusionTransformer(config)
        
        # Forward scheduler network (for training - operates on clean x)
        self.forward_scheduler = ForwardScheduler(
            hidden_size=config.hidden_size,
            num_heads=config.scheduler_num_heads,
            mlp_ratio=config.scheduler_mlp_ratio,
            c1=config.c1,
            c2=config.c2,
        )
        
        # Reverse scheduler network (for inference - operates on masked z_t)
        self.reverse_scheduler = ReverseScheduler(
            hidden_size=config.hidden_size,
            num_heads=config.scheduler_num_heads,
            mlp_ratio=config.scheduler_mlp_ratio,
            c1=config.c1,
            c2=config.c2,
        )
        
        # Store mask token id
        self.mask_token_id = config.mask_token_id
    
    def get_backbone_features(
        self,
        input_ids: torch.Tensor,
        stop_gradient: bool = True,
    ) -> torch.Tensor:
        """Get backbone hidden states (before lm_head) for scheduler networks.
        Stop-gradient by default per Section 4.2."""
        if stop_gradient:
            with torch.no_grad():
                # x = (B, seq_len, embed_dim)
                # feature = (B, seq_len, embed_dim)
                # x[i][j] = mask token 
                # featur[i][j] = any value
                _, features = self.backbone(input_ids, return_hidden_states=True)
            return features
        else:
            _, features = self.backbone(input_ids, return_hidden_states=True)
            return features
    
    def forward(
        self,
        input_ids: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the backbone only (for inference).
        
        Args:
            input_ids: Token IDs (possibly masked), shape (B, L)
            t: Time values (optional), shape (B,)
            attention_mask: Attention mask (optional)
            
        Returns:
            logits: Output logits, shape (B, L, V)
        """
        logits, _ = self.backbone(input_ids, t, attention_mask)
        return logits
    
    def training_step(
        self,
        x: torch.Tensor, 
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform one training step following Algorithm 1.
        
        Steps:
        1. Get backbone features from clean x (stop-gradient)
        2. Compute forward scheduler α_φ(x, t) and A_φ(x, t)
        3. Sample two masked sequences z1_t, z2_t ~ q_αφ(·|x)
        4. For each z_t: compute L_main + L_velocity
        5. Compute RLOO loss for gradient estimation
        6. Return combined loss
        
        Args:
            x: Clean token sequence, shape (B, L)
            t: Time values, shape (B,)
            
        Returns:
            loss: Total loss for backpropagation
            loss_dict: Dictionary with loss components
        """
        B, L = x.shape
        
        # Step 1: Get backbone features from clean x (with stop-gradient)
        clean_features = self.get_backbone_features(x, stop_gradient=True)
        
        # Step 2: Compute forward scheduler
        alpha_phi, velocity_phi = self.forward_scheduler(clean_features, t)
        #[B, seq_len]
        
        # Step 3: Sample two masked sequences independently
        # Detach alpha for sampling (gradients for φ flow through RLOO log q and velocity terms only)
        z1_t, mask1 = sample_forward_process(x, alpha_phi.detach(), self.mask_token_id)
        z2_t, mask2 = sample_forward_process(x, alpha_phi.detach(), self.mask_token_id)
        
        # Step 4: Single batched forward pass for both z1 and z2 (per author: pairwise batching)
        z_both = torch.cat([z1_t, z2_t], dim=0)  # (2B, L)
        t_both = t.repeat(2) if self.config.time_conditioning else None
        logits_both, hidden_both = self.backbone(
            z_both, t_both, return_hidden_states=True,
        )
        features_both = hidden_both.detach()
        logits1, logits2 = logits_both.chunk(2, dim=0)
        features_z1, features_z2 = features_both.chunk(2, dim=0)

        _, velocity_psi1 = self.reverse_scheduler(features_z1, t)
        loss1, loss_dict1 = compute_lomdm_loss(
            logits1, x, velocity_phi, velocity_psi1, mask1, self.mask_token_id,
        )

        _, velocity_psi2 = self.reverse_scheduler(features_z2, t)
        loss2, loss_dict2 = compute_lomdm_loss(
            logits2, x, velocity_phi, velocity_psi2, mask2, self.mask_token_id,
        )
        
        # Step 5: Compute RLOO loss for φ gradient
        rloo_loss = compute_rloo_loss(
            z1_t, z2_t, alpha_phi, self.mask_token_id, loss1, loss2
        )
        
        # Step 6: Combine losses — average per-sample losses over batch
        avg_loss = 0.5 * (loss1.mean() + loss2.mean())
        total_loss = avg_loss + rloo_loss
        
        # Aggregate loss dict
        loss_dict = {
            "total_loss": total_loss,
            "avg_loss": avg_loss,
            "rloo_loss": rloo_loss,
            "main_loss": 0.5 * (loss_dict1["main_loss"] + loss_dict2["main_loss"]),
            "velocity_loss": 0.5 * (loss_dict1["velocity_loss"] + loss_dict2["velocity_loss"]),
            "mask_ratio": 0.5 * (mask1.float().mean() + mask2.float().mean()),
        }
        
        return total_loss, loss_dict
    
    def get_generation_velocity(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the generation velocity Â_ψ(z_t, t) for sampling.
        
        Higher velocity positions will be unmasked first.
        
        Args:
            z_t: Current masked sequence, shape (B, L)
            t: Current time, shape (B,)
            
        Returns:
            velocity: Denoising velocity per position, shape (B, L)
        """
        features = self.get_backbone_features(z_t, stop_gradient=True)
        _, velocity_psi = self.reverse_scheduler(features, t)
        return velocity_psi
    
    def sample_step(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        use_learned_scheduler: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Perform one reverse diffusion step from z_t to z_s.
        
        Args:
            z_t: Current masked sequence, shape (B, L)
            t: Current time, shape (B,)
            s: Target time (s < t), shape (B,)
            use_learned_scheduler: Whether to use learned scheduler
            temperature: Sampling temperature
            
        Returns:
            z_s: Sampled sequence at time s
        """
        from .diffusion import get_reverse_transition_probs
        import torch.nn.functional as F
        
        B, L = z_t.shape
        
        # Get model predictions
        logits = self.forward(z_t, t if self.config.time_conditioning else None)

        # Apply temperature to logits before computing transition probabilities
        if temperature != 1.0:
            logits = logits / temperature

        if use_learned_scheduler:
            # Get learned scheduler values
            features = self.get_backbone_features(z_t, stop_gradient=True)
            alpha_t, velocity_t = self.reverse_scheduler(features, t)
            alpha_s, _ = self.reverse_scheduler(features, s)
        else:
            # Use standard MDLM linear schedule
            alpha_t = 1.0 - t.unsqueeze(-1).expand(-1, L)
            alpha_s = 1.0 - s.unsqueeze(-1).expand(-1, L)

        # Get transition probabilities (SUBS zero-masking applied inside)
        probs = get_reverse_transition_probs(
            logits, z_t, alpha_t, alpha_s, self.mask_token_id
        )
        
        # Sample
        probs_flat = probs.view(-1, probs.size(-1))
        samples = torch.multinomial(probs_flat.clamp(min=1e-10), num_samples=1)
        z_s = samples.squeeze(-1).view(B, L)
        
        return z_s
    
    
    def get_num_params(self) -> Dict[str, int]:
        """Get number of parameters for each component."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        return {
            "backbone": count_params(self.backbone),
            "forward_scheduler": count_params(self.forward_scheduler),
            "reverse_scheduler": count_params(self.reverse_scheduler),
            "total": count_params(self),
        }