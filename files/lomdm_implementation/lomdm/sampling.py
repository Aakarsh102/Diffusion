"""
Sampling algorithms for LoMDM.

Implements various sampling strategies including:
- Standard ancestral sampling (DDPM)
- Cached ancestral sampling (faster)
- Confidence-based sampling with learned scheduler
- Semi-autoregressive generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass


@dataclass
class SamplingOutput:
    """Output from sampling."""
    samples: torch.Tensor  # Generated sequences
    intermediate_samples: Optional[List[torch.Tensor]] = None  # Trajectories if requested
    

class LoMDMSampler:
    """
    Sampler for LoMDM that leverages learned generation ordering.
    
    The key insight is that the reverse scheduler Â_ψ(z_t, t) tells us
    which positions are easier to reconstruct at the current time.
    Positions with higher velocity should be unmasked first.
    """
    
    def __init__(
        self,
        model: nn.Module,
        mask_token_id: int,
    ):
        self.model = model
        self.mask_token_id = mask_token_id
    
    @torch.no_grad()
    def sample_ddpm(
        self,
        batch_size: int,
        seq_length: int,
        num_steps: int = 1000,
        temperature: float = 1.0,
        use_learned_scheduler: bool = True,
        device: Optional[torch.device] = None,
        return_trajectory: bool = False,
    ) -> SamplingOutput:
        """
        Standard ancestral sampling following the reverse diffusion process.
        
        Args:
            batch_size: Number of sequences to generate
            seq_length: Length of sequences
            num_steps: Number of diffusion steps
            temperature: Sampling temperature
            use_learned_scheduler: Use learned vs standard schedule
            device: Device for generation
            return_trajectory: Whether to return intermediate samples
            
        Returns:
            SamplingOutput with generated samples
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        # Start from fully masked
        z = torch.full(
            (batch_size, seq_length),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        
        trajectory = [z.clone()] if return_trajectory else None
        
        # Time schedule
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t = timesteps[i].expand(batch_size)
            s = timesteps[i + 1].expand(batch_size)
            
            z = self.model.sample_step(
                z, t, s,
                use_learned_scheduler=use_learned_scheduler,
                temperature=temperature,
            )
            
            if return_trajectory:
                trajectory.append(z.clone())
        
        return SamplingOutput(
            samples=z,
            intermediate_samples=trajectory,
        )
    
    @torch.no_grad()
    def sample_ddpm_cache(
        self,
        batch_size: int,
        seq_length: int,
        num_steps: int = 1000,
        temperature: float = 1.0,
        use_learned_scheduler: bool = True,
        device: Optional[torch.device] = None,
    ) -> SamplingOutput:
        """
        Cached ancestral sampling (faster than standard DDPM).
        
        The key optimization: once a position is unmasked, its prediction
        is cached and reused. This is valid because of "carry-over unmasking"
        - unmasked positions deterministically copy their values.
        
        Args:
            batch_size: Number of sequences to generate
            seq_length: Length of sequences
            num_steps: Number of diffusion steps
            temperature: Sampling temperature
            use_learned_scheduler: Use learned vs standard schedule
            device: Device for generation
            
        Returns:
            SamplingOutput with generated samples
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        # Start from fully masked
        z = torch.full(
            (batch_size, seq_length),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        
        # Cache for unmasked token probabilities
        cached_probs = None
        
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t = timesteps[i].expand(batch_size)
            s = timesteps[i + 1].expand(batch_size)
            
            # Check which positions are still masked
            is_masked = (z == self.mask_token_id)  # (B, L)
            
            if is_masked.any():
                # Get model predictions only for masked positions
                logits = self.model(z, t if self.model.config.time_conditioning else None)
                # SUBS zero-masking: set mask token logit to -inf before softmax
                logits[:, :, self.mask_token_id] = float('-inf')
                probs = F.softmax(logits / temperature, dim=-1)

                # Get scheduler values
                if use_learned_scheduler:
                    features = self.model.get_backbone_features(z, stop_gradient=True)
                    alpha_t, _ = self.model.reverse_scheduler(features, t)
                    alpha_s, _ = self.model.reverse_scheduler(features, s)
                else:
                    alpha_t = 1.0 - t.unsqueeze(-1).expand(-1, seq_length)
                    alpha_s = 1.0 - s.unsqueeze(-1).expand(-1, seq_length)
                
                # Compute unmasking probability: (α_s - α_t) / (1 - α_t)
                unmask_prob = (alpha_s - alpha_t) / (1.0 - alpha_t).clamp(min=1e-6)
                unmask_prob = unmask_prob.clamp(0, 1)
                
                # Sample which positions to unmask
                unmask_mask = torch.rand_like(unmask_prob) < unmask_prob
                unmask_mask = unmask_mask & is_masked  # Only unmask currently masked
                
                # Sample tokens for positions to unmask
                if unmask_mask.any():
                    B, L, V = probs.shape
                    probs_flat = probs.view(B * L, V)
                    samples = torch.multinomial(probs_flat.clamp(min=1e-10), 1).squeeze(-1)
                    samples = samples.view(B, L)
                    
                    # Update only unmasked positions
                    z = torch.where(unmask_mask, samples, z)
        
        return SamplingOutput(samples=z)
    
    @torch.no_grad()
    def sample_confidence(
        self,
        batch_size: int,
        seq_length: int,
        num_steps: int = 1000,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> SamplingOutput:
        """
        Confidence-based sampling using the learned scheduler.
        
        At each step, we unmask the positions with highest predicted
        confidence (highest velocity Â_ψ), following the learned
        generation order.
        
        Args:
            batch_size: Number of sequences to generate
            seq_length: Length of sequences
            num_steps: Number of diffusion steps
            temperature: Sampling temperature
            device: Device for generation
            
        Returns:
            SamplingOutput with generated samples
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        # Start from fully masked
        z = torch.full(
            (batch_size, seq_length),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        
        # Number of tokens to unmask per step
        tokens_per_step = seq_length / num_steps
        
        for step in range(num_steps):
            # Current mask ratio
            is_masked = (z == self.mask_token_id)
            num_masked = is_masked.sum(dim=-1, keepdim=True).float()
            
            if not is_masked.any():
                break
            
            # Get model predictions and scheduler velocity
            t = torch.tensor([1.0 - step / num_steps], device=device).expand(batch_size)
            logits = self.model(z, t if self.model.config.time_conditioning else None)
            # SUBS zero-masking: set mask token logit to -inf before softmax
            logits[:, :, self.mask_token_id] = float('-inf')
            probs = F.softmax(logits / temperature, dim=-1)

            # Get generation velocity from scheduler
            features = self.model.get_backbone_features(z, stop_gradient=True)
            _, velocity = self.model.reverse_scheduler(features, t)
            
            # Mask out already unmasked positions with -inf
            velocity = torch.where(is_masked, velocity, torch.tensor(float('-inf'), device=device))
            
            # Determine how many tokens to unmask this step
            target_masked = max(0, seq_length - int((step + 1) * tokens_per_step))
            num_to_unmask = (num_masked - target_masked).clamp(min=1).long()
            
            # Select top-velocity positions to unmask
            for b in range(batch_size):
                n_unmask = min(num_to_unmask[b, 0].item(), is_masked[b].sum().item())
                if n_unmask > 0:
                    # Get top positions by velocity
                    _, top_indices = velocity[b].topk(int(n_unmask))
                    
                    # Sample tokens for these positions
                    for idx in top_indices:
                        token = torch.multinomial(probs[b, idx].clamp(min=1e-10), 1)
                        z[b, idx] = token
        
        return SamplingOutput(samples=z)
    
    @torch.no_grad()
    def sample_semi_ar(
        self,
        num_samples: int,
        total_length: int,
        block_length: int,
        num_steps_per_block: int = 100,
        temperature: float = 1.0,
        use_learned_scheduler: bool = True,
        device: Optional[torch.device] = None,
    ) -> SamplingOutput:
        """
        Semi-autoregressive generation for arbitrary-length sequences.
        
        This generates sequences longer than the training context by:
        1. Generating the first block using diffusion
        2. Using the end of the previous block as prefix for the next
        3. Repeating until desired length is reached
        
        Args:
            num_samples: Number of sequences to generate
            total_length: Total target length
            block_length: Length of each generation block (should match training)
            num_steps_per_block: Diffusion steps per block
            temperature: Sampling temperature
            use_learned_scheduler: Use learned scheduler
            device: Device for generation
            
        Returns:
            SamplingOutput with generated samples
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        # Calculate number of blocks needed
        num_blocks = (total_length + block_length - 1) // block_length
        
        # Generate first block
        output = self.sample_ddpm_cache(
            batch_size=num_samples,
            seq_length=block_length,
            num_steps=num_steps_per_block,
            temperature=temperature,
            use_learned_scheduler=use_learned_scheduler,
            device=device,
        )
        
        generated = output.samples  # (num_samples, block_length)
        
        # Generate additional blocks using previous as prefix
        for block_idx in range(1, num_blocks):
            # Use second half of previous block as prefix
            prefix_length = block_length // 2
            prefix = generated[:, -prefix_length:]  # (num_samples, prefix_length)
            
            # Initialize new block: prefix + masked
            new_length = min(block_length, total_length - generated.shape[1] + prefix_length)
            z = torch.full(
                (num_samples, new_length),
                self.mask_token_id,
                dtype=torch.long,
                device=device,
            )
            z[:, :prefix_length] = prefix
            
            # Generate this block
            timesteps = torch.linspace(1.0, 0.0, num_steps_per_block + 1, device=device)
            
            for i in range(num_steps_per_block):
                t = timesteps[i].expand(num_samples)
                s = timesteps[i + 1].expand(num_samples)
                
                # Mask indicating which positions need generation (not prefix)
                generation_mask = torch.arange(new_length, device=device) >= prefix_length
                
                # Get predictions
                logits = self.model(z, t if self.model.config.time_conditioning else None)

                # Apply temperature to logits before computing transition probabilities
                if temperature != 1.0:
                    logits = logits / temperature

                if use_learned_scheduler:
                    features = self.model.get_backbone_features(z, stop_gradient=True)
                    alpha_t, _ = self.model.reverse_scheduler(features, t)
                    alpha_s, _ = self.model.reverse_scheduler(features, s)
                else:
                    alpha_t = 1.0 - t.unsqueeze(-1).expand(-1, new_length)
                    alpha_s = 1.0 - s.unsqueeze(-1).expand(-1, new_length)

                # Compute transition probs (SUBS zero-masking applied inside)
                from .diffusion import get_reverse_transition_probs
                probs = get_reverse_transition_probs(
                    logits, z, alpha_t, alpha_s, self.mask_token_id
                )
                
                # Sample
                B, L, V = probs.shape
                probs_flat = probs.view(B * L, V)
                samples = torch.multinomial(probs_flat.clamp(min=1e-10), 1).squeeze(-1)
                z_new = samples.view(B, L)
                
                # Only update generation positions (keep prefix fixed)
                z = torch.where(generation_mask.unsqueeze(0), z_new, z)
            
            # Append new tokens (excluding prefix which overlaps)
            generated = torch.cat([generated, z[:, prefix_length:]], dim=1)
        
        # Truncate to exact length
        generated = generated[:, :total_length]
        
        return SamplingOutput(samples=generated)


def compute_perplexity(
    model: nn.Module,
    data_loader,
    mask_token_id: int,
    num_samples: int = 100,
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute perplexity bound using importance sampling.
    
    Following MDLM, we estimate the log-likelihood using the ELBO
    and convert to perplexity.
    
    Args:
        model: LoMDM model
        data_loader: Data loader yielding token sequences
        mask_token_id: Token ID for [MASK]
        num_samples: Number of time samples for estimation
        device: Device for computation
        
    Returns:
        perplexity: Estimated perplexity bound
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                x = batch["input_ids"].to(device)
            else:
                x = batch.to(device)
            
            B, L = x.shape
            batch_nll = 0.0
            
            # Monte Carlo estimation over time
            for _ in range(num_samples):
                t = torch.rand(B, device=device)
                
                # Get forward scheduler values
                features = model.get_backbone_features(x, stop_gradient=True)
                alpha, velocity = model.forward_scheduler(features, t)
                
                # Sample masked sequence
                from .diffusion import sample_forward_process
                z_t, mask = sample_forward_process(x, alpha, mask_token_id)
                
                # Get predictions
                logits = model(z_t)
                # SUBS zero-masking: set mask token logit to -inf before softmax
                logits[:, :, mask_token_id] = float('-inf')
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get log prob of correct tokens at masked positions
                target_log_probs = torch.gather(
                    log_probs, -1, x.unsqueeze(-1)
                ).squeeze(-1)
                
                # Weight by velocity and mask
                weighted_nll = -velocity * target_log_probs * mask.float()
                batch_nll += weighted_nll.sum()
            
            batch_nll /= num_samples
            total_nll += batch_nll.item()
            total_tokens += B * L
    
    # Convert to perplexity
    avg_nll = total_nll / total_tokens
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    
    return perplexity
