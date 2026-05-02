"""
Learnable scheduler networks for LoMDM.

Implements the forward scheduler α_φ(x, t) and reverse scheduler α_ψ(z_t, t)
that determine position-dependent generation ordering.

From Section 4.2 and Figure 3 of the paper:
- Each scheduler consists of 1 Transformer Block + 1 MLP Layer
- They share the same architecture but have separate parameters
- Features come from the backbone transformer (with stop-gradient)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class SchedulerTransformerBlock(nn.Module):
    """
    Single transformer block for the scheduler network.
    
    This is a simpler version compared to the backbone, consisting of:
    - Multi-head self-attention
    - Feed-forward MLP
    Using pre-norm architecture.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # Attention
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        
        # MLP
        intermediate_size = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, H = x.shape
        
        # Self-attention with pre-norm
        residual = x
        x = self.ln1(x)
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, H)
        attn_output = self.o_proj(attn_output)
        
        x = residual + attn_output
        
        # MLP with pre-norm
        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        
        return x


class SchedulerNetwork(nn.Module):
    """
    Scheduler network g_φ or g_ψ that outputs position-dependent scheduler logits.
    
    Architecture (from Figure 3):
    - 1 Transformer Block (self-attention + MLP)
    - 1 MLP Layer (projects to scalar per position)
    
    Input: Features from backbone transformer, shape (B, L, H)
    Output: Scheduler logits, shape (B, L)
    
    These logits are then passed through NormSig and used to compute
    the position-dependent α and A values.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 1 Transformer Block
        self.transformer_block = SchedulerTransformerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        
        # 1 MLP Layer that projects to scalar per position
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),  # Output scalar per position
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute scheduler logits from backbone features.
        
        Args:
            features: Backbone hidden states, shape (B, L, H)
            
        Returns:
            logits: Scheduler logits, shape (B, L)
        """
        # Process through transformer block
        x = self.transformer_block(features)
        
        # Project to scalar per position
        logits = self.output_mlp(x).squeeze(-1)  # (B, L)
        
        return logits


class ForwardScheduler(nn.Module):
    """
    Forward scheduler α_φ(x, t) that computes position-dependent noise schedule
    given the clean sequence x.
    
    This determines which positions are more likely to be masked first during
    the forward diffusion process, effectively learning a corruption order
    that complements the model's denoising capabilities.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        c1: float = 0.7,
        c2: float = 0.65,
    ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        
        self.network = SchedulerNetwork(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
    
    def forward(
        self,
        features: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute α_φ(x, t) and A_φ(x, t) from backbone features of clean sequence.
        
        Args:
            features: Backbone features from clean x, shape (B, L, H)
            t: Time values, shape (B,) or (B, 1)
            
        Returns:
            alpha: Position-dependent α values, shape (B, L)
            velocity: Position-dependent A values (velocity), shape (B, L)
        """
        from .diffusion import compute_alpha, compute_velocity
        
        # Get scheduler logits
        logits = self.network(features)  # (B, L)
        
        # Compute alpha and velocity
        alpha = compute_alpha(logits, t, self.c1, self.c2)
        velocity = compute_velocity(logits, t, self.c1, self.c2)
        
        return alpha, velocity
    
    def get_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Get raw scheduler logits (before NormSig)."""
        return self.network(features)


class ReverseScheduler(nn.Module):
    """
    Reverse scheduler α_ψ(z_t, t) that computes position-dependent denoising schedule
    given the masked sequence z_t.
    
    This determines which positions should be unmasked first during generation,
    effectively learning to prioritize positions that are easier to reconstruct
    given the current context.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        c1: float = 0.7,
        c2: float = 0.65,
    ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        
        self.network = SchedulerNetwork(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
    
    def forward(
        self,
        features: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute α_ψ(z_t, t) and Â_ψ(z_t, t) from backbone features of masked sequence.
        
        Args:
            features: Backbone features from masked z_t, shape (B, L, H)
            t: Time values, shape (B,) or (B, 1)
            
        Returns:
            alpha: Position-dependent α values, shape (B, L)
            velocity: Position-dependent Â values (velocity), shape (B, L)
        """
        from .diffusion import compute_alpha, compute_velocity
        
        # Get scheduler logits
        logits = self.network(features)  # (B, L)
        
        # Compute alpha and velocity
        alpha = compute_alpha(logits, t, self.c1, self.c2)
        velocity = compute_velocity(logits, t, self.c1, self.c2)
        
        return alpha, velocity
    
    def get_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Get raw scheduler logits (before NormSig)."""
        return self.network(features)


