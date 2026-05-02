"""
Diffusion Transformer (DiT) backbone for LoMDM.

Implements a transformer architecture similar to the one used in MDLM,
with rotary position embeddings (RoPE) and optional time conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .config import LoMDMConfig


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) as used in LLaMA and modern transformers."""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cache
        self._set_cos_sin_cache(max_seq_len)
    
    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    # cos, sin: (seq_len, dim)
    # q, k: (batch, heads, seq_len, dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional rotary position embeddings."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rotary: bool = True,
        rotary_dim: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rotary = use_rotary
        self.rotary_dim = rotary_dim
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        if use_rotary:
            self.rotary_emb = RotaryEmbedding(rotary_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if self.use_rotary:
            cos, sin = self.rotary_emb(q, L)
            # Only apply to first rotary_dim dimensions
            q_rot = q[..., :self.rotary_dim]
            k_rot = k[..., :self.rotary_dim]
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
            q = torch.cat([q_rot, q[..., self.rotary_dim:]], dim=-1)
            k = torch.cat([k_rot, k[..., self.rotary_dim:]], dim=-1)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        use_rotary: bool = True,
        rotary_dim: int = 64,
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            use_rotary=use_rotary,
            rotary_dim=rotary_dim,
        )
        
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm architecture
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer (DiT) backbone for masked diffusion language models.
    
    This is a bidirectional transformer that takes in (possibly masked) token sequences
    and produces logits over the vocabulary for each position.
    
    Following MDLM, we use:
    - Bidirectional attention (no causal mask)
    - Rotary position embeddings (RoPE)
    - Time-agnostic architecture (no time conditioning by default)
    """
    
    def __init__(self, config: LoMDMConfig):
        super().__init__()
        self.config = config
        
        # Token embedding (vocab + 1 for mask token if needed)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Optional time embedding
        if config.time_conditioning:
            self.time_embed = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 4),
                nn.SiLU(),
                nn.Linear(config.hidden_size * 4, config.hidden_size),
            )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                dropout=config.hidden_dropout_prob,
                use_rotary=config.use_rotary_embeddings,
                rotary_dim=config.rotary_dim,
            )
            for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.final_ln = nn.LayerNorm(config.hidden_size)
        
        # Output projection to vocabulary (weight-tied with embed_tokens)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Weight tying: share weights between input embeddings and output projection
        # Standard in MDLM/BERT/GPT-2 baselines
        #self.lm_head.weight = self.embed_tokens.weight
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def get_timestep_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal timestep embedding."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(
        self,
        input_ids: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the diffusion transformer.
        
        Args:
            input_ids: Token IDs, shape (B, L)
            t: Timesteps (optional, only used if time_conditioning=True), shape (B,)
            attention_mask: Attention mask (optional), shape (B, L)
            return_hidden_states: Whether to return hidden states for scheduler networks
            
        Returns:
            logits: Output logits over vocabulary, shape (B, L, V)
            hidden_states: Last layer hidden states (optional), shape (B, L, H)
        """
        B, L = input_ids.shape
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)  # (B, L, H)
        
        # Add time embedding if using time conditioning
        if self.config.time_conditioning and t is not None:
            t_emb = self.get_timestep_embedding(t, self.config.hidden_size)
            t_emb = self.time_embed(t_emb)  # (B, H)
            hidden_states = hidden_states + t_emb.unsqueeze(1)
        
        # Process attention mask
        if attention_mask is not None:
            # Convert from (B, L) to (B, 1, 1, L) for broadcasting
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Store hidden states before final projection if needed
        #final_hidden_states = hidden_states
        
        # Final layer norm and projection
        final_hidden_states = self.final_ln(hidden_states)
        logits = self.lm_head(final_hidden_states)  # (B, L, V)
        
        if return_hidden_states:
            return logits, final_hidden_states
        return logits, None