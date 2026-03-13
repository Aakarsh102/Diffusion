# -------------------------------------------------------
# bidirectional Transformer (no time-embedding)
# based on QWen architecture
# -------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple
from model.utils import RotaryEmbedding, RMSNorm, apply_rope
from torch.backends.cuda import sdp_kernel
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



# -------------------------------------------------------
# TransformerConfig
# -------------------------------------------------------
@dataclass
class MDMConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    max_position: int = 1024
    rms_norm_eps: float = 1e-6
    tie_lm_head: bool = False
    bias_qkv: bool = True
    dropout: float = 0.0
    causal: bool = False
    arm_init: Optional[str] = None
    predict_next_token: bool = False   # True when initialized from ARM

    def head_dim(self) -> int:
        # dimension for each head
        assert self.hidden_size % self.num_attention_heads == 0
        return self.hidden_size // self.num_attention_heads


# -------------------------------------------------------
# MLP
# -------------------------------------------------------
class SwiGLU(nn.Module):
    """
    SwiGLU: (xW1) ⊗ swish(xW2) @ W3
    Shapes:
      in=hidden, hidden=intermediate_size
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias = False)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias = False)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        b = F.silu(self.w2(x))
        out = self.w3(a * b)
        return self.dropout(out)


# -------------------------------------------------------
# Qwen attention
# -------------------------------------------------------
class QwenAttention(nn.Module):
    def __init__(self, config: MDMConfig):
        super().__init__()
        self.config = config
        H = config.num_attention_heads
        K = config.num_kv_heads
        d = config.hidden_size
        Hd = config.head_dim()
        assert H % K ==0, "num_attention_heads must be divisible by num_kv_heads"
        self.kv_repeats = H // K

        # qkv projection
        self.q_proj = nn.Linear(d, H * Hd, bias=config.bias_qkv)
        self.k_proj = nn.Linear(d, K * Hd, bias=config.bias_qkv)
        self.v_proj = nn.Linear(d, K * Hd, bias=config.bias_qkv)
        self.o_proj = nn.Linear(H * Hd, d, bias=config.bias_qkv)

        # rope
        self.rope = RotaryEmbedding(Hd, max_position=config.max_position)

        self.dropout_p = config.dropout

    def _shape(self, x, B, L, H, Hd):
        return x.view(B, L, H, Hd)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # for now we just have a full bidirectional attention implemented
        # L = seqlen
        B, L, _ = x.shape

        H = self.config.num_attention_heads
        K = self.config.num_kv_heads
        Hd = self.config.head_dim()

        # qkv projection and reshape
        q = self._shape(self.q_proj(x), B, L, H, Hd)
        k = self._shape(self.k_proj(x), B, L, K, Hd)
        v = self._shape(self.v_proj(x), B, L, K, Hd)

        # RoPE
        cos, sin = self.rope(L)
        q, k = apply_rope(q, k, cos, sin)
        
        # bidirectional attention
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        
        # wrapper for flashattention debugging:
        # with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=(self.dropout_p if self.training else 0.0),
                    is_causal=getattr(self.config, "causal", False),
                    enable_gqa=(H != K),
            )  # (B, H, L, Hd) :contentReference[oaicite:2]{index=2}
        
        out = out.transpose(1, 2).contiguous().view(B, L, H * Hd)
        return self.o_proj(out)


# -------------------------------------------------------
# Transformeer Block
# -------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, config: MDMConfig):
        super().__init__()
        self.config = config
        self.attn_norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.attn = QwenAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, dropout = config.dropout)

    def forward(self, x: torch.Tensor):
        # pre-norm
        h = x + self.attn(self.attn_norm(x))
        h = h + self.mlp(self.mlp_norm(h))
        return h


# -------------------------------------------------------
# Full model
# -------------------------------------------------------
class MDMTransformer(nn.Module):
    def __init__(self, config: MDMConfig):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)
        if config.tie_lm_head:
            self.lm_head.weight = self.emb.weight

    def forward(self, input_ids: torch.Tensor):
        B, L = input_ids.shape
        device = input_ids.device

        # token embeddings
        x = self.emb(input_ids)

        # forward pass
        for layer in self.layers:
            x = layer(x)

        # final layer
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x


if __name__ == "__main__":
    print("Building a MDM Transformer...")

    cfg = MDMConfig(
        vocab_size = 2000,
        hidden_size = 256,
        intermediate_size = 1024,
        num_layers = 4,
        num_attention_heads = 4,
        num_kv_heads = 2)
        
    model = MDMTransformer(cfg)
    B, L = 2, 16
    
    # random input
    x = torch.randint(0, cfg.vocab_size, (B, L))
    print(f"Input shape: {x.shape}")

    # model forward pass
    logits = model(x)
    print(f"Logits shape: {logits.shape}")