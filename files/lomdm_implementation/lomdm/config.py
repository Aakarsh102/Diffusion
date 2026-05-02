"""
Configuration for LoMDM.
"""

from dataclasses import dataclass


@dataclass
class LoMDMConfig:
    # Vocabulary and sequence
    vocab_size: int = 30522
    max_seq_length: int = 256
    mask_token_id: int = 103

    # Backbone Transformer
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1

    # Scheduler network
    scheduler_num_heads: int = 12
    scheduler_mlp_ratio: float = 4.0

    # LoMDM hyperparameters (Table 4): c1 > c2 required (Proposition C.3)
    c1: float = 0.7
    c2: float = 0.65

    time_conditioning: bool = False

    # RoPE
    use_rotary_embeddings: bool = True
    rotary_dim: int = 64

    initializer_range: float = 0.02
