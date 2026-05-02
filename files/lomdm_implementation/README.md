# LoMDM: Learnable-Order Masked Diffusion Model

A complete PyTorch implementation of the paper:
**"Unifying Masked Diffusion Models with Various Generation Orders and Beyond"**
by Chunsan Hong, Sanghyun Lee, and Jong Chul Ye (arXiv:2602.02112)

## Overview

LoMDM (Learnable-order Masked Diffusion Model) extends standard masked diffusion models (like MDLM) by learning position-dependent noise schedules that determine the generation order. Unlike prior work that either hard-codes an ordering (e.g., blockwise left-to-right) or post-trains an ordering policy, LoMDM jointly learns the generation ordering and diffusion backbone through a single NELBO objective from scratch.

## Key Features

- **Position-dependent schedules**: Each position has its own noise schedule α^(i)(x,t)
- **Learnable generation order**: The model learns which positions to unmask first
- **Joint training**: Backbone and scheduler networks trained together via single NELBO
- **RLOO gradient estimator**: Low-variance gradient estimation for the discrete sampling operation
- **Multiple sampling strategies**: Standard DDPM, cached DDPM, confidence-based, semi-autoregressive

## Architecture

```
                ┌─────────────────────────────────────┐
                │         N-Layer Transformer         │
                │          (Backbone θ)               │
                │      [Bidirectional + RoPE]         │
                └─────────────────┬───────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
         ┌──────────▼──────────┐     ┌──────────▼──────────┐
         │  Forward Scheduler  │     │  Reverse Scheduler  │
         │       α_φ(x,t)      │     │      α_ψ(z_t,t)     │
         │  [1 TF Block + MLP] │     │  [1 TF Block + MLP] │
         └─────────────────────┘     └─────────────────────┘
              (Training)                  (Inference)
```

## Key Equations

**Position-dependent schedule (Equations 4-7):**
```
α^(i)_φ(x,t) = 1 - t^{c1 + c2·[NormSig(g_φ(f(x)))]_i}
A^(i)_φ(x,t) = (c1 + c2·[NormSig(g_φ(f(x)))]_i) / t
```

Where:
- `NormSig(v)_i = σ(v_i) - Σ_j σ(v_j)/L` (normalized sigmoid)
- `c1` controls overall velocity (default: 0.7)
- `c2` controls variance of generation order priority (default: 0.65)
- `c1 > c2` is required for valid scheduler

**NELBO (Equation 3):**
```
L_LoMDM = ∫ E_qαφ [ Σ_i <z_t^(i), m> * (L_main^(i) + L_velocity^(i)) ] dt
```

Where:
- `L_main = -A_φ^(i) * log<x_θ^(i)(z_t,t), x^(i)>` (weighted reconstruction)
- `L_velocity = A_φ(log A_φ - log Â_ψ) - (A_φ - Â_ψ)` (velocity matching)

**RLOO Gradient Estimator (Equation 8):**
```
L_rloo = 0.5 * log(q_αφ(z1|x)/q_αφ(z2|x)) * Sgd(L_z1 - L_z2)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train on LM1B (128 tokens, BERT tokenizer)
python train.py --dataset lm1b --batch_size 256 --max_steps 1000000

# Train on OpenWebText (1024 tokens, GPT-2 tokenizer)
python train.py --dataset openwebtext --tokenizer gpt2 --max_length 1024 \
    --batch_size 32 --max_steps 1000000

# With mixed precision and wandb logging
python train.py --dataset lm1b --fp16 --wandb_project lomdm
```

### Sampling

```bash
# Generate samples
python sample.py --checkpoint outputs/checkpoint-final.pt --num_samples 64

# With cached sampling (3-4x faster)
python sample.py --checkpoint outputs/checkpoint-final.pt --method ddpm_cache

# Semi-autoregressive for long sequences
python sample.py --checkpoint outputs/checkpoint-final.pt --semi_ar \
    --total_length 2048 --block_length 1024

# Compute generative perplexity
python sample.py --checkpoint outputs/checkpoint-final.pt --compute_ppl
```

### Evaluation

```bash
# Evaluate perplexity on test set
python evaluate.py --checkpoint outputs/checkpoint-final.pt --dataset lm1b

# Zero-shot evaluation on other datasets
python evaluate.py --checkpoint outputs/checkpoint-final.pt --dataset wikitext
```

## Project Structure

```
lomdm_implementation/
├── lomdm/
│   ├── __init__.py          # Package exports
│   ├── config.py             # Configuration classes
│   ├── backbone.py           # Diffusion Transformer (DiT)
│   ├── scheduler.py          # Forward/Reverse scheduler networks
│   ├── diffusion.py          # Forward/reverse diffusion processes
│   ├── losses.py             # NELBO loss computation
│   ├── model.py              # Main LoMDM model class
│   ├── sampling.py           # Sampling algorithms
│   └── utils.py              # Training utilities
├── train.py                  # Training script
├── sample.py                 # Sampling script
├── evaluate.py               # Evaluation script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Hyperparameters

From Table 4 in the paper, the best configuration is:
- `c1 = 0.7` (overall velocity)
- `c2 = 0.65` (variance of generation order priority)

When `c2 = 0`, the scheduler is non-learnable and reduces to polynomial form `α(t) = 1 - t^c1`.
Setting `(c1, c2) = (1, 0)` recovers standard MDLM.

## Results (from paper)

| Method | LM1B | LM1B (packed) | OWT |
|--------|------|---------------|-----|
| MDLM | 27.0 | 31.8 | 23.2 |
| BD3LM (L'=4) | - | 28.2 | 20.7 |
| GenMD4 | 26.9 | 30.0 | ≥21.8 |
| **LoMDM (Ours)** | **25.4** | **27.2** | **20.4** |

LoMDM achieves the 1M-step MDLM performance (PPL=23.0 on OWT) at only **180K steps**, demonstrating ~5.5x faster learning efficiency.

## Key Insights

1. **Why learnable order helps**: The forward scheduler A_φ assigns higher weights to tokens the model predicts correctly, focusing training on learnable positions.

2. **Velocity-confidence correlation**: During training, A_φ and Â_ψ become positively correlated with reconstruction confidence (Figure 4), showing the scheduler learns meaningful generation priorities.

3. **Cache sampling speedup**: Due to "carry-over unmasking", cached DDPM is 3-4x faster than standard sampling while maintaining quality.

## Citation

```bibtex
@article{hong2026unifying,
  title={Unifying Masked Diffusion Models with Various Generation Orders and Beyond},
  author={Hong, Chunsan and Lee, Sanghyun and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2602.02112},
  year={2026}
}
```

## Acknowledgements

This implementation builds upon:
- [MDLM](https://github.com/kuleshov-group/mdlm) for the masked diffusion foundation
- The original LoMDM paper for the learnable order framework
