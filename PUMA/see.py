#!/usr/bin/env python3
"""
generate_mdm_samples.py
Generates 5-6 samples using standard (non-block) MDM iterative unmasking
from the PUMA paper (Section 2).

Usage:
  # With a trained checkpoint (Sudoku):
  python generate_mdm_samples.py --ckpt path/to/ckpt.pt --dataset sudoku

  # With a trained checkpoint (TinyGSM):
  python generate_mdm_samples.py --ckpt path/to/ckpt.pt --dataset tinygsm

  # Demo mode (random weights, just to exercise the loop):
  python generate_mdm_samples.py --demo
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from model.transformer import MDMTransformer, MDMConfig
from sampling import mdm_sampling


# ---------- configs matching the paper's yaml_files ----------
CONFIGS = {
    "sudoku": dict(
        model=dict(
            vocab_size=11, hidden_size=256, intermediate_size=768,
            num_layers=8, num_attention_heads=8, num_kv_heads=8,
            max_position=162, rms_norm_eps=1e-6, dropout=0.0,
            tie_lm_head=False, bias_qkv=True, causal=False,
        ),
        mask_id=10,
        seq_len=162,       # 81 clues + 81 solution cells
        prompt_len=81,     # first 81 = clue region
    ),
    "tinygsm": dict(
        model=dict(
            vocab_size=151645, hidden_size=512, intermediate_size=1536,
            num_layers=14, num_attention_heads=8, num_kv_heads=8,
            max_position=512, rms_norm_eps=1e-6, dropout=0.0,
            tie_lm_head=True, bias_qkv=True, causal=False,
        ),
        mask_id=151644,
        seq_len=512,
        prompt_len=0,
    ),
}


def load_model(dataset, ckpt_path, device):
    cfg = CONFIGS[dataset]
    model_cfg = MDMConfig(**cfg["model"])
    model = MDMTransformer(model_cfg).to(device)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("WARNING: No checkpoint — using random weights (demo mode).")

    n = sum(p.numel() for p in model.parameters())
    print(f"Model: {n/1e6:.1f}M params on {device}")
    model.eval()
    return model


def make_masked_input(dataset, n_samples, device):
    """Build the initial fully-masked (or prompt + masked) tensor."""
    cfg = CONFIGS[dataset]
    mask_id = cfg["mask_id"]
    L = cfg["seq_len"]
    prompt_len = cfg["prompt_len"]

    if dataset == "sudoku":
        # prompt = zeros (empty clues), solution region = all masked
        prompt = torch.zeros(n_samples, prompt_len, dtype=torch.long, device=device)
        masked = torch.full((n_samples, L - prompt_len), mask_id, dtype=torch.long, device=device)
        return torch.cat([prompt, masked], dim=1)
    else:
        # fully masked
        return torch.full((n_samples, L), mask_id, dtype=torch.long, device=device)


def pretty_sudoku(flat):
    lines = []
    for r in range(9):
        row = [str(int(flat[r*9+c])) for c in range(9)]
        parts = [" ".join(row[i:i+3]) for i in (0, 3, 6)]
        lines.append(" | ".join(parts))
        if r in (2, 5):
            lines.append("------+-------+------")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="sudoku", choices=["sudoku", "tinygsm"])
    parser.add_argument("--demo", action="store_true", help="Run with random weights")
    parser.add_argument("--n_samples", type=int, default=6)
    parser.add_argument("--confidence", type=str, default="top_k",
                        choices=["top_k", "top_k_margin", "entropy"])
    parser.add_argument("--unmasking_num", type=int, default=2, help="|S| per step")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build sampling config (SimpleNamespace mimics the OmegaConf object sampling.py expects)
    sampling_cfg = SimpleNamespace(
        temperature=args.temperature,
        confidence=args.confidence,
        unmasking_num=args.unmasking_num,
    )

    # load model
    model = load_model(args.dataset, args.ckpt, device)
    mask_id = CONFIGS[args.dataset]["mask_id"]

    # build initial masked input
    xt = make_masked_input(args.dataset, args.n_samples, device)
    n_to_unmask = int((xt == mask_id).sum().item())
    print(f"\nGenerating {args.n_samples} samples  |  {n_to_unmask} total masks to fill")
    print(f"  confidence={args.confidence}  |S|={args.unmasking_num}  temp={args.temperature}\n")

    # ---- run standard MDM sampling (from sampling.py) ----
    with torch.no_grad():
        out = mdm_sampling(model, xt, mask_id, sampling_cfg, device=device)

    remaining = int((out == mask_id).sum().item())
    print(f"Done. Remaining masks: {remaining}\n")

    # ---- display results ----
    for i in range(args.n_samples):
        print(f"{'='*50}")
        print(f"  Sample {i+1}")
        print(f"{'='*50}")
        if args.dataset == "sudoku":
            print(pretty_sudoku(out[i, 81:].cpu().numpy()))
        else:
            toks = out[i].cpu().tolist()
            print(f"  tokens (first 60): {toks[:60]} ...")
        print()


if __name__ == "__main__":
    main()
