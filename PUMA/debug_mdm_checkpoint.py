#!/usr/bin/env python3
"""
debug_ckpt.py

Standalone diagnostic script for your checkpoint:
 /lus/eagle/projects/lighthouse-purdue/rai53/PUMA/ckpts/date=2026-02-24-12-30/step=50000.pt
"""

import torch
from transformers import AutoTokenizer
from model.transformer import MDMTransformer, MDMConfig

CKPT_PATH = "/lus/eagle/projects/lighthouse-purdue/rai53/PUMA/ckpts/date=2026-02-24-12-30/step=50000.pt"
TOKENIZER_NAME = "Qwen/Qwen2-0.5B"   # change if needed


def infer_vocab_from_ckpt(sd):
    for k, v in sd.items():
        if "lm_head.weight" in k:
            return v.shape
        if "embed_tokens.weight" in k:
            return v.shape
    raise RuntimeError("Could not infer vocab size from checkpoint")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    vocab_size, hidden = infer_vocab_from_ckpt(sd)
    print("Checkpoint vocab_size:", vocab_size)
    print("Checkpoint hidden_size:", hidden)

    print("\nBuilding tiny test model just to inspect weights...")
    cfg = MDMConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=hidden * 3,
        num_layers=2,
        num_attention_heads=2,
        num_kv_heads=2,
        max_position=32,
        causal=False,
    )

    model = MDMTransformer(cfg).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    print("\n==== LOAD STATE DICT ====")
    print("Missing keys:", missing[:20])
    print("Unexpected keys:", unexpected[:20])

    print("\nLoading tokenizer...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    try:
        tok_vocab = tok.vocab_size or len(tok.get_vocab())
    except:
        tok_vocab = "unknown"

    print("\n==== VOCAB CHECK ====")
    print("Tokenizer vocab_size:", tok_vocab)
    print("Checkpoint vocab_size:", vocab_size)

    print("\n==== SPECIAL TOKENS ====")
    print("mask_token_id:", getattr(tok, "mask_token_id", None))
    print("pad_token_id :", getattr(tok, "pad_token_id", None))
    print("unk_token_id :", getattr(tok, "unk_token_id", None))
    print("eos_token_id :", getattr(tok, "eos_token_id", None))

    print("\n==== TROUBLE TOKEN ====")
    try:
        print("151643 ->", tok.convert_ids_to_tokens(151643))
    except:
        print("Cannot decode 151643")

    print("\n==== LOGIT CHECK ====")
    xt = torch.randint(0, vocab_size, (1, 16), device=device)

    with torch.no_grad():
        logits = model(xt)

    probs = torch.softmax(logits[0, 0], dim=-1)
    topk = torch.topk(probs, 10)

    print("Top-10 token ids:", topk.indices.tolist())
    print("Top-10 probs:", topk.values.tolist())
    print("Entropy:", -(probs * torch.log(probs + 1e-12)).sum().item())

    print("\nDone.\n")


if __name__ == "__main__":
    main()
