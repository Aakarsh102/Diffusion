"""
LM1B dataset for PUMA training with SentencePiece tokenizer.

Supports two workflows:
  1. Preprocessing:  python data_lm1b.py --preprocess --data_dir <raw_text_dir> --sp_model <sp.model> --out_dir <out>
     Reads all .txt files in data_dir (one sentence per line), tokenizes with
     SentencePiece, and writes memory-mapped arrays for fast training.

  2. Training:  imported by train_lm1b.py — loads the preprocessed memmap files.
"""

import os
import json
import argparse
import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ---------------------------------------------------------------------------
# Preprocessing: tokenize raw text -> memmap
# ---------------------------------------------------------------------------

def pretokenize_lm1b(data_dir: str, sp_model_path: str, out_dir: str, max_len: int = 256):
    """
    Tokenize all .txt files under *data_dir* (one sentence per line) using
    the SentencePiece model at *sp_model_path*.  Each sentence is either
    truncated or padded to *max_len* tokens.  Results are saved as:
        out_dir/labels.bin   — uint32 memmap  (N, max_len)
        out_dir/meta.json    — {n_examples, max_len, vocab_size, sp_model}
    """
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    vocab_size = sp.get_piece_size()
    eos_id = sp.eos_id()
    pad_id = eos_id  # pad with EOS

    # Collect all text files
    txt_files = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(data_dir)
        for f in files if f.endswith(".txt")
    )
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under {data_dir}")

    # First pass: count lines so we can allocate the memmap
    total_lines = 0
    for path in txt_files:
        with open(path, "r", encoding="utf-8") as fh:
            for _ in fh:
                total_lines += 1
    print(f"Found {total_lines} lines across {len(txt_files)} files")

    os.makedirs(out_dir, exist_ok=True)
    labels_path = os.path.join(out_dir, "labels.bin")
    labels_mm = np.memmap(labels_path, dtype=np.uint32, mode="w+", shape=(total_lines, max_len))

    idx = 0
    for path in txt_files:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    # keep blank lines as all-pad
                    labels_mm[idx] = pad_id
                    idx += 1
                    continue
                ids = sp.encode(line, out_type=int)
                if len(ids) > max_len:
                    ids = ids[:max_len]
                else:
                    ids = ids + [pad_id] * (max_len - len(ids))
                labels_mm[idx] = np.array(ids, dtype=np.uint32)
                idx += 1
                if idx % 500_000 == 0:
                    print(f"  tokenized {idx}/{total_lines}")
    labels_mm.flush()

    meta = dict(
        n_examples=int(total_lines),
        max_len=max_len,
        vocab_size=vocab_size,
        sp_model=os.path.abspath(sp_model_path),
        eos_id=eos_id,
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Done — wrote {total_lines} examples to {out_dir}")


# ---------------------------------------------------------------------------
# Dataset: read preprocessed memmap
# ---------------------------------------------------------------------------

class LM1BDataset(Dataset):
    """
    Reads the preprocessed memmap produced by pretokenize_lm1b().
    Returns {"labels": LongTensor[max_len], "prompt_mask": BoolTensor[max_len]}
    prompt_mask is all-False (unconditional LM — no prompt region).
    """

    def __init__(self, data_dir: str):
        meta_path = os.path.join(data_dir, "meta.json")
        with open(meta_path, "r") as fh:
            self.meta = json.load(fh)
        n = self.meta["n_examples"]
        L = self.meta["max_len"]
        labels_path = os.path.join(data_dir, "labels.bin")
        self.labels = np.memmap(labels_path, dtype=np.uint32, mode="r", shape=(n, L))
        self.L = L

    def __len__(self):
        return self.meta["n_examples"]

    def __getitem__(self, idx):
        ids = torch.from_numpy(self.labels[idx].astype(np.int64)).long()
        prompt_mask = torch.zeros(self.L, dtype=torch.bool)
        return {"labels": ids, "prompt_mask": prompt_mask}


# ---------------------------------------------------------------------------
# Helper: build train/val loaders
# ---------------------------------------------------------------------------

def setup_lm1b_loaders(data_dir: str, batch_size: int, val_ratio: float = 0.02, seed: int = 2026, num_workers: int = 4):
    ds = LM1BDataset(data_dir)
    n_val = max(1, int(len(ds) * val_ratio))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CLI entry point for preprocessing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess LM1B text for PUMA training")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with .txt files (one sentence per line)")
    parser.add_argument("--sp_model", type=str, required=True, help="Path to SentencePiece .model file")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for memmap files")
    parser.add_argument("--max_len", type=int, default=256, help="Max sequence length (default 256)")
    args = parser.parse_args()

    if args.preprocess:
        pretokenize_lm1b(args.data_dir, args.sp_model, args.out_dir, args.max_len)
    else:
        print("Pass --preprocess to tokenize raw text. Otherwise import this module.")
