import json
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

VOCAB_TOKENS = [str(i) for i in range(10)] + ["[SEP]", "[MASK]"]
TOKEN2ID = {tok:i for i,tok in enumerate(VOCAB_TOKENS)}
ID2TOKEN = VOCAB_TOKENS

SEP_ID = TOKEN2ID["[SEP]"]
MASK_ID = TOKEN2ID["[MASK]"]
VOCAB_SIZE = len(VOCAB_TOKENS)

def encode_tokens(tokens: list[str]) -> list[int]:
    return [TOKEN2ID[i] for i in tokens]

def decode_ids(ids: list[int]) -> list[str]:
    return [ID2TOKEN[i] for i in ids]

class ChecksumDataset(Dataset):
    def __init__(self, jsonl_path: str | Path):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError
        self.records = []
        with self.path.open("r") as f:
            for line in f:
                rec = json.loads(line)
                tokens = rec["tokens"]
                order0 = [i - 1 for i in rec["order"]]

                self.records.append({
                    "input_ids": torch.tensor(encode_tokens(tokens), dtype=torch.long),
                    "order": torch.tensor(order0, dtype=torch.long),
                    "meta": {
                        "order_id": rec.get("order_id"),
                        "order_name": rec.get("order_name"),
                        "bases": rec.get("bases"),
                        "sums": rec.get("sums"),
                    }
                })

        if len(self.records) == 0:
            raise ValueError("No records loaded from JSONL.")
        Ls = {r["input_ids"].numel() for r in self.records}
        assert Ls == {10}, f"Expected all sequences length 10, got lengths {Ls}"

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx: int):
        return self.records[idx]


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.stack([b['input_ids'] for b in batch], dim = 0)
    orders = torch.stack([b["order"]     for b in batch], dim=0)
    metas = [b["meta"] for b in batch]
    return {"input_ids": input_ids, "order": orders, "meta": metas}

class TokenPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size = VOCAB_SIZE, max_len = 10, d_model = 32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        self.d_model = d_model

        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: LongTensor [B, L]
        returns:   FloatTensor [B, L, d_model]
        """
        B, L = input_ids.shape
        device = input_ids.device
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        token_embeddings = self.token_emb(input_ids)    # [B, L, d_model]
        pos_embeddings   = self.pos_emb(pos)            # [B, L, d_model]

        return token_embeddings + pos_embeddings  

def build_revealed_mask(order: torch.Tensor, step: int) -> torch.Tensor:
    """
    Given a batch of orders and a step, mark which positions are revealed.
      order: LongTensor [B, L]  (0-indexed permutation of positions)
      step:  int in [0..L]      (how many positions have been revealed)
    returns:
      revealed: BoolTensor [B, L]  (True where revealed)
    Example:
      order[b] = [2,0,3,1,...]; step=2 -> positions {2,0} are revealed.
    """
    B, L = order.shape
    revealed = torch.zeros((B, L), dtype=torch.bool, device=order.device)
    if step > 0:
        idx = order[:, :step]
        arange_b = torch.arange(B, device = order.device). unsqueeze(1).expand(-1, step)
        revealed[arange_b, idx] = True 
    return revealed 
    
if __name__ == "__main__":
    jsonl_path = "data/checksum10_orders_50k.jsonl"  # update if your file lives elsewhere
    # jsonl_path = ""
    ds = ChecksumDataset(jsonl_path)

    # split 90/10
    g = torch.Generator().manual_seed(42)
    n_train = int(0.9 * len(ds))
    n_val   = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, collate_fn=collate_batch)

    emb = TokenPositionalEmbedding(vocab_size=VOCAB_SIZE, max_len=10, d_model=32)

    batch = next(iter(train_loader))
    x = batch["input_ids"]        # [B,L]
    z = batch["order"]            # [B,L]
    h = emb(x)                    # [B,L,32]

    # Example: reveal mask after 3 order steps
    revealed = build_revealed_mask(z, step=3)  # True for 3 positions per row
    print("input_ids shape:", x.shape)
    print("order shape:", z.shape)
    print("embeddings shape:", h.shape)
    print("revealed mask shape:", revealed.shape, "revealed count per row:", revealed.sum(dim=1)[:5])
    print(x)
    print("*********************")
    print(z)
    
