# order_ar_train.py
import hashlib
import json
import os, math, random, time
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, set_seed
from sklearn.model_selection import train_test_split

from model import Model

def get_cache_path(cache_dir, dataset_name, tokenizer_name, block_size):
    """Generate a unique cache path based on dataset config"""
    # Create a hash of the configuration
    config_str = f"{dataset_name}_{tokenizer_name}_{block_size}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    cache_path = Path(cache_dir) / f"tokenized_{config_hash}"
    return cache_path

def build_tokenizer(model_name = "gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    added = tokenizer.add_special_tokens({"pad_token": "<|pad|>", "mask_token": "<mask>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    return tokenizer

def load_openwebtext(block_size=1024):
    ds = load_dataset("openwebtext", cache_dir = "/scratch/gilbreth/rai53/hf_cache")["train"].train_test_split(test_size=0.005, seed = 42)
    return {"train": ds["train"], "validation": ds["test"]}

def tokenize_and_chunk(ds, tokenizer, block_size=1024, cache_dir="./data_cache"):
    """
    Tokenize and chunk dataset with caching support
    
    Args:
        ds: Dataset dict with 'train' and 'validation' splits
        tokenizer: Tokenizer to use
        block_size: Size of each chunk
        cache_dir: Directory to cache processed data
    
    Returns:
        Processed dataset dict
    """
    # Create cache directory
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate cache path
    cache_path = get_cache_path(
        cache_dir,
        "openwebtext",
        tokenizer.name_or_path,
        block_size
    )
    
    # Check if cache exists
    if cache_path.exists():
        print(f"Loading cached tokenized dataset from {cache_path}")
        try:
            cached_ds = load_from_disk(str(cache_path))
            print(f"✓ Loaded from cache: {len(cached_ds['train'])} train, {len(cached_ds['validation'])} val")
            return cached_ds
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")
    
    print(f"Tokenizing and chunking (this may take a while)...")
    print(f"Will cache to: {cache_path}")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    
    # Tokenize
    print("Step 1/2: Tokenizing...")
    tokenized = {}
    for split in ["train", "validation"]:
        tokenized[split] = ds[split].map(
            tokenize_function,
            batched=True,
            num_proc=64,
            remove_columns=ds[split].column_names,
            desc=f"Tokenizing {split}"
        )
    
    # Chunk into blocks
    def group_texts(examples):
        # Concatenate all texts
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        
        # Drop remainder
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        else:
            # If we don't have enough for even one block, pad
            if total_length > 0:
                pad_length = block_size - total_length
                for k in concatenated.keys():
                    concatenated[k].extend([tokenizer.pad_token_id] * pad_length)
                total_length = block_size
        
        # Split into chunks
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        return result
    
    print("Step 2/2: Chunking into blocks...")
    chunked = {}
    for split in ["train", "validation"]:
        chunked[split] = tokenized[split].map(
            group_texts,
            batched=True,
            num_proc=64,
            desc=f"Chunking {split}"
        )
    
    # Create dataset dict
    
    final_ds = DatasetDict(chunked)
    
    # Save to cache
    print(f"Saving to cache: {cache_path}")
    final_ds.save_to_disk(str(cache_path))
    
    # Save metadata
    metadata = {
        "dataset": "openwebtext",
        "tokenizer": tokenizer.name_or_path,
        "block_size": block_size,
        "vocab_size": len(tokenizer),
        "train_size": len(final_ds["train"]),
        "val_size": len(final_ds["validation"])
    }
    with open(cache_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Cached dataset: {len(final_ds['train'])} train, {len(final_ds['validation'])} val")
    
    return final_ds
    
    
class DataCollatorOrderAR:
    """
    Returns:
      x_in: masked inputs [B, L]
      x_target: ground-truth [B, L]
    """
    def __init__(self, mask_id: int, pad_id: int, mask_prob: float = 0.15, ensure_one_mask=True):
        self.mask_id = mask_id
        # self.pad_id = pad_id
        # r = random.randint(0, len(mask_prob) - 1)
        # self.mask_prob = mask_prob[r]
        # self.ensure_one_mask = ensure_one_mask
    
    def __call__(self, batch):
        x = torch.tensor([i["input_ids"] for i in batch], dtype=torch.long) # [B, L]
        x_target = x.clone()

        # not_pad = (x != self.pad_id) # [B, L]

        # mask_shape = x.shape
        # M = (torch.rand(mask_shape) < self.mask_prob) & not_pad # [B, L]
        # M = (torch.rand(mask_shape) < self.mask_prob)
        # x_in = x.clone()
        # x_in[M] = self.mask_id
#        return {"x_in": x_in, "x_target": x_target, "mask": M, "not_pad": not_pad}
        # return {"x_in": x, "x_target": x_target, "mask": M}
        return {"x_in": x, "x_target": x_target}
    

def create_masked_input(x_target, orderings, num_positions, mask_id):
    """
    Create masked input where first num_positions of each ordering are unmasked.
    
    Args:
        x_target: [B, L] ground truth tokens
        orderings: [B, num_positions] indices to unmask (first num_positions of full ordering)
        num_positions: int, same for all samples in batch
        mask_id: mask token id
    
    Returns:
        x_masked: [B, L] with only specified positions unmasked
        mask: [B, L] boolean (True = masked)
    """
    x_masked = torch.full_like(x_target, mask_id)
    B, L = x_target.shape
    for b in range(B):
        pos_to_unmask = orderings[b, :num_positions]
        x_masked[b, pos_to_unmask] = x_target[b, pos_to_unmask]
    mask = (x_masked == mask_id)
    return x_masked, mask

    
@torch.no_grad()
def eval_masked_nll(model, data_loader, device):
    model.eval()
    losses = []
    for batch in data_loader:
        x_in = batch["x_in"].to(device)
        x_target = batch["x_target"].to(device)
        losses.append(model(x_in, x_target)["token_loss"].item())

    mean_nll = sum(losses) / max(1,len(losses))
    ppl = math.exp(mean_nll)
    return mean_nll, ppl

@torch.no_grad()
def generate_samples(model, tokenizer, device, num_samples=4, seq_len=128, temperature=0.9):
    """Generate sample sequences for qualitative evaluation"""
    model.eval()
    
    samples = model.generate(
        batch_size=num_samples,
        seq_len=seq_len,
        temperature=temperature,
        sample_order=True,
        sample_token=True,
        #use_attention_mask=True
    )
    
    # Decode samples
    texts = []
    for sample in samples:
        text = tokenizer.decode(sample, skip_special_tokens=True)
        texts.append(text)
    
    return texts


""" 
#def train_step_autoregressive(model, x_in, x_target, not_pad, device, num_ar_steps=5):
#    Autoregressive training step that simulates the generation process
#    
#    Args:
#        num_ar_steps: Number of autoregressive unmasking steps per training iteration
    #print("x_in:", x_in)

    total_tok_loss = 0.0
    total_ord_loss = 0.0

    x_current = x_in.clone()
    num_steps = 0
    B = x_in.size(0)
    for step in range(num_ar_steps):
        M = (x_current == model.mask_id)
        if not M.any():
            break
        has_masks = M.any(dim=1)  # [B] - True if sample has any masks
        num_with_masks = has_masks.sum().item()
        has_masks = (M.sum(dim = -1) > 1) & has_masks

        if num_with_masks == 0:
            break

        x_fwd = x_current.clone()
        h = model.encode(x_fwd, use_mask=False)
        
        # Get token predictions and order scores
        logits_tok = model.token_head(h)  # [B, L, V]
        order_scores = model.order_head(h).squeeze(-1)  # [B, L]
        
        # Compute order policy distribution over masked positions
        order_scores_masked = order_scores.masked_fill(~M, float('-inf'))
        

        next_pos = torch.zeros(B, dtype=torch.long, device = x_in.device)
        valid_order_scores = order_scores_masked[has_masks]
        valid_probs = F.softmax(valid_order_scores, dim=-1)
       
#        if torch.isnan(valid_probs).any():
#            print(f"Warning: NaN in valid_probs at step {step}")
#            # For debugging - let's see what's happening
#            print(f"valid_order_scores min: {valid_order_scores.min()}, max: {valid_order_scores.max()}")
#            print(f"has_masks sum: {has_masks.sum()}")
#            print(f"(I added) valid order scores: {valid_order_scores}")
#            print(f"(I added) valid probs: {valid_probs}")


        valid_next_pos = torch.multinomial(valid_probs, num_samples=1).squeeze(-1)
        next_pos[has_masks] = valid_next_pos



        order_probs = F.softmax(order_scores_masked, dim=-1)  # [B, L]
        # Sample next position to unmask from the order policy
        # During training, we sample to explore different orderings
        next_pos = torch.multinomial(order_probs, num_samples=1).squeeze(-1)  # [B]
        
        # Compute token loss for the selected positions
        B = x_target.size(0)
        selected_logits = logits_tok[torch.arange(B), next_pos]  # [B, V]
        selected_targets = x_target[torch.arange(B), next_pos]  # [B]
        
        tok_loss = F.cross_entropy(selected_logits[has_masks], selected_targets[has_masks], reduction='mean')
        
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                all_probs = F.softmax(logits_tok, dim=-1)  # [B, L, V]
                if torch.isnan(all_probs).any():
                    print("Warning: NaN in all_probs")
                ent = - torch.sum(all_probs * torch.log(all_probs + 1e-9), dim = -1)
                #ent = torch.clamp(ent, max=20.0)
                #ent = ent.masked_fill(~M, float('inf'))  # only masked compete
                ent = ent.masked_fill(~M, 100)  # only masked compete
                target = torch.zeros_like(ent)
                if has_masks.any():
                    target[has_masks] = torch.softmax(-ent[has_masks], dim = -1)
                    
                

                
            if has_masks.any():
                order_logprobs = order_scores_masked[has_masks].log_softmax(dim = -1) # (B, L)
                order_loss = F.kl_div(
                    order_logprobs, 
                    target[has_masks], 
                    reduction='batchmean'
                )
            else:
                order_loss = torch.tensor(0.0, device = x_in.device)
        #order_logprobs = order_scores_masked.log_softmax(-1)  # [B, L]
        #order_loss = F.kl_div(order_logprobs, target, reduction='batchmean')

        total_tok_loss += tok_loss
        total_ord_loss += order_loss
        num_steps += 1

        x_next = x_current.clone()
        x_next[torch.arange(B), next_pos] = x_target[torch.arange(B), next_pos]
        x_current=x_next.detach();
        #x_current[torch.arange(B), next_pos] = x_target[torch.arange(B), next_pos]

    if num_steps == 0:
        return {
            "loss": torch.tensor(0.0, device=device),
            "token_loss": torch.tensor(0.0, device=device),
            "order_loss": torch.tensor(0.0, device=device)
        }
    
    avg_tok_loss = total_tok_loss / num_steps
    avg_ord_loss = total_ord_loss / num_steps
    total_loss = avg_tok_loss + model.order_weight * avg_ord_loss
    
    return {
        "loss": total_loss,
        "token_loss": avg_tok_loss.detach(),
        "order_loss": avg_ord_loss.detach()
    }


"""


def train_step_autoregressive(model, x_in, x_target, not_pad, device, num_ar_steps=5):
    total_tok_loss = 0.0
    total_ord_loss = 0.0
    x_current = x_in.clone()
    num_steps = 0
    B = x_in.size(0)
    
    
    for step in range(num_ar_steps):
        M = (x_current == model.mask_id)
        if not M.any():
            break
            
        has_masks = M.any(dim=1)
        num_with_masks = has_masks.sum().item()
        if num_with_masks == 0:
            break

        h = model.encode(x_current, use_mask=False)
        
        logits_tok = model.token_head(h)
        order_scores = model.order_head(h).squeeze(-1)
        
        order_scores_masked = order_scores.masked_fill(~M, float('-inf'))
        
        # Sample positions
        next_pos = torch.zeros(B, dtype=torch.long, device=x_in.device)
        valid_order_scores = order_scores_masked[has_masks]
        valid_probs = F.softmax(valid_order_scores.float(), dim=-1)
        valid_next_pos = torch.multinomial(valid_probs, num_samples=1).squeeze(-1)
        next_pos[has_masks] = valid_next_pos

        # Token loss
        selected_logits = logits_tok[torch.arange(B), next_pos]
        selected_targets = x_target[torch.arange(B), next_pos]
        tok_loss = F.cross_entropy(
            selected_logits[has_masks], 
            selected_targets[has_masks], 
            reduction='mean'
        )

        # Order loss - ONLY compute for samples with multiple masks
        # (need at least 2 positions to have meaningful ordering)
        num_masks_per_sample = M.sum(dim=1)  # [B]
        valid_for_order = (num_masks_per_sample >= 2) & has_masks
        
        if valid_for_order.any():
            with torch.no_grad():
                logits_fp32 = logits_tok[valid_for_order].float()
                M_valid = M[valid_for_order]
                
                all_probs = F.softmax(logits_fp32, dim=-1)
                ent = -torch.sum(all_probs * torch.log(all_probs.clamp(min=1e-10)), dim=-1)
                
                # Use finite large value instead of inf
                ent_masked = torch.where(M_valid, ent, torch.tensor(1000000.0, device=ent.device))
                
                # Compute target distribution (lower entropy = higher priority)
                target_valid = F.softmax(-ent_masked, dim=-1)
            
            # Get order scores for valid samples
            order_scores_valid = order_scores[valid_for_order].float()
            order_scores_masked_valid = order_scores_valid.masked_fill(~M_valid, -1000000.0)  # Use -100 instead of -inf
            
            # Compute log probabilities
            order_logprobs = F.log_softmax(order_scores_masked_valid, dim=-1)
            order_loss = F.kl_div(order_logprobs, target_valid, reduction='batchmean')
            
            # Sanity check: skip if we still have inf/nan
            #if not (torch.isinf(order_logprobs).any() or torch.isnan(order_logprobs).any() or 
            #        torch.isinf(target_valid).any() or torch.isnan(target_valid).any()):
            #    order_loss = F.kl_div(order_logprobs, target_valid, reduction='batchmean')
            #    
            #    # Clamp loss to prevent explosions
            #    order_loss = torch.clamp(order_loss, max=10.0)
            #else:
            #    order_loss = torch.tensor(0.0, device=x_in.device)
        else:
            order_loss = torch.tensor(0.0, device=x_in.device)


        # Inside train_step_autoregressive, after computing order_loss
        if step == 0 and num_steps % 100 == 0:  # Print occasionally
            print(f"\n=== Order Loss Debug ===")
            print(f"Order loss: {order_loss.item():.6f}")
            if valid_for_order.any():
                print(f"Order logprobs sample: {order_logprobs[0, :5]}")
                print(f"Target sample: {target_valid[0, :5]}")
                print(f"Are they nearly identical? {torch.allclose(order_logprobs.exp(), target_valid, atol=0.1)}")

        total_tok_loss += tok_loss
        total_ord_loss += order_loss
        num_steps += 1

        # Update without in-place operation
        x_next = x_current.clone()
        x_next[torch.arange(B), next_pos] = x_target[torch.arange(B), next_pos]
        x_current = x_next.detach()

    if num_steps == 0:
        return {
            "loss": torch.tensor(0.0, device=device),
            "token_loss": torch.tensor(0.0, device=device),
            "order_loss": torch.tensor(0.0, device=device)
        }
    
    avg_tok_loss = total_tok_loss / num_steps
    avg_ord_loss = total_ord_loss / num_steps
    total_loss = avg_tok_loss + model.order_weight * avg_ord_loss
    
    return {
        "loss": total_loss,
        "token_loss": avg_tok_loss.detach(),
        "order_loss": avg_ord_loss.detach()
    }

def compute_log_q_ordering(model, x_target, ordering_prefix, q_logits = None):
    """
    Compute log qθ(z<i|x) for a partial ordering.
    
    Args:
        model: the model
        x_target: [B, L] ground truth (fully unmasked)
        ordering_prefix: [B, prefix_len] the ordering z<i
    Returns:
        log_q: [B] log probability of the ordering
    """
    B, L = x_target.shape
    S = ordering_prefix.size(1)
    prefix_len = ordering_prefix.shape[1]
    #q_logits = model.get_variational_logits(x_target)  # [B, L]
    device = x_target.device
    log_q_total = torch.zeros(B, device=device)
    mask = torch.ones(B, L, dtype=torch.bool, device=device)

    for s in range(S):
        q_logprobs = F.log_softmax(q_logits.masked_fill(~mask, float('-inf')), dim=-1)  # [B, L]
        idx = ordering_prefix[:, s]  # [B]
        log_q_total += q_logprobs[torch.arange(B, device=device), idx]
        mask[torch.arange(B, device=device), idx] = False  # remove chosen index

    return log_q_total

def compute_F_theta(model, x_masked, x_target, mask, ordering_prefix, device, q_logits = None):
    """
    x_masked: [B,L] - only z_<i positions revealed, others are mask_id
    mask:     [B,L] - True where still masked (eligible k ∈ z_{≥i})
    returns:  F_theta per sample, shape [B]
    """

    B, L = x_target.shape

    # qθ over full x
   # q_logits_full = model.get_variational_logits(x_target).float()  # [B,L]
    q_logits_full = q_logits.float()
    q_logits_masked = q_logits_full.masked_fill(~mask, float('-inf'))
    q_logprobs = F.log_softmax(q_logits_masked, dim=-1)             # [B,L]
    q_probs    = q_logprobs.exp()

    h_masked = model.encode(x_masked, use_mask=False)               # [B,L,d]
    p_logits = model.order_head(h_masked).squeeze(-1).float()       # [B,L]
    p_logits_masked = p_logits.masked_fill(~mask, float('-inf'))
    p_logprobs = F.log_softmax(p_logits_masked, dim=-1)

    token_logits = model.token_head(h_masked).float()               # [B,L,V]
    logp_all = F.log_softmax(token_logits, dim=-1)                  # [B,L,V]
    gold_logp = logp_all.gather(-1, x_target.unsqueeze(-1)).squeeze(-1)  # [B,L]
    token_logprobs = torch.where(mask, gold_logp, torch.zeros_like(gold_logp))

    # Fθ = Σ_k q(k) * [log p(k) + log p(x_k) - log q(k)]
    F_theta = (q_probs * (p_logprobs + token_logprobs - q_logprobs)).sum(dim=-1)  # [B]
    return F_theta


def train_step_rloo(model, x_target, device, seq_len):
    """
    RLOO training step with SAME number of unmasked positions across batch.
    """
    B, L = x_target.shape
    mask_id = model.mask_id
    q_logits = model.get_variational_logits(x_target)  # [B, L]
    # Step 1: Sample ONE random step i for entire batch
    i = random.randint(1, L - 1)  # Single integer, not per-sample!
    with torch.no_grad():
        # q_logits = model.get_variational_logits(x_target)  # [B, L]
        
        # Each sample gets its own ordering
        z1_full = model.sample_ordering_gumbel(q_logits)  # [B, L]
        z2_full = model.sample_ordering_gumbel(q_logits)  # [B, L]

    z1_prefix = z1_full[:, :i]  # [B, i] - all samples have i unmasked
    z2_prefix = z2_full[:, :i]  # [B, i] - all samples have i unmasked
    x_masked_1, mask_1 = create_masked_input(x_target, z1_prefix, i, mask_id)
    x_masked_2, mask_2 = create_masked_input(x_target, z2_prefix, i, mask_id)
    q_logits_det   = q_logits.detach()
    F1 = compute_F_theta(model, x_masked_1, x_target, mask_1, z1_prefix, device, q_logits=q_logits_det)
    F2 = compute_F_theta(model, x_masked_2, x_target, mask_2, z2_prefix, device, q_logits=q_logits_det)

    log_q1 = compute_log_q_ordering(model, x_target, z1_prefix, q_logits=q_logits)
    log_q2 = compute_log_q_ordering(model, x_target, z2_prefix, q_logits=q_logits)

    Delta_F = F1 - F2
    loss = -(L / 2.0) * torch.mean(
        (log_q1 - log_q2) * Delta_F.detach() + F1 + F2
    )

    return {
        "loss": loss,                        # The actual training loss
        "F1": F1.mean().detach(),           # F_theta for ordering 1
        "F2": F2.mean().detach(),           # F_theta for ordering 2
        "log_q1": log_q1.mean().detach(),   # Log prob of ordering 1
        "log_q2": log_q2.mean().detach(),   # Log prob of ordering 2
        "Delta_F": Delta_F.mean().detach(), # Difference (for RLOO baseline)
        "num_unmasked": i 
    }



def main(config):
    set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    wandb.init(project = config["wandb_project"], config = config)

    tokenizer = build_tokenizer(config["model_name"])
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")
    print(f"Mask ID: {mask_id}, Pad ID: {pad_id}, EOS ID: {eos_id}")

    ds = load_openwebtext(config["block_size"])
    ds = tokenize_and_chunk(ds, tokenizer, block_size=config["block_size"])
    collator = DataCollatorOrderAR(
        mask_id=mask_id,
        pad_id=pad_id,
        mask_prob=config["mask_prob"],
    )
    train_dataloader = DataLoader(
        ds["train"],
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ds["validation"],
        batch_size=config["eval_batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    model = Model(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_len=config["block_size"],
        mask_id=mask_id,
        device=device,  # Pass device to model
        variational_policy=config.get("variational_policy", "shared_torso")
    ).to(device)

    model.eos_id = eos_id
    model.pad_id = pad_id

    optimizer = torch.optim.AdamW(model.parameters(), lr = config["lr"], betas=(0.9, 0.95), weight_decay=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    global_step = 0
    best_val_nll = float("inf")
    for epoch in range(config["epochs"]):
        model.train()
        running = {
        "loss": 0.0, 
        "F1": 0.0, 
        "F2": 0.0,
        "log_q1": 0.0,
        "log_q2": 0.0,
        "Delta_F": 0.0,
        }
        step_counts = torch.zeros(config["block_size"], dtype=torch.long)
        t0 = time.time()
        for it, batch in enumerate(train_dataloader):
            x_in = batch["x_in"].to(device)
            x_target = batch["x_target"].to(device)

            with torch.cuda.amp.autocast(enabled = (device=="cuda")):
                out = train_step_rloo(
                model, x_target, device, config["block_size"]
            )
            loss = out["loss"]

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running["loss"] += loss.item()
            running["F1"] += out["F1"].item()
            running["F2"] += out["F2"].item()
            running["log_q1"] += out["log_q1"].item()
            running["log_q2"] += out["log_q2"].item()
            running["Delta_F"] += out["Delta_F"].item()
            step_counts[out["num_unmasked"] - 1] += 1

            if (it + 1) % 50 == 0:
                for key in running:
                    running[key] /= 50
                
                wandb.log({
                    "train/loss": running["loss"],
                    "train/F1": running["F1"],
                    "train/F2": running["F2"],
                    "train/log_q1": running["log_q1"],
                    "train/log_q2": running["log_q2"],
                    "train/Delta_F": running["Delta_F"],
                    "train/step": global_step
                })
                print(f"Step {global_step}: loss={running['loss']:.4f}, "
                    f"F1={running['F1']:.4f}, F2={running['F2']:.4f}, "
                    f"log_q1={running['log_q1']:.4f}, Delta_F={running['Delta_F']:.4f}")
            
            # Reset
                running = {k: 0.0 for k in running}

            global_step += 1

        # val_nll_masked, val_ppl_masked = eval_masked_nll(model, val_loader, device)

        # wandb.log({
        #     "val/token_nll_masked": val_nll_masked,
        #     "val/masked_ppl": val_ppl_masked,
        #     "epoch": epoch
        # })


        # print(f"Epoch {epoch} done in {time.time()-t0:.1f}s | "
        #       f"val masked NLL: {val_nll_masked:.3f} (PPL {val_ppl_masked:.2f}) | ")

        step_probs = step_counts.float() / step_counts.sum()
        print(f"Step sampling distribution (should be ~uniform):")
        print(f"  Min: {step_probs.min().item():.4f}, Max: {step_probs.max().item():.4f}, "
            f"Mean: {step_probs.mean().item():.4f}")
        wandb.log({
            "train/step_sampling_min": step_probs.min().item(),
            "train/step_sampling_max": step_probs.max().item(),
            "train/step_sampling_mean": step_probs.mean().item(),  
        })

    print("Training complete.")
    wandb.finish()

if __name__ == "__main__":
    # Default config dict (edit here or update from a JSON/YAML loader)
    CONFIG = {
        "wandb_project": "order-ar-openwebtext",
        "model_name": "gpt2",
        "block_size": 512,           # try 1024 if you have VRAM
        "batch_size": 2,
        "eval_batch_size": 2,
        "epochs": 3,
        "lr": 3e-4,
        "mask_prob": [1, 0.15,0.15,0.15, 0.2, 0.5, 0.15, 0.15, 0.7, 0.9, 1],
        "seed": 42,
        "eval_stride": 32,
        "eval_pppl_batches": 25,
        "wandb_mode": "online",      # or "offline"
        # model dims
        "d_model": 384,
        "n_layers": 6,
        "n_heads": 6,
        "num_ar_steps": 10,        # number of autoregressive unmasking steps per training iteration
        "cache_dir": "./data_cache",
        "variational_policy": "shared_torso"  # or "separate_heads"
    }
    # Example: quick overrides
    # CONFIG.update({"batch_size": 16, "lr": 2e-4})
    main(CONFIG)