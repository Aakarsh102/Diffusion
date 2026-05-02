"""
Supervised UPM pre-training via confidence distillation.

Usage:
    python supervised_upm_train.py \
        --config yaml_files/tinygsm_puma.yaml \
        --mdm_ckpt ~/ckpts/ema_step=400000.pt \
        --output_dir ~/ckpts/upm_supervised \
        --num_steps 5000 \
        --lr 1e-4 \
        --tau 1.0 \
        --batch_size 64
"""
import wandb
import os
import sys
import math
import json
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from model.transformer import MDMTransformer, MDMConfig
from upm import UPM
from data import setup_data_bundle


def load_mdm_from_checkpoint(ckpt_path, device, is_main=True, config_override=None):
    """Load a pre-trained MDM from checkpoint. Handles both EMA and regular snapshots."""
    if is_main:
        print(f"Loading MDM checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Get config from checkpoint or override
    if config_override is not None:
        model_cfg_dict = config_override
    else:
        cfg = ckpt["config"]
        model_cfg_dict = cfg["model"]

    model_config = MDMConfig(**model_cfg_dict)
    model = MDMTransformer(model_config)

    # Load weights (handles both EMA and regular)
    sd = ckpt.get("model_state_dict", ckpt)
    # Strip 'module.' prefix if saved from DDP
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    is_ema = ckpt.get("is_ema_snapshot", False)
    step = ckpt.get("global_step", "unknown")
    if is_main:
        print(f"  Loaded {'EMA' if is_ema else 'regular'} snapshot from step {step}")
        print(f"  Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M (frozen)")

    return model, model_config


def random_mask_batch(x0, prompt_mask, mask_id):
    """
    Randomly mask each sequence at a uniformly sampled ratio.
    Returns xt, t_step (fraction unmasked), mask_idx (bool: True = masked).
    """
    B, L = x0.shape
    device = x0.device

    # Effective length per sequence (non-prompt tokens)
    L_eff = (~prompt_mask).sum(dim=1).float().clamp_min(1)  # (B,)

    # Sample masking ratio uniformly: fraction of non-prompt tokens to MASK
    # We want t_step = fraction UNMASKED, so mask_ratio = 1 - t_step
    # Sample t_step ∈ (0, 1) to ensure at least some tokens are masked
    t_step = torch.rand(B, device=device) * 0.95 + 0.025  # avoid extremes

    num_unmask = (t_step * L_eff).long().clamp(min=0)  # how many to reveal

    # Build xt: start fully masked (non-prompt), then reveal num_unmask tokens randomly
    xt = torch.where(prompt_mask, x0, torch.full_like(x0, mask_id))

    # For each sequence, unmask a random subset of non-prompt positions
    rand_scores = torch.rand(B, L, device=device)
    rand_scores = rand_scores.masked_fill(prompt_mask, -1.0)  # don't pick prompt positions
    # Sort by score descending; top num_unmask[i] positions get unmasked
    _, sorted_idx = rand_scores.sort(dim=1, descending=True)
    # Create position ranks
    ranks = torch.zeros_like(sorted_idx)
    ranks.scatter_(1, sorted_idx, torch.arange(L, device=device).unsqueeze(0).expand(B, L))

    unmask_positions = ranks < num_unmask.unsqueeze(1)
    xt = torch.where(unmask_positions & ~prompt_mask, x0, xt)

    mask_idx = (xt == mask_id)
    return xt, t_step, mask_idx


def supervised_upm_loss(model, upm, x0, prompt_mask, mask_id, tau=1.0):
    """
    Compute KL-divergence loss: train UPM to match MDM confidence ranking.
    
    1. Randomly mask the batch at various ratios
    2. Forward through frozen MDM to get logits + hidden states
    3. Compute confidence = log(max_prob) over masked positions
    4. Train UPM scores to match confidence distribution via KL div
    """
    B, L = x0.shape
    device = x0.device

    # Step 1: Random masking
    xt, t_step, mask_idx = random_mask_batch(x0, prompt_mask, mask_id)

    # Skip if no masked tokens in the entire batch
    if mask_idx.sum() == 0:
        return upm.score_head.weight.sum() * 0.0, {}

    # Step 2: Frozen MDM forward
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits, hidden = model(xt, return_hidden=True)

    # Step 3: MDM confidence as target distribution (over masked positions per sequence)
    # log_conf = log(max_prob) = max_logit - logsumexp
    # you can just DO LOG_CONF = LOGUSUMEXP(LOGITS - MAX_LOGIT)
    log_conf = (logits.max(dim=-1).values - logits.logsumexp(dim=-1)).float()  # (B, L)

    # Step 4: UPM forward (with gradients)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        upm_scores = upm(hidden.detach(), t_step, mask_idx).float()  # (B, L)

    # Step 5: KL divergence over masked positions
    # Set non-masked positions to -inf so softmax ignores them
    target_logits = log_conf.masked_fill(~mask_idx, float('-inf'))  # (B, L)
    pred_logits = upm_scores.masked_fill(~mask_idx, float('-inf'))  # (B, L)

    # Softmax to get distributions over masked positions
    target_dist = F.softmax(target_logits / tau, dim=-1)  # (B, L)
    pred_log_dist = F.log_softmax(pred_logits / tau, dim=-1)  # (B, L)

    # KL divergence: sum over positions, mean over batch
    # Only masked positions contribute (others are 0 * -inf handled by softmax)
    kl = F.kl_div(pred_log_dist, target_dist, reduction='none')  # (B, L)
    # Mask out non-masked positions to avoid NaN from 0 * log(0)
    kl = kl.masked_fill(~mask_idx, 0.0)
    loss = kl.sum(dim=-1).mean()  # mean over batch

    # Diagnostics
    with torch.no_grad():
        # Rank correlation: do UPM and confidence agree on ordering?
        n_agree = 0
        n_total = 0
        for i in range(min(B, 8)):  # sample a few sequences
            m = mask_idx[i]
            if m.sum() < 2:
                continue
            conf_vals = log_conf[i][m]
            upm_vals = upm_scores[i][m]
            # Spearman-like: fraction of pairs where ordering matches
            conf_ranks = conf_vals.argsort().argsort().float()
            upm_ranks = upm_vals.argsort().argsort().float()
            n = conf_ranks.shape[0]
            n_pairs = n * (n - 1) / 2
            if n_pairs > 0:
                concordant = 0
                for a in range(n):
                    for b in range(a + 1, min(a + 10, n)):  # sample pairs for speed
                        if (conf_ranks[a] - conf_ranks[b]) * (upm_ranks[a] - upm_ranks[b]) > 0:
                            concordant += 1
                        n_total += 1
                n_agree += concordant

        rank_corr = n_agree / max(n_total, 1)

    diagnostics = {
        "loss": loss.item(),
        "rank_correlation": rank_corr,
        "avg_masked_per_seq": mask_idx.float().sum(dim=1).mean().item(),
        "avg_t_step": t_step.mean().item(),
    }
    return loss, diagnostics


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device=torch.device(f"cuda:{local_rank}")
    is_main = rank == 0
    parser = argparse.ArgumentParser(description="Supervised UPM pre-training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--mdm_ckpt", type=str, required=True, help="Path to pre-trained MDM checkpoint")
    parser.add_argument("--output_dir", type=str, default="./upm_supervised_ckpts", help="Where to save UPM")
    parser.add_argument("--num_steps", type=int, default=20000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for UPM")
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature for KL target (lower = peakier)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--log_every", type=int, default=50, help="Log every N steps")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N steps")
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)
    
    if is_main:
        print(f"Device: {device}")
        if cfg.wandb.get("wandb", False):
            wandb.init(
                project=cfg.wandb.get("project", "mdm-pretraining"),
                entity=cfg.wandb.get("entity", None),
                name=cfg.wandb.get("name", "puma") + "-supervised_upm",
                config=vars(args)
            )

    # Load frozen MDM
    model, model_config = load_mdm_from_checkpoint(args.mdm_ckpt, device, is_main=is_main, config_override=dict(cfg.model))

    # Data
    data_cfg = cfg.data
    mask_id = data_cfg.mask_id

    if data_cfg.dataset == "lm1b":
        from data_lm1b import setup_lm1b_loaders
        meta_path = os.path.join(data_cfg.data_dir, "meta.json")
        with open(meta_path) as fh:
            meta = json.load(fh)
        mask_id = meta["vocab_size"] + 1
        train_loader, _ = setup_lm1b_loaders(
            data_cfg.data_dir,
            batch_size=args.batch_size or cfg.training.batch_size,
            val_ratio=getattr(data_cfg, "val_ratio", 0.02),
            seed=getattr(data_cfg, "seed", 2026),
        )
    else:
        data_bundle = setup_data_bundle(data_cfg)
        train_loader = data_bundle.train_loader
        train_samples = DistributedSampler(train_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = DataLoader(
            train_loader.dataset, 
            batch_size=args.batch_size or cfg.training.batch_size,
            sampler=train_samples,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

    if is_main:
        print(f"Dataset: {data_cfg.dataset}, mask_id: {mask_id}")
        print(f"Training batches available: {len(train_loader)}")

    # Initialize fresh UPM
    condition_dim = model_config.hidden_size
    upm = UPM(
        hidden_size=model_config.hidden_size,
        condition_dim=condition_dim,
        num_heads=8,
    ).to(device)
    upm.load_state_dict(torch.load("ckpts/upm_supervised/upm_step=3000.pt", map_location=device, weights_only=True))
    upm = DDP(upm, device_ids=[local_rank], output_device=local_rank)
    num_params = sum(p.numel() for p in upm.parameters())
    print(f"UPM parameters: {num_params / 1e6:.2f}M")

    # Optimizer
    optimizer = optim.AdamW(upm.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps, eta_min=1e-6)

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    upm.train()
    data_iter = iter(train_loader)
    
    if is_main:
        pbar = tqdm(range(1, args.num_steps + 1), desc="Supervised UPM")
    else:
        pbar = range(1, args.num_steps + 1)
        
    running_loss = 0.0
    running_corr = 0.0

    for step in pbar:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x0 = batch["labels"].to(device)
        prompt_mask = batch["prompt_mask"].to(device) if "prompt_mask" in batch else torch.zeros_like(x0, dtype=torch.bool)

        # Compute loss
        loss, diag = supervised_upm_loss(model, upm, x0, prompt_mask, mask_id, tau=args.tau)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(upm.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        running_loss += diag["loss"]
        running_corr += diag["rank_correlation"]

        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            avg_corr = running_corr / args.log_every
            lr = optimizer.param_groups[0]["lr"]
            if is_main:
                pbar.set_postfix(loss=f"{avg_loss:.4f}", rank_corr=f"{avg_corr:.3f}", lr=f"{lr:.2e}")
                print(f"  Step {step}: loss={avg_loss:.4f}, rank_corr={avg_corr:.3f}, "
                      f"masked/seq={diag['avg_masked_per_seq']:.1f}, t_step={diag['avg_t_step']:.3f}")
                
                if cfg.wandb.get("wandb", True):
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/rank_correlation": avg_corr,
                        "train/lr": lr,
                        "diagnostics/avg_masked_per_seq": diag['avg_masked_per_seq'],
                        "diagnostics/avg_t_step": diag['avg_t_step'],
                    }, step=step)
                    
            running_loss = 0.0
            running_corr = 0.0

        # Save checkpoint
        if is_main and (step % args.save_every == 0 or step == args.num_steps):
            save_path = os.path.join(args.output_dir, f"upm_step={step}.pt")
            torch.save({
                "step": step,
                "upm_state_dict": upm.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "hidden_size": model_config.hidden_size,
                    "condition_dim": condition_dim,
                    "num_heads": 8,
                    "tau": args.tau,
                    "lr": args.lr,
                },
            }, save_path)
            print(f"  Saved UPM checkpoint to: {save_path}")

    print("\nDone! Final UPM saved to:", os.path.join(args.output_dir, f"upm_step={args.num_steps}.pt"))
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
