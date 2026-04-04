"""
PUMA training script for LM1B with SentencePiece tokenizer.

Usage (single GPU):
    python train_lm1b.py --cfg yaml_files/lm1b_puma.yaml --data_dir /path/to/preprocessed_lm1b

Usage (multi-GPU with torchrun):
    torchrun --nproc_per_node=4 train_lm1b.py --cfg yaml_files/lm1b_puma.yaml --data_dir /path/to/preprocessed_lm1b

The --data_dir flag overrides data.data_dir in the YAML config.
"""

import math, os, time, json, random, sys, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import torch.distributed as dist
import argparse
from copy import deepcopy
from tqdm import tqdm
from model.transformer import MDMTransformer, MDMConfig
from data_lm1b import LM1BDataset, setup_lm1b_loaders
from torch.utils.data import DataLoader, random_split
from typing import Optional, List, Tuple, Union
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf, DictConfig, ListConfig
from model.ema import ExponentialMovingAverage, save_ema_snapshot, save_model_snapshot
from progressive import PhasedMasking, mdm_loss_fn
from sampling import mdm_sampling


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to preprocessed LM1B data directory")
    return parser.parse_args()


# ---------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------

def setup_ddp():
    if torch.cuda.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.norm(p=2).item()
    return total ** 0.5


# ---------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------

def mdm_loss(model, input_ids, mask_id: int, prompt_mask: Optional[torch.Tensor] = None):
    """Standard MDLM loss: random masking + reweighted cross-entropy."""
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    device = input_ids.device
    B, L = input_ids.shape
    L_eff = L - prompt_mask.sum(dim=1, keepdim=True)

    num_mask = torch.floor(torch.rand(B, 1, device=device) * L_eff.clamp(min=1)).long() + 1

    scores = torch.rand((B, L), device=device).masked_fill(prompt_mask, float('inf')).argsort(dim=1)
    order = scores.argsort(dim=1)
    mask_indices = (order < num_mask)
    masked_input = torch.where(mask_indices, mask_id, input_ids)
    logits = model(masked_input)

    num_mask = num_mask.float().expand_as(mask_indices)
    ce = F.cross_entropy(logits[mask_indices], input_ids[mask_indices], reduction="none")
    loss = ce / num_mask[mask_indices]
    return loss.sum() / B


# ---------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------

def val_loss_ddp(model, val_loader, mask_id: int, device, rank: int, world_size: int, strategy: str):
    model.eval()
    if world_size > 1 and dist.is_initialized() and not isinstance(val_loader.sampler, DistributedSampler):
        sampler = DistributedSampler(val_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=val_loader.batch_size or 16,
            sampler=sampler,
            num_workers=4,
            pin_memory=False,
            drop_last=False,
        )

    local_sum = 0.0
    local_count = 0
    c = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", disable=(rank != 0)):
            if c > 100:
                break
            x0 = batch["labels"].to(device)
            pm = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                loss = mdm_loss(model, x0, mask_id, prompt_mask=pm)
            B = x0.shape[0]
            local_sum += float(loss.item() * B)
            local_count += B
            c += 1

    tensor = torch.tensor([local_sum, local_count], dtype=torch.float, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_sum, global_count = tensor.tolist()
    return global_sum / max(int(global_count), 1)


# ---------------------------------------------------------------
# Validation sampling (perplexity-free quality check)
# ---------------------------------------------------------------

def evaluate_lm1b_sampling(model, val_loader, mask_id, sampling_cfg, device, rank, world_size, max_batches=10):
    """
    Generate a few samples from fully-masked validation sequences and
    return the fraction of non-mask tokens produced (sanity check).
    """
    model.eval()
    total_tokens = 0
    filled_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            x0 = batch["labels"].to(device)
            B, L = x0.shape
            xt = torch.full_like(x0, mask_id)  # fully masked
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = mdm_sampling(model, xt, mask_id, sampling_cfg)
            total_tokens += B * L
            filled_tokens += (out != mask_id).sum().item()
    fill_rate = filled_tokens / max(total_tokens, 1)
    return fill_rate


# ---------------------------------------------------------------
# K-schedule parsing
# ---------------------------------------------------------------

def parse_k_schedule_increasing(k_schedule) -> List[Tuple[int, int]]:
    if k_schedule is None:
        return []
    sched = []
    prev_step = None
    for item in list(k_schedule):
        if not isinstance(item, (list, tuple, ListConfig)) or len(item) != 2:
            raise ValueError(f"k_schedule entries must be [K, step], got {item}")
        K, step = int(item[0]), int(item[1])
        if K <= 0:
            raise ValueError(f"Invalid K in k_schedule: {K}")
        if step < 0:
            raise ValueError(f"Invalid step in k_schedule: {step}")
        if prev_step is not None and step <= prev_step:
            raise ValueError(f"k_schedule must have strictly increasing steps, but got step {step} after {prev_step}")
        sched.append((K, step))
        prev_step = step
    return sched


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main(cfg: DictConfig, data_dir: str):
    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)
    if is_main:
        print("Starting LM1B PUMA training!")
        print(f"Training with {world_size} GPU(s)")

    base_seed = 2026
    seed = base_seed + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # Checkpoint directory
    ckpt_dir = f"ckpts_lm1b/date={datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    os.makedirs(ckpt_dir, exist_ok=True)
    if is_main:
        print(f"Checkpoints: {ckpt_dir}")

    # Device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ---- Data (LM1B with SentencePiece) ----
    data_cfg = cfg.data
    train_cfg = cfg.training
    val_cfg = cfg.validation
    assert train_cfg.save_steps % train_cfg.eval_steps == 0, "save_steps must be divisible by eval_steps"

    # Load meta to get vocab_size
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    sp_vocab_size = meta["vocab_size"]

    # mask_id = vocab_size (one beyond last valid SentencePiece token)
    mask_id = sp_vocab_size
    if is_main:
        print(f"SentencePiece vocab_size={sp_vocab_size}, mask_id={mask_id}, max_len={meta['max_len']}")

    # Override model config with actual vocab
    model_cfg_dict = dict(cfg.model)
    model_cfg_dict["vocab_size"] = sp_vocab_size + 1  # +1 for mask token
    model_cfg_dict["max_position"] = meta["max_len"]

    # Build loaders
    val_ratio = getattr(data_cfg, "val_ratio", 0.02)
    data_seed = getattr(data_cfg, "seed", 2026)
    train_loader, val_loader = setup_lm1b_loaders(
        data_dir,
        batch_size=train_cfg.batch_size,
        val_ratio=val_ratio,
        seed=data_seed,
    )
    if is_main:
        print(f"Train: {len(train_loader.dataset)} examples, Val: {len(val_loader.dataset)} examples")

    # ---- Model ----
    model_config = MDMConfig(**model_cfg_dict)
    model = MDMTransformer(model_config).to(device)

    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params / 1e6:.2f}M")

    # DDP wrapping
    if world_size > 1 and torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            print("DDP wrapping done")

    # DDP sampler for training
    if world_size > 1 and torch.cuda.is_available():
        train_sampler = DistributedSampler(train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_cfg.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=False,
            drop_last=True,
        )
    else:
        train_sampler = None

    # ---- Optimizer & Scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    num_training_steps = train_cfg.num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_cfg.warmup_steps, num_training_steps=num_training_steps)

    # EMA
    if train_cfg.ema is not None:
        assert 0.0 < train_cfg.ema < 1.0
        model_to_ema = model.module if isinstance(model, DDP) else model
        ema_params = [p for p in model_to_ema.parameters() if p.requires_grad]
        ema = ExponentialMovingAverage(ema_params, decay=train_cfg.ema)
        if is_main:
            print(f"EMA enabled, decay={train_cfg.ema}")

    strategy = train_cfg.strategy

    # ---- Progressive setup ----
    if strategy == "progressive":
        k_schedule = parse_k_schedule_increasing(getattr(train_cfg, "k_schedule", None))
        if len(k_schedule) == 0:
            k_schedule = [(train_cfg.K, 0)]
        current_k = k_schedule[0][0]

        if is_main:
            print("K Schedule:")
            for K, step in k_schedule:
                print(f"  Step {step}: K={K}")

        def make_pool(K):
            return PhasedMasking(
                train_loader, train_cfg.batch_size, mask_id, K, device, model_config.max_position,
                mode=train_cfg.mode,
                confidence_threshold=train_cfg.confidence_threshold,
                eos_id=getattr(train_cfg, "eos_id", None),
            )
        pool = make_pool(current_k)
        next_k_idx = 1

    # ---- Training loop ----
    global_step = 0
    accum_steps = getattr(train_cfg, "grad_accum_steps", 1)
    last_ema_ckpt_path = None
    last_model_ckpt_path = None

    # Wandb
    if cfg.wandb.wandb and is_main:
        wandb.init(project=cfg.wandb.project, entity=getattr(cfg.wandb, "entity", None), name=cfg.wandb.name)

    from contextlib import nullcontext

    for epoch in range(train_cfg.num_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if strategy == "progressive":
            pool.reset_loader_iter()
            steps_per_epoch = len(train_loader)
            iterable = range(steps_per_epoch)
        elif strategy == "standard":
            iterable = train_loader

        if is_main:
            pbar = tqdm(iterable, desc=f"Epoch {epoch + 1}")
        else:
            pbar = iterable

        optimizer.zero_grad()
        accum_loss = 0.0
        micro_step = 0

        use_ddp = isinstance(model, DDP)

        def maybe_no_sync():
            if use_ddp and (micro_step + 1) % accum_steps != 0:
                return model.no_sync()
            return nullcontext()

        for itr in pbar:
            # K schedule update
            cur_opt_step = global_step // accum_steps
            if strategy == "progressive" and next_k_idx < len(k_schedule) and cur_opt_step == k_schedule[next_k_idx - 1][1]:
                current_k = k_schedule[next_k_idx][0]
                if is_main:
                    print(f"[K-SWITCH] Opt step {cur_opt_step}: K={current_k}")
                pool = make_pool(current_k)
                pool.reset_loader_iter()
                next_k_idx += 1

            with maybe_no_sync():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    if strategy == "progressive":
                        xt = pool.current_batch()
                        logits = model(xt)
                        log_probs = F.log_softmax(logits, dim=-1)
                        loss = mdm_loss_fn(log_probs, pool.x0, pool.xt, mask_id, prompt_mask=pool.state['prompt_mask'])
                    elif strategy == "standard":
                        batch = itr
                        input_ids = batch["labels"].to(device)
                        prompt_mask = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None
                        loss = mdm_loss(model, input_ids, mask_id, prompt_mask=prompt_mask)
                    else:
                        raise ValueError(f"Invalid training strategy: {strategy}")

                scaled_loss = loss / accum_steps
                scaled_loss.backward()
            accum_loss += loss.item()

            # Update pool every micro-step
            if strategy == "progressive":
                with torch.no_grad():
                    pool.update_with_logits(log_probs)

            global_step += 1
            micro_step += 1

            if micro_step % accum_steps == 0:
                if train_cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                optimizer.step()

                if train_cfg.ema is not None:
                    ema.update(ema_params)
                scheduler.step()

                opt_step = global_step // accum_steps
                if is_main:
                    avg_loss = accum_loss / accum_steps
                    pbar.set_postfix(loss=avg_loss, lr=optimizer.param_groups[0]["lr"], opt_step=opt_step)

                    if opt_step % train_cfg.logging_steps == 0:
                        print(f"Epoch {epoch + 1}, Step {opt_step}, Loss {avg_loss:.4f}")
                        if cfg.wandb.wandb:
                            wandb.log({"loss": avg_loss}, step=opt_step)
                            gn = grad_norm(model.parameters())
                            wandb.log({"grad_norm": gn}, step=opt_step)
                            if strategy == "progressive":
                                wandb.log({"current_k": current_k}, step=opt_step)

                optimizer.zero_grad()
                accum_loss = 0.0

            # ---- Evaluation & Checkpointing ----
            opt_step = global_step // accum_steps
            if global_step % accum_steps == 0 and opt_step % train_cfg.eval_steps == 0 and opt_step > 0:
                model.eval()

                # Validation loss
                val_l = val_loss_ddp(model, val_loader, mask_id, device, rank, world_size, strategy)

                # EMA evaluation
                if train_cfg.ema is not None:
                    torch.cuda.empty_cache()
                    model_to_ema = model.module if isinstance(model, DDP) else model
                    ema.store(model_to_ema.parameters())
                    ema.copy_to(model_to_ema.parameters())

                    with torch.inference_mode():
                        ema_val_l = val_loss_ddp(model, val_loader, mask_id, device, rank, world_size, strategy)
                    ema.restore(model_to_ema.parameters())
                else:
                    ema_val_l = None

                if is_main:
                    print(f"Epoch {epoch + 1}, Step {opt_step}, Val Loss: {val_l:.4f}")
                    if ema_val_l is not None:
                        print(f"  EMA Val Loss: {ema_val_l:.4f}")
                    if cfg.wandb.wandb:
                        wandb.log({"val_loss": val_l}, step=opt_step)
                        if ema_val_l is not None:
                            wandb.log({"ema_val_loss": ema_val_l}, step=opt_step)

                    if opt_step % train_cfg.save_steps == 0:
                        if train_cfg.ema is not None:
                            saved = save_ema_snapshot(ckpt_dir, model, ema, cfg, epoch, opt_step, val_l)
                            if saved:
                                if last_ema_ckpt_path and os.path.exists(last_ema_ckpt_path):
                                    os.remove(last_ema_ckpt_path)
                                last_ema_ckpt_path = saved
                                print(f"EMA checkpoint: {saved}")

                        saved = save_model_snapshot(ckpt_dir, model, cfg, epoch, opt_step, val_loss=val_l)
                        if saved:
                            if last_model_ckpt_path and os.path.exists(last_model_ckpt_path):
                                os.remove(last_model_ckpt_path)
                            last_model_ckpt_path = saved
                            print(f"Model checkpoint: {saved}")

                model.train()

    if cfg.wandb.wandb and is_main:
        wandb.finish()
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    main(cfg, args.data_dir)
