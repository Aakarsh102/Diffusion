"""
Training script for LoMDM.

This script trains the Learnable-Order Masked Diffusion Model following
Algorithm 1 from the paper.

Usage:
    python train.py --dataset lm1b --batch_size 256 --max_steps 1000000
    python train.py --dataset openwebtext --batch_size 32 --max_steps 1000000
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import wandb
from typing import Dict

from lomdm import LoMDM, LoMDMConfig
from lomdm.utils import (
    set_seed,
    get_cosine_schedule_with_warmup,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    get_grad_norm,
    GradScaler,
)
from lomdm.diffusion import sample_forward_process
from lomdm.sampling import LoMDMSampler
from torch.optim.lr_scheduler import LambdaLR

import re

def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r" \'(\w+)", r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    return x

def get_constant_schedule_with_warmup(optimizer, num_warmup_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        return 1.0
    return LambdaLR(optimizer, lr_lambda)

def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
def cleanup():
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoMDM")
    
    # Data
    parser.add_argument("--dataset", type=str, default="lm1b",
                        choices=["lm1b", "openwebtext", "text8"])
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    
    # Model
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--c1", type=float, default=0.7)
    parser.add_argument("--c2", type=float, default=0.65)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--warmup_steps", type=int, default=2500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # Mixed precision
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    
    # Logging
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="comparision")
    
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="lomdm")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    
    return parser.parse_args()


def get_dataset(args, tokenizer):
    """Load and preprocess dataset."""
    
    if args.dataset == "lm1b":
        dataset = load_dataset("lm1b",cache_dir="/lus/eagle/projects/lighthouse-purdue/rai53/files/lm1b", split="train")
        text_column = "text"
    elif args.dataset == "openwebtext":
        dataset = load_dataset("openwebtext", cache_dir="/lus/eagle/projects/lighthouse-purdue/rai53/files/openwebtext", split="train")
        text_column = "text"
    elif args.dataset == "text8":
        # Text8 requires special handling
        dataset = load_dataset("text8", split="train")
        text_column = "text"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    eos_id = tokenizer.sep_token_id or tokenizer.eos_token_id

    #def tokenize_function(examples):
    #    texts = [lm1b_detokenizer(t) for t in examples[text_column]]
    #    tokens = tokenizer(examples[text_column], truncation=False)
    #    return {"input_ids": tokens["input_ids"]}

    #tokenized_dataset = dataset.map(
    #    tokenize_function,
    #    batched=True,
    #    remove_columns=dataset.column_names,
    #    num_proc=8,
    #    load_from_cache_file=True,
    #)
    def tokenize_function(examples):
        texts = [lm1b_detokenizer(t) for t in examples[text_column]]
        tokens = tokenizer(texts, truncation=False)
        return {"input_ids": tokens["input_ids"]}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=8,
        load_from_cache_file=True,  # force recompute
    )

    # Concatenate all documents with [EOS] between them, then chunk
    def group_texts(examples):
        all_ids = []
        for doc in examples["input_ids"]:
            all_ids.extend(doc)
            if eos_id is not None:
                all_ids.append(eos_id)
        total_length = (len(all_ids) // args.max_length) * args.max_length
        result = {
            "input_ids": [all_ids[i : i + args.max_length]
                          for i in range(0, total_length, args.max_length)]
        }
        return result

    packed_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=8)
    packed_dataset.set_format(type="torch", columns=["input_ids"])

    return packed_dataset


def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    return {"input_ids": input_ids}


def reduce_metrics(metrics_dict, device):
    """All-reduce a dict of scalar metrics across all DDP ranks, returning the mean."""
    keys = sorted(metrics_dict.keys())
    vals = torch.tensor([metrics_dict[k] for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    vals /= dist.get_world_size()
    return {k: v.item() for k, v in zip(keys, vals)}


def unwrap_model(m):
    return getattr(m, "module", m)


#def warmup_training(model, batch, optimizer, scaler, grad_clip=1.0, device="cuda"): 
def warmup_step(model, batch, optimizer, scaler, step, grad_clip = 1.0, device = "cuda"):
    model.train()
    optimizer.zero_grad()

    x = batch["input_ids"].to(device)
    B, L = x.shape
    t = torch.rand(B, device=device)
    if (step <= 10000):
        t = t/2.0
    t = torch.clamp(t, min = 0.02)
#    masked_input = torch.full((B, L), 0.0, device=device, dtype=torch.float)
    keep_mask = torch.rand(B, L, device=device) < t.unsqueeze(1)
    masked_input = x.clone()
    masked_input[keep_mask] = unwrap_model(model).mask_token_id
    if (not keep_mask.any()):
        return {"warmup/warmup_loss": 500}
    with torch.autocast(enabled=True, device_type="cuda"):
        logits = model(masked_input)
        B, L, V = logits.shape
        logits_flat = logits.view(B * L, V)
        targets_flat = x.view(B * L)
        loss = torch.nn.functional.cross_entropy(logits_flat[keep_mask.view(B*L)], targets_flat[keep_mask.view(B * L)])

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = get_grad_norm(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return {"warmup/warmup_loss": loss.item(), "warmup/grad_norm":grad_norm}


def train_step(
    model: LoMDM,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    grad_clip: float,
    device: torch.device,
    autocast_dtype
) -> Dict[str, float]:
    """Perform one training step."""
    
    model.train()
    optimizer.zero_grad()
    
    # Get input
    x = batch["input_ids"].to(device)
    B, L = x.shape
    device = x.device
    
    # Sample time uniformly
    sampling_eps = 1e-3
    t = torch.rand(B, device=device)
    offset = torch.arange(B, device=device).float() / B
    t = (t / B + offset) % 1
    t = (1 - sampling_eps) * t + sampling_eps
    
    # Forward pass with mixed precision
    with torch.autocast(enabled=scaler.enabled, dtype=autocast_dtype, device_type = "cuda"):
        loss, loss_dict = unwrap_model(model).training_step(x, t)
    
    # Backward pass
    scaler.scale(loss).backward()
    
    # Gradient clipping
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    
    # Get grad norm for logging
    grad_norm = get_grad_norm(model)
    
    # Return metrics
    metrics = {
        "loss": loss.item(),
        "main_loss": loss_dict["main_loss"].item(),
        "velocity_loss": loss_dict["velocity_loss"].item(),
        "rloo_loss": loss_dict["rloo_loss"].item(),
        "mask_ratio": loss_dict["mask_ratio"].item(),
        "grad_norm": grad_norm,
    }
    
    return metrics


@torch.no_grad()
def evaluate(
    model: LoMDM,
    eval_dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 100,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()

    loss_sum = main_sum = vel_sum = 0.0
    count = 0

    for i, batch in enumerate(eval_dataloader):
        if i >= num_batches:
            break

        x = batch["input_ids"].to(device)
        B, L = x.shape
        t = torch.rand(B, device=device)

        clean_features = model.get_backbone_features(x, stop_gradient=True)
        alpha_phi, velocity_phi = model.forward_scheduler(clean_features, t)
        z_t, mask = sample_forward_process(x, alpha_phi, model.mask_token_id)

        logits = model(z_t)
        features_z = model.get_backbone_features(z_t, stop_gradient=True)
        _, velocity_psi = model.reverse_scheduler(features_z, t)

        from lomdm.losses import compute_lomdm_loss
        _, loss_dict = compute_lomdm_loss(
            logits, x, velocity_phi, velocity_psi, mask, model.mask_token_id
        )

        loss_sum += loss_dict["loss"].item() * B
        main_sum += loss_dict["main_loss"].item() * B
        vel_sum += loss_dict["velocity_loss"].item() * B
        count += B

    return {
        "eval_loss": loss_sum / count,
        "eval_main_loss": main_sum / count,
        "eval_velocity_loss": vel_sum / count,
    }


def main():
    args = parse_args()
    if args.fp16:
        autocast_dtype = torch.float16
    elif args.bf16:
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = None
    

    
    # Set seed
    set_seed(args.seed)
    setup()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = (not dist.is_initialized()) or rank == 0

    # Device — use local_rank so multi-node works
    device = torch.device("cuda", local_rank)
    print(f"[rank {rank}] Using device: {device}")
    
    # Initialize wandb
    if is_main and (not args.no_wandb):
        wandb.init(
            project="LO_Training",
            entity="aakarshnrai-purdue-university",
            name=args.wandb_run_name or f"lomdm-{args.dataset}-scaling-alcf-final-training1028",
            #id="jj600lae",
            #resume="allow",
            config=vars(args),
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Add mask token if not present
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    mask_token_id = tokenizer.mask_token_id
    vocab_size = len(tokenizer)
    
    print(f"Vocab size: {vocab_size}, Mask token ID: {mask_token_id}")
    
    # Create config
    config = LoMDMConfig(
        vocab_size=vocab_size,
        max_seq_length=args.max_length,
        mask_token_id=mask_token_id,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
  #      scheduler_num_heads=args.num_heads,
        c1=args.c1,
        c2=args.c2,
    )
    
    # Create model
    model = LoMDM(config).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    print(f"Model parameters: {count_parameters(model):,}")
    param_breakdown = unwrap_model(model).get_num_params()
    print(f"  Backbone: {param_breakdown['backbone']:,}")
    print(f"  Forward scheduler: {param_breakdown['forward_scheduler']:,}")
    print(f"  Reverse scheduler: {param_breakdown['reverse_scheduler']:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        #model.parameters(),
        [
            {"params": unwrap_model(model).backbone.parameters(), "lr": 3e-4},
            {"params": unwrap_model(model).forward_scheduler.parameters(), "lr":1e-5},
            {"params": unwrap_model(model).reverse_scheduler.parameters(), "lr":1e-5},
        ],
        #lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    
    # Create scheduler
    #scheduler = get_cosine_schedule_with_warmup(
    #    optimizer,
    #    num_warmup_steps=args.warmup_steps,
    #    num_training_steps=args.max_steps,
    #)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=2500)
    
    # Create gradient scaler
    scaler = GradScaler(enabled=args.bf16 and torch.cuda.is_available())
    
    # Load dataset
    print("Loading dataset...")
    if is_main:
        dataset = get_dataset(args, tokenizer)  # rank 0: computes + writes cache
    dist.barrier()                               # everyone waits
    if not is_main:
        dataset = get_dataset(args, tokenizer)  # ranks 1+: .map() finds cache, loads it
    #dataset = get_dataset(args, tokenizer)
    sampler = DistributedSampler(dataset, shuffle = True)
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # For evaluation, use a subset
    eval_dataloader = DataLoader(
        dataset.select(range(min(10000, len(dataset)))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    # Resume from checkpoint
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_step, _ = load_checkpoint(
            args.resume, unwrap_model(model), optimizer, scheduler, device
        )
        #for pg in optimizer.param_group():
            #o
        # Change weight decay for all param groups
        #for pg in optimizer.param_groups:
        #    pg['weight_decay'] = 0.005

        #scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)
# Or change LR if needed
# optimizer.param_groups[0]['lr'] = 3e-4  # backbone
# optimizer.param_groups[1]['lr'] = 1e-5  # forward scheduler
# optimizer.param_groups[2]['lr'] = 1e-5  # reverse scheduler
        # Optionally broadcast model params from rank 0 to others, in case only rank0 loaded
        if dist.is_initialized():
            for param in unwrap_model(model).state_dict().values():
                dist.broadcast(param, src=0)

    
    
    # Training loop
    print("Starting training...")
    step = start_step
    
    metrics_sum = {"loss": 0.0, "main_loss": 0.0, "velocity_loss": 0.0, "rloo_loss": 0.0}
    metrics_count = 0
    
    pbar = tqdm(total=args.max_steps - start_step, initial=start_step)
    ff = len(train_dataloader)
    while step < args.max_steps:
        sampler.set_epoch(step // ff)
        for batch in train_dataloader:
            if step >= args.max_steps:
                break
            if (step > 10000000):
                metrics = warmup_step(model, batch, optimizer, scaler, step = step, device=device)
                if (metrics["warmup/warmup_loss"] == 500):
                    continue;
                step += 1
                scheduler.step()
                pbar.update(1)
                if step % 100 == 0:
                    # Reduce warmup metrics across ranks before logging
                    reduced_warmup = reduce_metrics(metrics, device)
                    if is_main and not args.no_wandb:
                        wandb.log(reduced_warmup, step)
                continue
            
            # Train step
            metrics = train_step(
                model, batch, optimizer, scaler, args.grad_clip, device, autocast_dtype
            )
            
            # Update scheduler
            scheduler.step()
            
            # Update metrics
            for k in metrics_sum:
                if k in metrics:
                    metrics_sum[k] += metrics[k]
            metrics_count += 1

            step += 1
            pbar.update(1)

            # Logging — all-reduce metrics across ranks so we log the true mean
            #print(torch.cuda.memory_allocated()/1e9)
            if step % args.log_every == 0:
                log_metrics = {
                    k: v / metrics_count for k, v in metrics_sum.items()
                }
                log_metrics["grad_norm"] = metrics["grad_norm"]
                log_metrics["mask_ratio"] = metrics["mask_ratio"]

                # All-reduce across ranks to get the global mean
                log_metrics = reduce_metrics(log_metrics, device)
                # LR is the same on all ranks, no need to reduce
                log_metrics["lr"] = scheduler.get_last_lr()[0]

                if is_main:
                    pbar.set_postfix(loss=f'{log_metrics["loss"]:.4f}')
                    if is_main:
                        log_metrics["vram_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                        log_metrics["vram_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
                        log_metrics["vram_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9

                    if not args.no_wandb:
                        wandb.log({"train/" + k: v for k, v in log_metrics.items()}, step=step)

                # Reset accumulators on ALL ranks
                for k in metrics_sum:
                    metrics_sum[k] = 0.0
                metrics_count = 0
            
            # Evaluation
            if step % args.eval_every == 0:
                if is_main:
                    print(f"\nEvaluating at step {step}...")
                eval_metrics = evaluate(unwrap_model(model), eval_dataloader, device)

                # All-reduce eval metrics across ranks
                eval_metrics = reduce_metrics(eval_metrics, device)

                if is_main:
                    print(f"Eval loss: {eval_metrics['eval_loss']:.4f}")
                    if not args.no_wandb:
                        wandb.log({"eval/" + k: v for k, v in eval_metrics.items()}, step=step)
            
            # Generate samples (rank 0 only, no reduce needed)
            if is_main and step % 1000 == 0:
                model.eval()
                gen_sampler = LoMDMSampler(unwrap_model(model), mask_token_id=unwrap_model(model).mask_token_id)
                output = gen_sampler.sample_ddpm_cache(
                    batch_size=4,
                    seq_length=args.max_length,
                    num_steps=256,
                    temperature=1.0,
                    use_learned_scheduler=True,
                    device=device,
                )
                texts = tokenizer.batch_decode(output.samples, skip_special_tokens=True)
                print(f"\n--- Samples at step {step} ---")
                for j, text in enumerate(texts):
                    print(f"[{j}] {text}")
                print("---\n")
                if not args.no_wandb:
                    wandb.log({"samples": wandb.Table(columns=["id", "text"], data=[[j, t] for j, t in enumerate(texts)])}, step=step)
                model.train()

            # Save checkpoint
            if is_main and step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}.pt")
                print(f"\nSaving checkpoint to {save_path}")
                save_checkpoint(
                    unwrap_model(model), optimizer, scheduler, step,
                    metrics_sum["loss"] / max(metrics_count, 1), save_path
                )
            dist.barrier()
    
    pbar.close()
    
    
    # Final save
    if is_main:
        save_path = os.path.join(args.output_dir, "checkpoint-final.pt")
        print(f"[rank {rank}] Saving final checkpoint to {save_path}")
        save_checkpoint(unwrap_model(model), optimizer, scheduler, step, 0, save_path)
    cleanup()

    
    print("Training complete!")
    
    if is_main and (not args.no_wandb):
        wandb.finish()


if __name__ == "__main__":
    main()