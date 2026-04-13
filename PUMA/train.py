import math, os, time, json, random, sys, datetime
from upm import UPM
from upm import compute_reward_pair, plackett_luce_log_prob, sample_plackett_luce, compute_reward_pair_old
#from torch.cuda.nvtx import range as nv_range
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
from data import setup_data_bundle
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from typing import Optional, List, Tuple, Union
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf, DictConfig, ListConfig
from model.ema import ExponentialMovingAverage, save_ema_snapshot, save_model_snapshot
from progressive import PhasedMasking, mdm_loss_fn
from eval.sudoku_eval import evaluate_ddp_sudoku
from eval.gsm8k_eval import evaluate_ddp_gsm8k
from sampling import mdm_sampling
from transformers import AutoTokenizer
#to = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    return parser.parse_args()


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

def evaluate_ddp_dict(model, cfg, device, rank, world_size, upm = None):
    sampling = cfg.validation.sampling
    if cfg.training.strategy == "arm":
        return {"arm": evaluate_ddp(model, cfg, device, rank, world_size, sampling, upm)}
    base_sampling = sampling
    out = {}

    for confidence in list(base_sampling.confidence):
        for unmasking_num in list(base_sampling.unmasking_num):
            sampling = deepcopy(base_sampling)
            sampling.confidence = confidence
            sampling.unmasking_num = unmasking_num
            out[f"{confidence}_unmasking_{unmasking_num}"] = evaluate_ddp(model, cfg, device, rank, world_size, sampling, upm)
    return out

def grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.norm(p=2).item()
    return total ** 0.5

def evaluate_ddp(model, cfg, device, rank: int, world_size: int, sampling, upm=None):
    if cfg.data.dataset == "sudoku":
        return evaluate_ddp_sudoku(model, cfg, device, rank, world_size, sampling)
    elif cfg.data.dataset == "tinygsm":
        return evaluate_ddp_gsm8k(model, cfg, device, rank, world_size, sampling, upm)
    elif cfg.data.dataset == "lm1b":
        return 0.0
    else:
        raise ValueError(f"Invalid dataset: {cfg.data.dataset}")

# mdm loss implementation
def mdm_loss(model, input_ids, mask_id: int, prompt_mask: Optional[torch.Tensor] = None, arm_init: bool = False):
    # sample integer uniformly for each batch from [1,L]
    # prompt_mask (boolean mask): 1 for prompt
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    device = input_ids.device
    B, L = input_ids.shape
    L_eff = L - prompt_mask.sum(dim=1 , keepdim=True)
    # uniformly sample the number of positions to mask
    num_mask = torch.floor(torch.rand(B, 1, device=device) * L_eff.clamp(min=1)).long() + 1

    # mask correspondent number of tokens for each batch, 0.0 for the prompt indices
    scores = torch.rand((B, L), device=device).masked_fill(prompt_mask, float('inf')).argsort(dim=1)
    order = scores.argsort(dim=1)
    mask_indices = (order < num_mask)
    masked_input = torch.where(mask_indices, mask_id, input_ids)
    logits = model(masked_input)

    # calculate (reweighted) loss
    num_mask = num_mask.float().expand_as(mask_indices)

    if arm_init:
        ce = F.cross_entropy(logits[:, :-1, :][mask_indices[:, 1:]], input_ids[:, 1:][mask_indices[:, 1:]], reduction="none")
    else:
        ce = F.cross_entropy(logits[mask_indices], input_ids[mask_indices], reduction="none")
    loss = ce / num_mask[mask_indices]
    return loss.sum() / B

def log_linear_alpha(t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """α(t) = exp(-t · log(1/ε)),  so α(0)=1, α(1)≈ε."""
    return torch.exp(-t * math.log(1.0 / eps))

def mdm_loss_loglinear(model, input_ids, mask_id: int, prompt_mask: Optional[torch.Tensor] = None, arm_init: bool = False, eps = 1e-3):
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(input_ids, dtype = torch.bool)
    device = input_ids.device
    B, L = input_ids.shape

    t = torch.rand(B, device = device) * (1 - eps) + eps 
    alpha_t = log_linear_alpha(t)
    mask_prob = ( 1 - alpha_t)

    rand = torch.rand(B, L, device = device)
    mask_indices = (rand < mask_prob.unsqueeze(1)) & ~prompt_mask

    no_mask = ~mask_indices.any(dim=1)
    if no_mask.any():
        # force-mask one random non-prompt position
        force = torch.rand(B, L, device=device).masked_fill(prompt_mask, 2.0)
        pos = force.argmin(dim=1)                                 # (B,)
        mask_indices[no_mask, pos[no_mask]] = True
    masked_input = torch.where(mask_indices, mask_id, input_ids)

    logits = model(masked_input)
    nll = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        input_ids.view(-1),
        reduction="none",
    ).view(B, L)

    if arm_init: 
        pass
    else:
        nll_masked = (nll * mask_indices).sum(dim = 1)
        n_masked = mask_indices.sum(dim=1).clamp_min(1).float()
    # R(t) = β · α(t) / (1 - α(t)),  where β = log(1/ε)
    beta = math.log(1.0 / eps)
    weight = beta * alpha_t / (1.0 - alpha_t)                     # (B,)
 
    # per-sequence: weight(t) * mean-NLL-over-masked  (mean keeps magnitude stable)
    per_seq = weight * (nll_masked / n_masked)
    return per_seq.mean()
    

def arm_loss(
    model,
    input_ids: torch.Tensor,                    # (B, L)
    eos_id: int,
    prompt_mask: Optional[torch.Tensor] = None, # True = prompt token
):
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    logits = model(input_ids)          # (B, L, V)
    targets = input_ids[:, 1:]         # (B, L-1)
    pred_logits = logits[:, :-1, :]    # (B, L-1, V)

    valid = ~prompt_mask[:, 1:]        # (B, L-1)

    if eos_id is not None:
        is_eos = (targets == eos_id)               # (B, L-1)
    else:
        is_eos = torch.zeros_like(targets, dtype=torch.bool)
    any_eos = is_eos.any(dim=1)                # (B,)
    first_eos = is_eos.float().argmax(dim=1)   # (B,) 0-based in targets
    first_eos = torch.where(
        any_eos,
        first_eos,
        torch.full_like(first_eos, targets.shape[1] - 1),
    )

    t = torch.arange(targets.shape[1], device=targets.device).unsqueeze(0)  # (1, L-1)
    valid = valid & (t <= first_eos.unsqueeze(1))

    if valid.sum().item() == 0:
        return pred_logits.sum() * 0.0
    return F.cross_entropy(pred_logits[valid], targets[valid], reduction="mean")

# validation loss helper
def val_loss_ddp(model, val_loader, mask_id: int, device, rank: int, world_size: int, strategy: str, eos_id: int, arm_init: bool = False):
    model.eval()
    # upm.eval()
    if world_size > 1 and dist.is_initialized() and not isinstance(val_loader.sampler, DistributedSampler):
        sampler = DistributedSampler(val_loader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(
        val_loader.dataset,
        batch_size=val_loader.batch_size or 16,
        sampler=sampler,
        #num_workers=getattr(val_loader, "num_workers", 4),
        num_workers = 0,
        pin_memory=getattr(val_loader, "pin_memory", False),
        drop_last=False,
        )
    else:
        sampler = None

    local_sum = 0.0
    local_count = 0
    c = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc = "Validating", disable = (rank != 0)):
            if c > 500:
                break;
            x0 = batch["labels"].to(device)
            pm = batch["prompt_mask"].to(device) if "prompt_mask" in batch else None
            
            # to enable flashattention, we do autocast
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
                if strategy == "arm":
                    loss = arm_loss(model, x0, eos_id=eos_id, prompt_mask=pm)
                elif strategy in ["progressive", "standard"]:
                    loss = mdm_loss(model, x0, mask_id, prompt_mask = pm, arm_init=arm_init)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
            B = x0.shape[0]
            local_sum += float(loss.item() * B)
            local_count += B
            c+=1
    
    tensor = torch.tensor([local_sum, local_count], dtype=torch.float, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_sum, global_count = tensor.tolist()

    return global_sum / max(int(global_count), 1)

def parse_k_schedule_increasing(k_schedule) -> List[Tuple[int, int]]:
    """
    Expects k_schedule as an *already increasing* list of [K, step] pairs.
    Validates:
      - each entry is [K, step]
      - steps are strictly increasing
      - (optionally) first step is 0
    Returns list of (K, step) in the same order.
    """
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
            raise ValueError(
                f"k_schedule must have strictly increasing steps, but got step {step} after {prev_step}. "
                f"Full schedule: {list(k_schedule)}"
            )

        sched.append((K, step))
        prev_step = step

    return sched



def main(cfg: DictConfig):
    # setup the DDP
    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)
    if is_main:
        print("Hey, we start training!")
        print(f"Training with {world_size} GPUs")
    
    base_seed = 2026
    seed = base_seed + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    # ckpt dir
    ckpt_dir = f"/home/aakarsh/ckptscompdeb/date={datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    os.makedirs(ckpt_dir, exist_ok=True)
    if is_main:
        print(f"Checkpoints will be saved to: {ckpt_dir}")

    # set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Initialize the model
    model_cfg_dict = cfg.model
    
    if cfg.data.dataset == "lm1b":
        meta_path = os.path.join(cfg.data.data_dir, "meta.json")
        with open(meta_path, "r") as fh:
            _meta = json.load(fh)
        # Force the model to build with enough tokens to encapsulate BOTH padding token (vocab_size) and mask token (vocab_size + 1)
        model_cfg_dict.vocab_size = _meta["vocab_size"] + 2
        model_cfg_dict.max_position = _meta["max_len"]

    model_config = MDMConfig(**model_cfg_dict)
    model = MDMTransformer(model_config).to(device)
    condition_dim = model_config.hidden_size  # must be even; each half = 128
    upm = UPM(
        hidden_size=model_config.hidden_size,
        condition_dim=condition_dim,
        num_heads=8,
    ).to(device)

    if is_main:
        num_params = sum(p.numel() for p in upm.parameters())
        print(f"UPM parameters: {num_params/1e6:.2f}M")

    # ARM initialization
    arm_init_path = model_cfg_dict.get("arm_init", "none")
    if arm_init_path != "none":
        model_config.predict_next_token = True
        if is_main:
            print(f"Initializing MDM from ARM checkpoint: {arm_init_path}")
        arm_ckpt = torch.load(arm_init_path, map_location="cpu")
        sd = arm_ckpt.get("model_state_dict", arm_ckpt)
        model.load_state_dict(sd, strict=True)


    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model is ready, parameters: {num_params/1e6:.2f}M")

    # model wrapping
    if world_size > 1 and torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        upm = DDP(upm, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            print(f"Model wrapping is done!")

    all_params = (
        list(model.parameters())
        + list(upm.parameters())
    )
    


    # data
    data_cfg = cfg.data
    train_cfg = cfg.training
    assert train_cfg.save_steps % train_cfg.eval_steps == 0, "save_steps must be divisible by eval_steps"
    val_cfg = cfg.validation
    if data_cfg.dataset == "lm1b":
        from data_lm1b import setup_lm1b_loaders
        
        # Load vocab stats to dynamically set mask_id
        meta_path = os.path.join(data_cfg.data_dir, "meta.json")
        with open(meta_path) as fh:
            meta = json.load(fh)
        
        # Guarantee mask_id is isolated from the pad_token_id (which rests precisely at vocab_size in Qwen2).
        data_cfg.mask_id = meta["vocab_size"] + 1
        
        train_loader, val_loader = setup_lm1b_loaders(
            data_cfg.data_dir,
            batch_size=train_cfg.batch_size,
            val_ratio=getattr(data_cfg, "val_ratio", 0.02),
            seed=getattr(data_cfg, "seed", 2026),
        )
    else:
        # Standard fallback for tinygsm, sudoku, etc.
        data_bundle = setup_data_bundle(data_cfg)
        train_loader, val_loader = data_bundle.train_loader, data_bundle.val_loader    
    mask_id = data_cfg.mask_id
    eos_id = getattr(val_cfg.sampling, "eos_id", None)

    # training hyperparemeters
    # attach DDP sampler
    if world_size > 1 and torch.cuda.is_available():
        train_sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_cfg.batch_size,
            sampler=train_sampler,
            # changed this in sohpia
            num_workers=2,
            pin_memory=False,
            drop_last=False
        )
    else:
        train_sampler = None

    # optimizer and scheduler
    optimizer = optim.AdamW(
        all_params,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    #optimizer = optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    num_training_steps = train_cfg.num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_cfg.warmup_steps, num_training_steps=num_training_steps)
    if train_cfg.ema is not None:
        assert 0.0 < train_cfg.ema < 1.0, "EMA decay must be between 0 and 1"
        model_to_ema = model.module if isinstance(model, DDP) else model
        ema_params = [p for p in model_to_ema.parameters() if p.requires_grad]
        ema = ExponentialMovingAverage(ema_params, decay=train_cfg.ema)
        if is_main:
            print("EMA is enabled with decay:", train_cfg.ema)

    strategy = train_cfg.strategy
    # k schedule for progressive unmasking. If None use fixed K. If "linear", linearly increase the unmasking steps from 1 to K over the training steps.
    # If a list of integers, use the list as the k_steps. If an integer, use constant interval increase.
    if strategy == "progressive":
        k_schedule = parse_k_schedule_increasing(getattr(train_cfg, "k_schedule", None))
        if len(k_schedule) == 0:
            k_schedule = [(train_cfg.K, 0)]
        
        current_k = k_schedule[0][0]

        if is_main:
            print("Using K Schedule:")
            for K, step in k_schedule:
                print(f"Step {step}: K={K}")

        # intialize the pool
        B = train_cfg.batch_size
        L = model_config.max_position
        def make_pool(K):
            B = train_cfg.batch_size
            L = model_config.max_position
            return PhasedMasking(
                train_loader, B, mask_id, K, device, L,
                mode=train_cfg.mode,
                confidence_threshold=train_cfg.confidence_threshold,
                eos_id=train_cfg.eos_id,
            )
        pool = make_pool(current_k)
        next_k_idx = 1


    # training loop
    global_step = 0
    accum_steps = getattr(train_cfg, "grad_accum_steps", 1)
    last_ema_ckpt_path = None
    last_model_ckpt_path = None


    # wandb initialize
    if cfg.wandb.wandb and is_main:
        wandb.init(project=cfg.wandb.project, entity = "aakarshnrai-purdue-university",name=cfg.wandb.name)
    from contextlib import nullcontext, ExitStack
    for epoch in range(train_cfg.num_epochs):
        model.train()
        upm.train()
        upm_module = upm.module if isinstance(upm, DDP) else upm
        model_module = model.module if isinstance(model, DDP) else model

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if strategy == "progressive":
            pool.reset_loader_iter()
            steps_per_epoch = len(train_loader)
            iterable = range(steps_per_epoch)
        elif strategy == "standard" or strategy == "arm":
            iterable = train_loader

        if is_main:
            pbar = tqdm(iterable, desc=f"Epoch {epoch+1}")
        else:
            pbar = iterable

        optimizer.zero_grad()
        accum_loss = 0.0
        accum_mdm_loss = 0.0
        accum_upm_loss = 0.0
        accum_adv_mag = 0.0
        accum_r_mag = 0.0
        accum_lp_diff = 0.0
        micro_step = 0

        # helper: skip DDP gradient sync on non-final micro-steps
        use_ddp = isinstance(model, DDP)
        def maybe_no_sync():
            stack = ExitStack()
            if use_ddp and (micro_step + 1) % accum_steps != 0:
                stack.enter_context(model.no_sync())
                if isinstance(upm, DDP):
                    stack.enter_context(upm.no_sync())
            return stack

        for itr in pbar:
            # update current K if using k schedule (based on optimizer steps, not micro-steps)
            cur_opt_step = global_step // accum_steps
            if strategy == "progressive" and next_k_idx < len(k_schedule) and cur_opt_step == k_schedule[next_k_idx - 1][1]:
                current_k = k_schedule[next_k_idx][0]
                if is_main:
                    print(f"[K-SWITCH] Opt step {cur_opt_step}: K={current_k}")

                pool = make_pool(current_k)
                pool.reset_loader_iter()
                next_k_idx += 1

            # to enable flashattention, we do the autocast
            # with torch.autograd.profiler.emit_nvtx():
            with maybe_no_sync():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
                    if strategy == "progressive":
                        xt = pool.current_batch()
                        # with nv_range("forward"):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits, hidden = model(xt, return_hidden=True)
                        # logits = model(xt)
                        
                        log_probs = F.log_softmax(logits, dim=-1)
                        mdm_l = mdm_loss_fn(log_probs, pool.x0, pool.xt, mask_id, prompt_mask = pool.state['prompt_mask'], arm_init=model_config.predict_next_token)
                        mask_idx = xt == mask_id
                        t_step = pool.state['phase'].float() / pool.K  
                        #mask_emb = mask_indicator_embedding(mask_idx.long())
                        upm_scores_raw = upm(hidden.detach(), t_step, mask_idx).float()         # (B, L), finite
                        # Blend UPM scores with log(max_prob) as confidence proxy.
                        # UPM learns a *correction* on top of confidence, not a replacement.
                        # Must match eval-time blending for train/eval consistency.
                        # log_prob_max = max_logit - logsumexp: exact, no extra (B,L,V) memory.
                        #log_conf = (logits.max(dim=-1).values - logits.logsumexp(dim=-1)).detach()  # (B, L)



                        log_conf = (logits.max(dim=-1).values - logits.logsumexp(dim=-1)).detach()  # (B, L)
                        blended_scores = upm_scores_raw + log_conf
                        #blended = upm_scores_raw

                        # --- THE FIX: Add Temperature Scaling for RL Exploration ---
                        tau = 2.0  # Adjust this: higher = more exploration, lower = deterministic
                        blended_scores = blended_scores / tau

                        upm_scores = blended_scores.masked_fill(~mask_idx, float('-inf'))  # for sampling
                        #blended_scores = upm_scores_raw + log_conf
                        #upm_scores = blended_scores.masked_fill(~mask_idx, float('-inf'))  # for sampling


                        # --- RLOO ---
                        mean_L_eff = pool.state['L_eff'].float().mean()
                        k_unmask = max(int(mean_L_eff / pool.K), 1)
                        #k_unmask = 4
                        n_masked_per_seq = mask_idx.sum(dim=1)
                        valid = n_masked_per_seq >= k_unmask

                        if valid.any():
                            U1, _ = sample_plackett_luce(upm_scores, mask_idx, k_unmask)
                            U2, _ = sample_plackett_luce(upm_scores, mask_idx, k_unmask)
                            
                            safe_indices = torch.arange(xt.shape[1], device=xt.device).unsqueeze(0).expand(xt.shape[0], -1)[:, :k_unmask]
                            U1 = torch.where(valid.unsqueeze(1), U1, safe_indices)
                            U2 = torch.where(valid.unsqueeze(1), U2, safe_indices)
                            
                            xt_after_1 = xt.clone()
                            xt_after_2 = xt.clone()
                            B_actual = xt.shape[0]
                            batch_indices = torch.arange(B_actual, device=xt.device).unsqueeze(1).expand(-1, k_unmask)
                            xt_after_1[batch_indices, U1] = pool.x0[batch_indices, U1]
                            xt_after_2[batch_indices, U2] = pool.x0[batch_indices, U2]
                            
                            r1, r2 = compute_reward_pair(
                                model_module, pool.x0, logits.detach(), xt_after_1, xt_after_2,
                                mask_id, pool.state['prompt_mask']
                            )

                            A1 = r1 - r2
                            A2 = r2 - r1
                            A1 = torch.where(valid, A1, torch.zeros_like(A1))
                            A2 = torch.where(valid, A2, torch.zeros_like(A2))

                            lp1 = plackett_luce_log_prob(upm_scores, U1, mask_idx)
                            lp2 = plackett_luce_log_prob(upm_scores, U2, mask_idx)
                            # Zero out lp for invalid samples to prevent inf*0=NaN
                            lp1 = torch.where(valid, lp1, torch.zeros_like(lp1))
                            lp2 = torch.where(valid, lp2, torch.zeros_like(lp2))
                            
                            per_sample = -(A1 * lp1 + A2 * lp2)
                            per_sample = per_sample * valid.float()
                            upm_loss = per_sample.sum() / valid.sum().clamp_min(1).float()

                            # diagnostics (detached, for logging only)
                            _adv_mag = A1[valid].abs().mean().item() if valid.any() else 0.0
                            _r_mag = ((r1[valid].abs() + r2[valid].abs()) / 2).mean().item() if valid.any() else 0.0
                            _lp_diff = (lp1[valid] - lp2[valid]).abs().mean().item() if valid.any() else 0.0
                        else:
                            # Connect to UPM graph so DDP sees the same params on every rank
                            upm_loss = upm_scores_raw.sum() * 0.0
                            _adv_mag = 0.0
                            _r_mag = 0.0
                            _lp_diff = 0.0

                        lambda_upm = 1.0
                        loss = mdm_l + lambda_upm * upm_loss

                    elif strategy == "standard":
                        batch = itr
                        input_ids = batch["labels"].to(device)
                        prompt_mask = batch["prompt_mask"].to(device) if "prompt_mask" in batch else (input_ids == 151643).to(device)
                        # loss = mdm_loss(model, input_ids, mask_id, prompt_mask = prompt_mask, arm_init=model_config.predict_next_token)
                        loss = mdm_loss_loglinear(model, input_ids, mask_id, prompt_mask = prompt_mask, arm_init=model_config.predict_next_token)
                    elif strategy == "arm":
                        batch = itr
                        input_ids = batch["labels"].to(device)
                        prompt_mask = batch["prompt_mask"].to(device) if "prompt_mask" in batch else (input_ids == 151643).to(device)
                        loss = arm_loss(model, input_ids, eos_id=eos_id, prompt_mask=prompt_mask)
                    else:
                        raise ValueError(f"Invalid training strategy: {strategy}")

                scaled_loss = loss / accum_steps
                # with nv_range("backward"):
                scaled_loss.backward()
            accum_loss += loss.item()
            if strategy == "progressive":
                accum_mdm_loss += mdm_l.item()
                accum_upm_loss += upm_loss.item() if isinstance(upm_loss, torch.Tensor) else upm_loss
                accum_adv_mag += _adv_mag
                accum_r_mag += _r_mag
                accum_lp_diff += _lp_diff

            # update pool every micro-step (chain advances regardless of accumulation)
            if strategy == "progressive":
                #pool.update_with_logits(log_probs)
                pool.update_with_upm_scores(upm_scores.detach(), log_probs.detach())


            global_step += 1
            micro_step += 1

            if micro_step % accum_steps == 0:
                if train_cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(upm.parameters()),
                        train_cfg.max_grad_norm
                    )
                optimizer.step()
                
                if train_cfg.ema is not None:
                    ema.update(ema_params)
                scheduler.step()

                opt_step = global_step // accum_steps
                should_log = (opt_step % train_cfg.logging_steps == 0)
                if is_main:
                    avg_loss = accum_loss / accum_steps
                    pbar.set_postfix(loss=avg_loss, lr=optimizer.param_groups[0]["lr"], opt_step=opt_step)

                    if opt_step % train_cfg.logging_steps == 0:
                        print(f"Epoch {epoch+1}, Step {opt_step}, Loss {avg_loss}")
                        if cfg.wandb.wandb:
                            wandb.log({"loss": avg_loss}, step=opt_step)

                            gn = grad_norm(model.parameters())
                            wandb.log({"grad_norm": gn}, step=opt_step)

                            if strategy == "progressive":
                                wandb.log({"current_k": current_k}, step=opt_step)

                # DDP-reduced mdm / upm losses (all ranks must participate)
                if strategy == "progressive" and should_log:
                    avg_mdm = torch.tensor(accum_mdm_loss / accum_steps, device=device)
                    avg_upm = torch.tensor(accum_upm_loss / accum_steps, device=device)
                    dist.all_reduce(avg_mdm, op=dist.ReduceOp.AVG)
                    dist.all_reduce(avg_upm, op=dist.ReduceOp.AVG)
                    if is_main and cfg.wandb.wandb:
                        wandb.log({
                            "mdm_loss": avg_mdm.item(),
                            "upm_loss": avg_upm.item(),
                            "upm/adv_mag": accum_adv_mag / accum_steps,
                            "upm/reward_mag": accum_r_mag / accum_steps,
                            "upm/lp_diff": accum_lp_diff / accum_steps,
                        }, step=opt_step)
                optimizer.zero_grad()
                accum_loss = 0.0
                accum_mdm_loss = 0.0
                accum_upm_loss = 0.0
                accum_adv_mag = 0.0
                accum_r_mag = 0.0
                accum_lp_diff = 0.0

            opt_step = global_step // accum_steps
            if global_step % accum_steps == 0 and opt_step % train_cfg.eval_steps == 0 and opt_step > 0:
                model.eval()
                upm.eval()

                # validaton on the downstream task; disabled when we use EMA
                if train_cfg.ema is None:
                    val_acc_dict = evaluate_ddp_dict(model_module, cfg, device, rank, world_size, upm=upm_module)
                else:
                    val_acc_dict = None

                # validation loss (mdm loss on the validation dataset)
                val_loss = val_loss_ddp(model, val_loader, mask_id, device, rank, world_size, strategy, eos_id, arm_init=model_config.predict_next_token)

                # EMA evaluation
                if train_cfg.ema is not None:
                    torch.cuda.empty_cache()
                    model_to_ema = model.module if isinstance(model, DDP) else model
                    ema.store(model_to_ema.parameters())
                    ema.copy_to(model_to_ema.parameters())

                    with torch.inference_mode():
                        # validaton on the downstream task
                        val_acc_dict = evaluate_ddp_dict(model_module, cfg, device, rank, world_size, upm=upm_module)
                    ema.restore(model_to_ema.parameters())

                if is_main:
                    # eval acc logging
                    for key, value in val_acc_dict.items():
                        print(f"Epoch {epoch+1}, Step {opt_step}, Validation Accuracy {key}: {value}")
                        if cfg.wandb.wandb:
                            if train_cfg.ema is not None:
                                wandb.log({"ema_val_acc_" + key: value}, step=opt_step)
                            else:
                                wandb.log({"val_acc_" + key: value}, step=opt_step)

                    # validation loss logging
                    print(f"Epoch {epoch+1}, Step {opt_step}, Validation Loss: {val_loss}")
                    if cfg.wandb.wandb:
                        wandb.log({"val_loss": val_loss}, step=opt_step)

                    if opt_step % train_cfg.save_steps == 0 and train_cfg.ema is not None:
                        saved_path = save_ema_snapshot(ckpt_dir, model, ema, cfg, epoch, opt_step, val_loss, val_acc_dict)
                        if saved_path is not None:
                            if last_ema_ckpt_path and os.path.exists(last_ema_ckpt_path):
                                os.remove(last_ema_ckpt_path)
                            last_ema_ckpt_path = saved_path
                            print(f"EMA Model saved to: {saved_path}")
#                        saved_path = save_ema_snapshot(ckpt_dir, model, ema, cfg, epoch, opt_step, val_loss, val_acc_dict)
#                        if saved_path is not None:
#                            print(f"EMA Model saved to: {saved_path}")

                    if opt_step % train_cfg.save_steps == 0:
                        # save non-EMA snapshot
                        saved_path = save_model_snapshot(
                            ckpt_dir, model, cfg, epoch, opt_step,
                            val_loss=val_loss,
                            extra=val_acc_dict,
                        )
                        if saved_path is not None:
                            if last_model_ckpt_path and os.path.exists(last_model_ckpt_path):
                                os.remove(last_model_ckpt_path)
                            last_model_ckpt_path = saved_path
                            print(f"Model saved to: {saved_path}")
                        #if saved_path is not None:
                            #print(f"Model saved to: {saved_path}")

                model.train()
                upm.train()
    
    if cfg.wandb.wandb and is_main:
        wandb.finish()
    
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.cfg
    cfg = OmegaConf.load(cfg_path)
    main(cfg)