"""
Evaluation script for LoMDM.

Compute test perplexity (ELBO bound) on various datasets.

Usage:
    python evaluate.py --checkpoint outputs/checkpoint-final.pt --dataset lm1b
    python evaluate.py --checkpoint outputs/checkpoint-final.pt --dataset openwebtext
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from lomdm import LoMDM, LoMDMConfig
from lomdm.diffusion import sample_forward_process
from lomdm.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoMDM")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    
    # Data
    parser.add_argument("--dataset", type=str, default="lm1b",
                        choices=["lm1b", "openwebtext", "ptb", "wikitext", 
                                 "lambada", "ag_news", "pubmed", "arxiv"])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    
    # Evaluation
    parser.add_argument("--num_time_samples", type=int, default=100,
                        help="Number of time samples for ELBO estimation")
    parser.add_argument("--use_learned_scheduler", action="store_true", default=True)
    parser.add_argument("--no_learned_scheduler", dest="use_learned_scheduler",
                        action="store_false")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config = LoMDMConfig()
    
    model = LoMDM(config).to(device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, config


def get_eval_dataset(dataset_name: str, tokenizer, max_length: int):
    """Load evaluation dataset."""
    
    dataset_configs = {
        "lm1b": ("lm1b", "test", "text"),
        "openwebtext": ("openwebtext", "train", "text"),  # No official test split
        "ptb": ("ptb_text_only", "test", "sentence"),
        "wikitext": ("wikitext", "test", "text"),
        "lambada": ("lambada", "test", "text"),
        "ag_news": ("ag_news", "test", "text"),
    }
    
    if dataset_name in dataset_configs:
        name, split, text_col = dataset_configs[dataset_name]
        try:
            if name == "wikitext":
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            elif name == "openwebtext":
                # Use last portion as "test"
                full_dataset = load_dataset(name, split="train")
                dataset = full_dataset.select(range(len(full_dataset) - 10000, len(full_dataset)))
            else:
                dataset = load_dataset(name, split=split)
        except Exception as e:
            print(f"Error loading {name}: {e}")
            print("Using LM1B test set instead")
            dataset = load_dataset("lm1b", split="test")
            text_col = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def tokenize_function(examples):
        tokens = tokenizer(
            examples[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return {"input_ids": tokens["input_ids"]}
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4,
    )
    tokenized.set_format(type="torch", columns=["input_ids"])
    
    return tokenized


def compute_elbo_bound(
    model: LoMDM,
    dataloader: DataLoader,
    device: torch.device,
    num_time_samples: int = 100,
    use_learned_scheduler: bool = True,
) -> dict:
    """
    Compute ELBO (negative log-likelihood upper bound).
    
    The ELBO for LoMDM is:
    E_t E_qα [Σ_i <z_t^i, m> * A_φ * log p(x^i | z_t)]
    
    We estimate this via Monte Carlo over time and positions.
    """
    
    model.eval()
    
    total_elbo = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing ELBO"):
            x = batch["input_ids"].to(device)
            B, L = x.shape
            
            batch_elbo = 0.0
            
            # Monte Carlo over time
            for _ in range(num_time_samples):
                t = torch.rand(B, device=device)
                
                # Get scheduler values
                if use_learned_scheduler:
                    features = model.get_backbone_features(x, stop_gradient=True)
                    alpha, velocity = model.forward_scheduler(features, t)
                else:
                    # Standard MDLM schedule: α = 1 - t
                    alpha = 1.0 - t.unsqueeze(-1).expand(-1, L)
                    velocity = 1.0 / t.unsqueeze(-1).clamp(min=1e-6).expand(-1, L)
                
                # Sample masked sequence
                z_t, mask = sample_forward_process(x, alpha, model.mask_token_id)
                
                # Get model predictions
                logits = model(z_t)
                # SUBS zero-masking: set mask token logit to -inf before softmax
                logits[:, :, model.mask_token_id] = float('-inf')
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get log prob of correct tokens
                target_log_probs = torch.gather(
                    log_probs, -1, x.unsqueeze(-1)
                ).squeeze(-1)
                
                # Weighted by velocity and mask
                # ELBO contribution: A * log p(x|z_t) for masked positions
                weighted_logprob = velocity * target_log_probs * mask.float()
                
                batch_elbo += weighted_logprob.sum().item()
            
            batch_elbo /= num_time_samples
            total_elbo += batch_elbo
            total_tokens += B * L
    
    # ELBO is negative, so perplexity = exp(-ELBO / num_tokens)
    avg_elbo = total_elbo / total_tokens
    
    # The ELBO gives us a bound on NLL
    # NLL <= -ELBO (approximately, depends on exact formulation)
    nll_bound = -avg_elbo
    ppl_bound = np.exp(nll_bound)
    
    return {
        "elbo": avg_elbo,
        "nll_bound": nll_bound,
        "ppl_bound": ppl_bound,
    }


def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataset = get_eval_dataset(args.dataset, tokenizer, args.max_length)
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: {"input_ids": torch.stack([b["input_ids"] for b in batch])},
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Compute ELBO
    results = compute_elbo_bound(
        model, dataloader, device,
        num_time_samples=args.num_time_samples,
        use_learned_scheduler=args.use_learned_scheduler,
    )
    
    print("\n=== Results ===")
    print(f"Dataset: {args.dataset}")
    print(f"ELBO: {results['elbo']:.4f}")
    print(f"NLL bound: {results['nll_bound']:.4f}")
    print(f"PPL bound: {results['ppl_bound']:.2f}")
    
    # Compare with standard MDLM schedule if using learned scheduler
    if args.use_learned_scheduler:
        print("\n--- Comparison with standard MDLM schedule ---")
        results_mdlm = compute_elbo_bound(
            model, dataloader, device,
            num_time_samples=args.num_time_samples,
            use_learned_scheduler=False,
        )
        print(f"MDLM PPL bound: {results_mdlm['ppl_bound']:.2f}")
        print(f"Improvement: {(results_mdlm['ppl_bound'] - results['ppl_bound']):.2f}")


if __name__ == "__main__":
    main()
