"""
Sampling script for LoMDM.

Generate text samples using trained LoMDM models.

Usage:
    python sample.py --checkpoint outputs/checkpoint-final.pt --num_samples 64
    python sample.py --checkpoint outputs/checkpoint-final.pt --semi_ar --total_length 2048
"""

import os
import argparse
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from lomdm import LoMDM, LoMDMConfig
from lomdm.sampling import LoMDMSampler, compute_perplexity
from lomdm.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from LoMDM")
    
    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    
    # Sampling
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    # Sampling method
    parser.add_argument("--method", type=str, default="ddpm_cache",
                        choices=["ddpm", "ddpm_cache", "confidence"])
    parser.add_argument("--use_learned_scheduler", action="store_true", default=True)
    parser.add_argument("--no_learned_scheduler", dest="use_learned_scheduler",
                        action="store_false")
    
    # Semi-autoregressive
    parser.add_argument("--semi_ar", action="store_true")
    parser.add_argument("--total_length", type=int, default=2048)
    parser.add_argument("--block_length", type=int, default=1024)
    
    # Output
    parser.add_argument("--output_file", type=str, default="samples.txt")
    parser.add_argument("--compute_ppl", action="store_true",
                        help="Compute generative perplexity with GPT-2")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to extract config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        # Use default config (you might need to adjust this)
        config = LoMDMConfig()
    
    model = LoMDM(config).to(device)
    
    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Assume checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, config


def compute_generative_perplexity(samples: torch.Tensor, tokenizer) -> float:
    """
    Compute generative perplexity using GPT-2 as evaluator.
    
    This measures how "natural" the generated text is according to GPT-2.
    """
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print("GPT-2 not available, skipping perplexity computation")
        return float("nan")
    
    # Load GPT-2
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    gpt2_model.eval()
    
    device = samples.device
    gpt2_model = gpt2_model.to(device)
    
    # Decode samples to text
    texts = tokenizer.batch_decode(samples, skip_special_tokens=True)
    
    # Compute perplexity
    total_nll = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing PPL"):
            # Re-tokenize with GPT-2
            inputs = gpt2_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024
            ).to(device)
            
            if inputs["input_ids"].shape[1] < 2:
                continue
            
            outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
            nll = outputs.loss * inputs["input_ids"].shape[1]
            
            total_nll += nll.item()
            total_tokens += inputs["input_ids"].shape[1]
    
    avg_nll = total_nll / max(total_tokens, 1)
    ppl = torch.exp(torch.tensor(avg_nll)).item()
    
    return ppl


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    
    # Create sampler
    sampler = LoMDMSampler(model, config.mask_token_id)
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    
    all_samples = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(num_batches), desc="Sampling"):
        batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
        
        if args.semi_ar:
            # Semi-autoregressive generation
            output = sampler.sample_semi_ar(
                num_samples=batch_size,
                total_length=args.total_length,
                block_length=args.block_length,
                num_steps_per_block=args.num_steps,
                temperature=args.temperature,
                use_learned_scheduler=args.use_learned_scheduler,
                device=device,
            )
        else:
            # Standard sampling
            if args.method == "ddpm":
                output = sampler.sample_ddpm(
                    batch_size=batch_size,
                    seq_length=args.seq_length,
                    num_steps=args.num_steps,
                    temperature=args.temperature,
                    use_learned_scheduler=args.use_learned_scheduler,
                    device=device,
                )
            elif args.method == "ddpm_cache":
                output = sampler.sample_ddpm_cache(
                    batch_size=batch_size,
                    seq_length=args.seq_length,
                    num_steps=args.num_steps,
                    temperature=args.temperature,
                    use_learned_scheduler=args.use_learned_scheduler,
                    device=device,
                )
            elif args.method == "confidence":
                output = sampler.sample_confidence(
                    batch_size=batch_size,
                    seq_length=args.seq_length,
                    num_steps=args.num_steps,
                    temperature=args.temperature,
                    device=device,
                )
        
        all_samples.append(output.samples)
    
    # Concatenate all samples
    samples = torch.cat(all_samples, dim=0)[:args.num_samples]
    
    # Decode to text
    texts = tokenizer.batch_decode(samples, skip_special_tokens=True)
    
    # Save samples
    print(f"\nSaving samples to {args.output_file}")
    with open(args.output_file, "w") as f:
        for i, text in enumerate(texts):
            f.write(f"=== Sample {i+1} ===\n")
            f.write(text.strip() + "\n\n")
    
    # Print some samples
    print("\n=== Sample outputs ===")
    for i in range(min(5, len(texts))):
        print(f"\n--- Sample {i+1} ---")
        print(texts[i][:500] + "..." if len(texts[i]) > 500 else texts[i])
    
    # Compute generative perplexity
    if args.compute_ppl:
        print("\nComputing generative perplexity with GPT-2...")
        ppl = compute_generative_perplexity(samples, tokenizer)
        print(f"Generative PPL: {ppl:.2f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
