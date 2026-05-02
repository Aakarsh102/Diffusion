import torch
import argparse
import os
import json
from copy import deepcopy
from omegaconf import OmegaConf, ListConfig
from model.transformer import MDMTransformer, MDMConfig
from upm import UPM
from transformers import AutoTokenizer
from sampling import mdm_sampling
from eval.gsm8k_eval import mdm_sampling_upm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--type", type=str, required=True, choices=["mdlm", "puma"], help="Model type to build (mdlm or puma)")
    parser.add_argument("--cfg", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--temp", type=float, default=1.5, help="Sampling temperature (e.g. 1.0 or 1.5)")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Determine mask_id and vocab extensions (like in train.py)
    data_cfg = cfg.data
    if data_cfg.dataset == "lm1b":
        meta_path = os.path.join(data_cfg.data_dir, "meta.json")
        with open(meta_path) as fh:
            meta = json.load(fh)
        mask_id = meta["vocab_size"] + 1
        cfg.model.vocab_size = meta["vocab_size"] + 2
        cfg.model.max_position = meta["max_len"]
    else:
        mask_id = data_cfg.mask_id

    # 2. Build the base model
    model_config = MDMConfig(**cfg.model)
    model = MDMTransformer(model_config).to(device)

    # 3. Load Checkpoint
    print(f"Loading {args.type.upper()} checkpoint from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    
    # Try to extract the EMA weights first, fallback to standard model weights
    sd = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    # 4. Build UPM (if PUMA)
    upm = None
    if args.type == "puma":
        condition_dim = model_config.hidden_size
        upm = UPM(hidden_size=model_config.hidden_size, condition_dim=condition_dim, num_heads=8).to(device)
        
        upm_sd = ckpt.get("upm_state_dict", None)
        if upm_sd is not None:
            upm_sd = {k.replace("module.", ""): v for k, v in upm_sd.items()}
            upm.load_state_dict(upm_sd, strict=True)
            print("Successfully explicitly loaded UPM weights!")
        else:
            print("Warning: No UPM weights found in checkpoint! Using random initialization.")
        
        upm.eval()

    # 5. Build blank context
    L = model_config.max_position
    x_t = torch.full((args.n_samples, L), mask_id, dtype=torch.long, device=device)

    # 6. Parse Sampling Config
    diag_sampling_cfg = deepcopy(cfg.validation.sampling)
    if isinstance(diag_sampling_cfg.confidence, (list, ListConfig)):
        diag_sampling_cfg.confidence = str(diag_sampling_cfg.confidence[0])
    if isinstance(diag_sampling_cfg.unmasking_num, (list, ListConfig)):
        diag_sampling_cfg.unmasking_num = int(diag_sampling_cfg.unmasking_num[0])
    
    diag_sampling_cfg.temperature = args.temp

    print(f"Generating {args.n_samples} unconditional samples with temperature={args.temp}...")
    with torch.inference_mode():
        if args.type == "puma" and upm is not None:
            samples = mdm_sampling_upm(model, upm, x_t, mask_id, diag_sampling_cfg, device=device)
        else:
            samples = mdm_sampling(model, x_t, mask_id, diag_sampling_cfg, device=device)
    
    # 7. Print Output
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    print(f"\n{'='*60}")
    print(f"GENERATED SAMPLES")
    print(f"{'='*60}")
    for i in range(args.n_samples):
        text = tok.decode(samples[i].tolist(), skip_special_tokens=True)
        print(f"\n--- Sample {i+1} ---")
        print(text[:1000])
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
