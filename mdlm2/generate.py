# generate_teacher_trajectories.py
import torch
import json
from tqdm import tqdm
import os
os.environ["OPENFOLD_DISABLE_CUDA_EXT"] = "1"
import sys
os.environ['HF_HOME'] = '/scratch/gilbreth/rai53/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/gilbreth/rai53/hf_cache'
from byprot.models.dplm import DiffusionProteinLanguageModel as DPLM

# Config
TEACHER = 'airkingbd/dplm_150m'
OUTPUT = './trajectories.jsonl'
NUM_SAMPLES = 99200
BATCH_SIZE = 128
SEQ_LEN = 128
STEPS = 64

def generate_with_history(model, input_tokens, max_iter):
    """Fixed version that tracks history AFTER remasking"""
    encoder_out = model.forward_encoder(input_tokens)
    init_tokens, init_scores = model.initialize_output_tokens(input_tokens, encoder_out=encoder_out)
    
    prev = {
        'output_tokens': init_tokens,
        'output_scores': init_scores,
        'output_masks': model.get_non_special_symbol_mask(init_tokens),
        'step': 0,
        'max_step': max_iter,
        'temperature': 1.0,
        'history': [init_tokens.clone()],
    }
    
    # Track our own history AFTER remasking
    true_history = [init_tokens.clone()]
    
    for step in range(max_iter):
        with torch.no_grad():
            decoder_out = model.forward_decoder(
                prev_decoder_out=prev,
                encoder_out=encoder_out,
                sampling_strategy='gumbel_argmax',
                disable_resample=False,
                resample_ratio=0.25,
            )
        
        non_special = model.get_non_special_symbol_mask(prev['output_tokens'])
        
        masks, tokens, scores = model._reparam_decoding(
            output_tokens=prev['output_tokens'].clone(),
            output_scores=prev['output_scores'].clone(),
            cur_tokens=decoder_out['output_tokens'].clone(),
            cur_scores=decoder_out['output_scores'].clone(),
            decoding_strategy="reparam-uncond-deterministic-linear",
            xt_neq_x0=prev['output_masks'],
            non_special_sym_mask=non_special,
            t=step + 1,
            max_step=max_iter,
            noise=model.mask_id,
        )
        
        # Save AFTER remasking
        true_history.append(tokens.clone())
        
        prev.update({
            'output_masks': masks,
            'output_tokens': tokens,
            'output_scores': scores,
            'step': step + 1,
            'history': decoder_out['history'],
        })
    
    prev['history'] = true_history
    return prev

def extract_trajectories(decoder_out, mask_id):
    """
    Same structure as before, but for each time step,
    unmasked_indices / unmasked_tokens are sorted by logit (output_scores)
    in decreasing order.
    """
    history = decoder_out['history']          # list of [B, L] token tensors
    scores = decoder_out['output_scores']     # [B, L] logits/scores for final tokens

    B, L = history[0].shape
    results = []

    for b in range(B):
        unmasked_indices = []
        unmasked_tokens = []

        ever_revealed = torch.zeros(L, dtype=torch.bool, device=history[0].device)
        x0_tokens = history[-1][b]   # final tokens
        x0_scores = scores[b]        # logits for those tokens

        for step in range(len(history) - 1):
            prev = history[step][b]        # [L]
            curr = history[step + 1][b]    # [L]

            # positions that changed from mask -> non-mask at THIS step
            changed = (prev == mask_id) & (curr != mask_id)
            newly_revealed = changed & (~ever_revealed)

            pos = torch.where(newly_revealed)[0]   # [k]
            if pos.numel() > 0:
                # sort these positions by logit (score) descending
                step_scores = x0_scores[pos]
                order = torch.argsort(step_scores, descending=True)
                pos_sorted = pos[order]

                idx = pos_sorted.cpu().tolist()
                tok = x0_tokens[pos_sorted].cpu().tolist()
            else:
                idx = []
                tok = []

            # mark as revealed so we don't count again
            ever_revealed = ever_revealed | newly_revealed

            unmasked_indices.append(idx)
            unmasked_tokens.append(tok)

        results.append({
            'x0_tokens': x0_tokens.cpu().tolist(),
            'unmasked_indices': unmasked_indices,
            'unmasked_tokens': unmasked_tokens,
            'T': len(unmasked_indices),
            'L': L,
        })

    return results

# Main
print(f"Loading DPLM: {TEACHER}")
dplm = DPLM.from_pretrained(TEACHER).cuda()
dplm.eval()

print(f"Mask ID: {dplm.mask_id}")
print(f"Pad ID: {dplm.pad_id}")
print(f"Tokenizer vocab size: {len(dplm.tokenizer)}")
print(f"Sample tokens: {dplm.tokenizer.convert_ids_to_tokens([0, 1, 2, 3, 4, 5])}")

input_tokens = torch.full((BATCH_SIZE, SEQ_LEN), dplm.mask_id, dtype=torch.long).cuda()
print(f"Input tokens (should be mask_id): {input_tokens[0, :5]}")

num_batches = NUM_SAMPLES // BATCH_SIZE

print(f"Generating {NUM_SAMPLES} samples...")
with open(OUTPUT, 'w') as f:
    for _ in tqdm(range(num_batches)):
        input_tokens = torch.full((BATCH_SIZE, SEQ_LEN), dplm.mask_id, dtype=torch.long).cuda()
        input_tokens[:, 0] = dplm.bos_id

        decoder_out = generate_with_history(dplm, input_tokens, STEPS)
        trajs = extract_trajectories(decoder_out, dplm.mask_id)

        for t in trajs:
            f.write(json.dumps(t) + '\n')
        f.flush()

print(f"Done! Saved to {OUTPUT}")