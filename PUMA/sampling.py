import torch
from omegaconf import ListConfig
import torch.nn.functional as F

def gumbel_softmax(logits, temperature):
    """
    Sample from the Gumbel-Softmax distribution and optionally apply softmax.
    """
    if temperature == 0.0:
        return logits
    else:
        noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
        return logits / temperature + gumbel_noise

@torch.no_grad()
def arm_sampling(model, xt, mask_id, sampling_cfg, device: torch.device = None, track: bool = False):
    """
    Autoregressive sampling:
      - generates one token at a time, left-to-right
      - starts at the first mask_id position (prompt end)
      - uses gumbel-softmax + argmax (same style as mdm_sampling)
      - if eos_id is provided, stops per sequence at first EOS and fills the rest with EOS
    """
    temperature = sampling_cfg.temperature
    eos_id = getattr(sampling_cfg, "eos_id", None)

    B, L = xt.shape
    xt = xt.clone()
    if track:
        track_xt = []

    # start position per row = first mask_id (prompt end)
    is_mask = (xt == mask_id)
    any_mask = is_mask.any(dim=1)
    start_pos = is_mask.float().argmax(dim=1)                 # 0 if no mask
    start_pos = torch.where(any_mask, start_pos, torch.full_like(start_pos, L))

    # nothing to generate
    if int(start_pos.min().item()) >= L:
        # if eos_id is set, still ensure "fill to L with eos" is trivially satisfied (nothing to do)
        return (xt, track_xt) if track else xt

    # track which sequences are finished (EOS generated)
    done = torch.zeros(B, dtype=torch.bool, device=xt.device) if eos_id is not None else None

    for pos in range(L):
        # only generate for rows where:
        #  - pos is in the generation region
        #  - token is still mask
        #  - and (if eos_id) we haven't already produced EOS
        can_fill = (pos >= start_pos) & (xt[:, pos] == mask_id)
        if eos_id is not None:
            can_fill = can_fill & (~done)

        if not can_fill.any():
            continue

        # to predict token at `pos`, we use logits at `pos-1`
        src = pos - 1
        if src < 0:
            continue

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits = model(xt)  # (B, L, V)

        logits_step = logits[:, src, :]  # predicts token at pos
        noisy = gumbel_softmax(logits_step, temperature=temperature)  # (B, V)
        next_tok = torch.argmax(noisy, dim=-1)                        # (B,)

        xt[can_fill, pos] = next_tok[can_fill]

        if eos_id is not None:
            # mark sequences as done if EOS was just produced at this position
            done = done | (can_fill & (next_tok == eos_id))

        if track:
            track_xt.append(xt.clone().detach().cpu())

    # If eos_id is set: for any sequence that has EOS in the generation region,
    # fill all later positions (up to length L) with EOS.
    if eos_id is not None:
        pos_idx = torch.arange(L, device=xt.device).unsqueeze(0)  # (1, L)
        gen_region = pos_idx >= start_pos.unsqueeze(1)            # (B, L)
        eos_in_region = (xt == eos_id) & gen_region               # (B, L)
        any_eos = eos_in_region.any(dim=1)                        # (B,)

        if any_eos.any():
            first_eos = eos_in_region.float().argmax(dim=1)       # (B,) (meaningful only where any_eos)
            # fill strictly after first_eos (and within gen_region) with eos_id
            fill_mask = any_eos.unsqueeze(1) & gen_region & (pos_idx > first_eos.unsqueeze(1))
            xt[fill_mask] = eos_id

            if track:
                # optional: record the post-fill final state once
                track_xt.append(xt.clone().detach().cpu())

    return (xt, track_xt) if track else xt

@torch.no_grad()
def mdm_sampling(model, xt, mask_id, sampling_cfg, device: torch.device = None, track: bool = False, arm_init: bool = False):
    # sampling hyperparameters
    temperature = sampling_cfg.temperature
    confidence = sampling_cfg.confidence
    unmasking_num = sampling_cfg.unmasking_num

    # Handle ListConfig from OmegaConf (when config has lists instead of scalars)
    if isinstance(unmasking_num, (ListConfig)):
        unmasking_num = 16
    try:
        unmasking_num = int(unmasking_num)
    except (TypeError, ValueError):
        print('ex')
        unmasking_num = 16

    try:
        confidence = (confidence)
    except (TypeError, ValueError):
        print('ex')
        unmasking_num = (confidence[0])
    #if isinstance(confidence, (ListConfig)):
    #    confidence = confidence[0]

    # ... rest of function continues unchanged

    # shape
    B, L = xt.shape
    xt = xt.clone()
    if track:
        track_xt = []

    if arm_init:
        xt_t1, xt = xt[:, :1], xt[:, 1:]
        L = L - 1

    for i in range(L // unmasking_num + 1):
        # mask indicies
        mask_indices = (xt == mask_id)

        if mask_indices.sum() == 0:
            break

        # calculate logits
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
            logits = model(torch.cat([xt_t1, xt], dim=1) if arm_init else xt) # [B, L, V]
        if arm_init:
            logits = logits[:, :-1, :]
        logits_with_noise = gumbel_softmax(logits, temperature = temperature)
        p = F.softmax(logits, dim = -1)

        if confidence == "top_k":
            unmasking_score = torch.where(mask_indices, p.max(dim = -1).values, -float('inf'))
        elif confidence == "top_k_margin":
            probs_top_2 = p.topk(k=2, dim=-1).values
            unmasking_score = torch.where(mask_indices, probs_top_2[..., 0] - probs_top_2[..., 1], -float('inf'))
        elif confidence == "entropy":
            entropy = (- p * torch.log(p + 1e-10)).sum(dim = -1)
            unmasking_score = torch.where(mask_indices, entropy, -float('inf'))
        elif confidence == "random":
            raise NotImplementedError("Random confidence sampling strategy yet to be implemented")
        else:
            raise NotImplementedError(f"Confidence sampling strategy '{confidence}' not supported")
        
        # update masked tokens by selecting top-k per batch this step
        for j in range(B):
            k = min(unmasking_num, int(mask_indices[j].sum().item())) # number of tokens to unmask
            if k > 0:
                _, select_indices = torch.topk(unmasking_score[j], k=k)
                xt[j, select_indices] = torch.argmax(logits_with_noise[j, select_indices], dim = -1)
            
        if track:
            cur = torch.cat([xt_t1, xt], dim=1) if arm_init else xt
            track_xt.append(cur.clone().detach().cpu())
    if arm_init:
        xt = torch.cat([xt_t1, xt], dim=1)
    if track:
        return xt, track_xt
    else:
        return xt
    
@torch.no_grad()
def mdm_sampling_upmi_old(model, upm, xt, mask_id, sampling_cfg, device=None):
    temperature = sampling_cfg.temperature
    unmasking_num = int(sampling_cfg.unmasking_num)
    B, L = xt.shape
    xt = xt.clone()
    K_steps = L // unmasking_num + 1

    for step in range(K_steps):
        mask_indices = (xt == mask_id)
        if mask_indices.sum() == 0:
            break

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, hidden = model(xt, return_hidden=True)

        logits_with_noise = gumbel_softmax(logits, temperature=temperature)

        # UPM scores instead of confidence
        t_step = torch.full((B,), step / K_steps, device=xt.device)
        upm_scores = upm(hidden, t_step, mask_indices)
        unmasking_score = upm_scores.masked_fill(~mask_indices, float('-inf'))

        for j in range(B):
            k = min(unmasking_num, int(mask_indices[j].sum().item()))
            if k > 0:
                _, idx = torch.topk(unmasking_score[j], k=k)
                xt[j, idx] = torch.argmax(logits_with_noise[j, idx], dim=-1)
    
    return xt

@torch.no_grad()
def mdm_sampling_upm(model, upm, xt, mask_id, sampling_cfg, device=None):
    temperature = sampling_cfg.temperature
    unmasking_num = int(sampling_cfg.unmasking_num)
    B, L = xt.shape
    xt = xt.clone()
    K_steps = L // unmasking_num + 1

    # prompt positions never change — identify them once
    prompt_mask = (xt != mask_id)  # True for prompt tokens at step 0
    L_eff = (~prompt_mask).sum(dim=1).clamp_min(1).float()  # non-prompt length per seq

    for step in range(K_steps):
        mask_indices = (xt == mask_id)
        if mask_indices.sum() == 0:
            break

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, hidden = model(xt, return_hidden=True)

        logits_with_noise = gumbel_softmax(logits, temperature=temperature)

        # t_step = fraction of non-prompt tokens already unmasked
        n_unmasked = (~mask_indices & ~prompt_mask).sum(dim=1).float()
        t_step = (n_unmasked / L_eff).clamp(0.0, 1.0)

        upm_scores = upm(hidden, t_step, mask_indices)
        # NaN/inf safety — topk has undefined behavior with NaN
        upm_scores = torch.nan_to_num(upm_scores, nan=0.0, posinf=10.0, neginf=-10.0)

        # Blend: UPM score + max logit (confidence proxy) so even untrained UPM
        # falls back to confidence-based ordering instead of random.
        max_logit = logits.max(dim=-1).values  # (B, L), no extra memory
        blended = upm_scores + max_logit

        unmasking_score = blended.masked_fill(~mask_indices, float('-inf'))

        for j in range(B):
            k = min(unmasking_num, int(mask_indices[j].sum().item()))
            if k > 0:
                _, idx = torch.topk(unmasking_score[j], k=k)
                xt[j, idx] = torch.argmax(logits_with_noise[j, idx], dim=-1)

    return xt













# @torch.no_grad()
# def mdm_sampling_upm(model, upm, xt, mask_id, sampling_cfg, device=None):
#     temperature = sampling_cfg.temperature
#     unmasking_num = int(sampling_cfg.unmasking_num)
#     B, L = xt.shape
#     xt = xt.clone()
#     K_steps = L // unmasking_num + 1

#     # prompt positions never change — identify them once
#     prompt_mask = (xt != mask_id)  # True for prompt tokens at step 0
#     L_eff = (~prompt_mask).sum(dim=1).clamp_min(1).float()  # non-prompt length per seq

#     for step in range(K_steps):
#         mask_indices = (xt == mask_id)
#         if mask_indices.sum() == 0:
#             break

#         with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#             logits, hidden = model(xt, return_hidden=True)

#         logits_with_noise = gumbel_softmax(logits, temperature=temperature)

#         # t_step = fraction of non-prompt tokens already unmasked
#         n_unmasked = (~mask_indices & ~prompt_mask).sum(dim=1).float()
#         t_step = (n_unmasked / L_eff).clamp(0.0, 1.0)

#         upm_scores = upm(hidden, t_step, mask_indices)
#         # NaN/inf safety — topk has undefined behavior with NaN
#         upm_scores = torch.nan_to_num(upm_scores, nan=0.0, posinf=10.0, neginf=-10.0)

#         # Blend: UPM score + log(model confidence) so even untrained UPM
#         # falls back to confidence-based ordering instead of random.
#         p = F.softmax(logits.float(), dim=-1)
#         log_conf = torch.log(p.max(dim=-1).values + 1e-8)  # (B, L)
#         blended = upm_scores + log_conf

#         unmasking_score = blended.masked_fill(~mask_indices, float('-inf'))

#         for j in range(B):
#             k = min(unmasking_num, int(mask_indices[j].sum().item()))
#             if k > 0:
#                 _, idx = torch.topk(unmasking_score[j], k=k)
#                 xt[j, idx] = torch.argmax(logits_with_noise[j, idx], dim=-1)

#     return xt











@torch.no_grad()
def mdm_sampling_block(model, xt, block_size, mask_id, sampling_cfg, device: torch.device = None):
    temperature = sampling_cfg.temperature
    confidence = sampling_cfg.confidence
    unmasking_num = sampling_cfg.unmasking_num
    device = xt.device

    # shape 
    B, L = xt.shape
    assert L % block_size == 0, "block size must be divisible by the max_length"
    n_blocks = L // block_size
    xt = xt.clone()

    for n in range(n_blocks):
        s = n * block_size
        e = s + block_size

        for i in range(block_size // unmasking_num + 2):
            valid_mask_ids = (xt[:, s:e] == mask_id) # [B, e]
            if valid_mask_ids.sum() == 0:
                break

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled = torch.cuda.is_available()):
                logits = model(xt[:, : e]) # [B, e, V]

            logits_with_noise = gumbel_softmax(logits[:, s:e, :], temperature = temperature) # [B, e, V]
            p = F.softmax(logits[:, s:e, :], dim = -1)

            if confidence == "top_k":
                unmasking_score = torch.where(valid_mask_ids, p.max(dim = -1).values, -float('inf'))
            elif confidence == "top_k_margin":
                probs_top_2 = p.topk(k=2, dim=-1).values
                unmasking_score = torch.where(valid_mask_ids, probs_top_2[..., 0] - probs_top_2[..., 1], -float('inf'))
            elif confidence == "random":
                raise NotImplementedError("Random confidence sampling strategy yet to be implemented")
            else:
                raise NotImplementedError(f"Confidence sampling strategy '{confidence}' not supported")
                
            for j in range(B):
                k = min(unmasking_num, int(valid_mask_ids[j].sum().item())) # number of tokens to unmask
                if k > 0:
                    _, select_indices = torch.topk(unmasking_score[j], k=k)
                    xt[j, s + select_indices] = torch.argmax(logits_with_noise[j, select_indices], dim = -1)
    
    assert (xt == mask_id).sum() == 0, "There are still masked tokens in the input"
            
    return xt
