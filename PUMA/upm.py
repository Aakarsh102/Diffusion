import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, hidden_size, condition_size, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine = False, eps = eps) 
        self.modulation = nn.Linear(condition_size, hidden_size * 2)
        self.set_params()

    def set_params(self):
        nn.init.xavier_uniform_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
    def forward(self, x, condition):
        x_norm = self.norm(x)
        scale, shift = self.modulation(condition).chunk(2, dim = -1)
        if x.ndim == 3 and scale.ndim == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return x_norm * (1 + scale) + shift
    
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, freq_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_size = freq_size
        self.hidden_size = hidden_size

    def set_params(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def timestep_embedding(t, dim, max_period = 10000):
        half_dim = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half_dim, dtype=torch.float32).to(t.device) / half_dim)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding
    def forward(self, t):
        # Scale continuous t in [0, 1] to [0, 1000] for standard sinusoidal embeddings.
        # Otherwise, the maximum argument to sin/cos is 1.0 (no oscillation), 
        # making different timesteps virtually indistinguishable.
        t_scaled = t * 1000.0
        t_freq = self.timestep_embedding(t_scaled, self.freq_size)
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)

class UPM(nn.Module):
    def __init__(self, hidden_size, condition_dim, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        # The paper specifies: Adaptive LayerNorm -> Transformer Block -> Adaptive LayerNorm
        self.ada_ln_1 = AdaptiveLayerNorm(hidden_size, condition_dim)
        self.time_embedding = TimestepEmbedder(hidden_size=condition_dim // 2)
        self.mask_embedding = nn.Embedding(2, condition_dim // 2) # To embed the mask indicators (0/1) into the condition vector
        # A single transformer encoder layer as specified in the paper
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            batch_first=True,
            norm_first=False # Handled by our custom AdaLN
        )
        
        self.ada_ln_2 = AdaptiveLayerNorm(hidden_size, condition_dim)
        
        # Final head to predict the scalar ranking score h_{\theta, n} for each token
        self.score_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, t, mask_indicator):
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        t: [batch_size]
        mask_indicator: [batch_size, seq_len] bool or long, True/1 = masked
        """
        seq_len = hidden_states.size(1)
        mask_emb = self.mask_embedding(mask_indicator.long())  # (B, L, cond//2)
        time_emb = self.time_embedding(t).unsqueeze(1).expand(-1, seq_len, -1)  # (B, L, cond//2)
        cond_embedding = torch.cat([time_emb, mask_emb], dim=-1)  # (B, L, cond)
        
        # 1. First Adaptive LayerNorm
        x = self.ada_ln_1(hidden_states, cond_embedding)
        
        # 2. Transformer Block
        x = self.transformer_block(x)
        
        # 3. Second Adaptive LayerNorm
        x = self.ada_ln_2(x, cond_embedding)
        
        # 4. Predict ranking scores h_{\theta, n}
        scores = self.score_head(x).squeeze(-1) # Shape: [batch_size, seq_len]
        return scores

def sample_plackett_luce(scores, mask_indicators, K):
    """
    Samples K tokens to unmask using the Gumbel-Max trick, which is standard for 
    differentiable sampling from a Plackett-Luce distribution.
    """
    # Gumbel noise for sampling
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
    perturbed_scores = scores + gumbel_noise
    
    # We only want to unmask tokens that are currently masked. 
    # Set scores of already unmasked tokens to -infinity so they aren't chosen.
    perturbed_scores = perturbed_scores.masked_fill(~mask_indicators.bool(), float('-inf'))
    
    # Get the top K tokens
    topk_scores, unmask_indices = torch.topk(perturbed_scores, k=K, dim=-1)
    
    return unmask_indices, topk_scores


def plackett_luce_log_prob(scores, sampled_indices, mask):
    B, K = sampled_indices.shape
    masked_scores = scores.masked_fill(~mask, float('-inf'))
    
    log_prob = torch.zeros(B, device=scores.device, dtype=scores.dtype)
    remaining = masked_scores  # no clone needed; we'll use out-of-place ops
    
    for k in range(K):
        idx_k = sampled_indices[:, k]                       # (B,)
        score_k = scores.gather(1, idx_k.unsqueeze(1)).squeeze(1)  # (B,)
        
        log_denominator = remaining.logsumexp(dim=-1).clamp(min=-30.0)
        log_prob = log_prob + (score_k - log_denominator)
        
        # out-of-place: scatter -inf into the selected positions
        neg_inf = torch.full_like(idx_k, float('-inf'), dtype=remaining.dtype).unsqueeze(1)
        remaining = remaining.scatter(1, idx_k.unsqueeze(1), neg_inf)
    
    return log_prob

def plackett_luce_log_prob2(scores, sampled_indices, mask):
    B, K = sampled_indices.shape
    masked_scores = scores.masked_fill(~mask, float('-inf'))
    
    log_prob = torch.zeros(B, device=scores.device)
    remaining = masked_scores.clone()
    
    for k in range(K):
        idx_k = sampled_indices[:, k]
        score_k = scores[torch.arange(B), idx_k]
        
        # logsumexp can be -inf if remaining is all -inf; guard it
        log_denominator = remaining.logsumexp(dim=-1)
        log_denominator = log_denominator.clamp(min=-30.0)  # numerical floor
        
        log_prob = log_prob + (score_k - log_denominator)
        remaining[torch.arange(B), idx_k] = float('-inf')
    
    return log_prob



@torch.no_grad()
def compute_reward(model, x0, logits_before, xt_after, mask_id, prompt_mask):
    """
    Reward = mean NLL reduction on remaining masked tokens after a single unmasking.

    r = mean_NLL_before[remaining] - mean_NLL_after[remaining]
    """
    B = x0.shape[0]
    remaining = (xt_after == mask_id) & ~prompt_mask

    has_signal = remaining.any(dim=1)
    if not has_signal.any():
        return torch.zeros(B, device=x0.device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits_after = model(xt_after)

    def nll(logits):
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            x0.view(-1), reduction='none'
        ).view_as(x0)

    nll_b = nll(logits_before)
    nll_a = nll(logits_after)

    n = remaining.sum(dim=1).clamp_min(1).float()
    r = (nll_b * remaining).sum(dim=1) / n - (nll_a * remaining).sum(dim=1) / n
    r = torch.where(has_signal, r, torch.zeros_like(r))

    return r


@torch.no_grad()
def compute_reward_pair(model, x0, logits_before, xt_after_1, xt_after_2,
                        mask_id, prompt_mask):
    """
    Reward = mean NLL reduction on each unmasking's OWN remaining masked tokens.

    r1 = mean_NLL_before[remaining_1] - mean_NLL_after_1[remaining_1]
    r2 = mean_NLL_before[remaining_2] - mean_NLL_after_2[remaining_2]

    This avoids the "common" intersection shrinking to zero when k_unmask
    is close to the total number of masked tokens.
    """
    B = x0.shape[0]
    remaining_1 = (xt_after_1 == mask_id) & ~prompt_mask   # tokens still masked after U1
    remaining_2 = (xt_after_2 == mask_id) & ~prompt_mask   # tokens still masked after U2

    has_signal = (remaining_1.any(dim=1) | remaining_2.any(dim=1))
    if not has_signal.any():
        z = torch.zeros(B, device=x0.device)
        return z, z

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits_1 = model(xt_after_1)
        logits_2 = model(xt_after_2)

    def nll(logits):
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            x0.view(-1), reduction='none'
        ).view_as(x0)

    nll_b = nll(logits_before)
    nll_1 = nll(logits_1)
    nll_2 = nll(logits_2)

    # reward for unmasking 1: NLL improvement on remaining_1
    n1 = remaining_1.sum(dim=1).clamp_min(1).float()
    r1 = (nll_b * remaining_1).sum(dim=1) / n1 - (nll_1 * remaining_1).sum(dim=1) / n1

    # reward for unmasking 2: NLL improvement on remaining_2
    n2 = remaining_2.sum(dim=1).clamp_min(1).float()
    r2 = (nll_b * remaining_2).sum(dim=1) / n2 - (nll_2 * remaining_2).sum(dim=1) / n2

    # zero out sequences with no remaining tokens
    r1 = torch.where(remaining_1.any(dim=1), r1, torch.zeros_like(r1))
    r2 = torch.where(remaining_2.any(dim=1), r2, torch.zeros_like(r2))

    return r1, r2


@torch.no_grad()
def compute_reward_pair_old(model, x0, logits_before, xt_after_1, xt_after_2,
                            mask_id, prompt_mask):
    """
    Rewards evaluated on the intersection of remaining masks. 
    (Old version, kept for reference/comparison)
    """
    B = x0.shape[0]
    common = (xt_after_1 == mask_id) & (xt_after_2 == mask_id) & ~prompt_mask

    has_signal = common.any(dim=1)
    if not has_signal.any():
        z = torch.zeros(B, device=x0.device)
        return z, z

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits_1 = model(xt_after_1)
        logits_2 = model(xt_after_2)

    def nll(logits):
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            x0.view(-1), reduction='none'
        ).view_as(x0)

    nll_b = nll(logits_before)
    nll_1 = nll(logits_1)
    nll_2 = nll(logits_2)

    n_common = common.sum(dim=1).clamp_min(1).float()
    r1 = (nll_b * common).sum(dim=1) / n_common - (nll_1 * common).sum(dim=1) / n_common
    r2 = (nll_b * common).sum(dim=1) / n_common - (nll_2 * common).sum(dim=1) / n_common

    r1 = torch.where(has_signal, r1, torch.zeros_like(r1))
    r2 = torch.where(has_signal, r2, torch.zeros_like(r2))

    return r1, r2