import torch
import torch.nn as nn
import torch.nn.functional as F

# orders = (B, seq, 10)
# time_step = int 

    

class Model(nn.Module):
    def __init__(self, device, d_model, seq_len, n_layers, n_heads, vocab_size, mask_id = 0, max_len = 1024):
        self.n_layers = 2
        self.n_heads = 2
        self.device = device
        # self.emb_dim = emb_dim
        self.d_model = d_model
        self.order_weight = 0.1
        # self.orders = 10
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.max_len = max_len
        self.layer = nn.TransformerEncoderLayer(d_model=self.d_model, 
                                                nhead = self.n_heads, 
                                                dim_feedforward=128, 
                                                device=self.device
                                                )
        self.transformer = nn.TransformerEncoder(self.layer, self.n_layers)

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)

        self.token_head = nn.Linear(self.d_model, self.vocab_size)
        self.order_head = nn.Linear(self.d_model, 1)

    def encode(self, batch, use_mask = True):
        #Batch, seq_len
        B, L = batch.size
        pos = torch.arange(L, device = batch.device).unsqueeze(0).expand((B, L))
        emb = self.tok_emb(batch) + self.pos_emb(pos)  # [B, L, d_model]
        if not use_mask:
            h = self.transformer(emb)  # [B, L, d_model]
            return h
        attn_mask = self.create_attention_mask_per_batch(batch)  # [B*n_heads, L, L]
        return h

    def forward(self, x_in, x_target):
        h = self.encode(x_in)
        logits_tok = self.token_head(h)             # [B, L, V]
        scores_ord = self.order_head(h).squeeze(-1)
        M = x_in.eq(self.mask_id)

        tok_loss = F.cross_entropy(
            logits_tok[M], x_target[M]
        ) if M.any() else torch.zeros((), device=x_in.device)

        with torch.no_grad():
            probs = logits_tok.softmax(-1)          # [B, L, V]
            ent = -(probs.clamp_min(1e-9).log() * probs).sum(-1)   # [B, L]
            ent = ent.masked_fill(~M, float('inf')) # only masked compete
            target = torch.softmax(-ent, dim=-1)    # low entropy => high prob

        masked_ords = scores_ord.masked_fill(~M, float('-inf'))
        order_logprobs = masked_ords.log_softmax(-1) # [B, L]
        order_loss = F.kl_div(order_logprobs, target, reduction='batchmean')

        total_loss = tok_loss + self.order_weight * order_loss
        return {"loss": total_loss, "token_loss": tok_loss.detach(), "order_loss": order_loss.detach()}
    

    @torch.no_grad()
    def generate(self, batch_size, seq_len, temperature=1.0, 
             sample_order=False, sample_token=False, topk=None,
             use_attention_mask=True, eos_token_id=None, 
             stop_on_eos=True):
        
        x = torch.full((batch_size, seq_len), self.mask_id, device = self.device, dtype=torch.bool)

        if stop_on_eos and eos_token_id is not None:
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        else:
            finished = None

        for step in range(seq_len):
            if finished is not None and finished.all():
                break
            h = self.encode(x, use_mask=use_attention_mask)  # [B, L, d_model]
            order_scores = self.order_head(h).squeeze(-1)    # [B, L]
            masked_scores = self.token_head(h)
            M = (x == self.mask_id)
            if finished is not None:
                M = M & ~finished.unsqueeze(1)
            if not M.any():
                break
            order_scores_masked = order_scores.masked_fill(~M, float('-inf'))
            if sample_order:
                order_probs = F.softmax(order_scores_masked / temperature, dim=-1)
                # Handle case where some batches have no valid positions
                order_probs = torch.nan_to_num(order_probs, 0.0)
                next_pos = torch.multinomial(order_probs, num_samples=1).squeeze(-1)
            else:
                next_pos = order_scores_masked.argmax(dim=-1)
            
            token_logits_all = self.token_head(h) # [B, L, V]
            token_logits = token_logits_all[torch.arange(batch_size), next_pos] # [B, V]
            tok_logits = token_logits / temperature

            if topk is not None: 
                v, _ = torch.topk(tok_logits, min(topk, tok_logits.size(-1)))
                tok_logits[tok_logits < v[:, [-1]]] = float('-inf')
            if sample_token:
                new_token = torch.distributions.Categorical(logits=tok_logits).sample()  # [B]
            else:
                new_token = tok_logits.argmax(dim = -1) # [B]
            
            x[torch.arange(batch_size), next_pos] = new_token

            if stop_on_eos and eos_token_id is not None:
                # Mark sequences as finished if they generated EOS
                eos_generated = (new_token == eos_token_id)
                if finished is not None:
                    finished = finished | eos_generated

        return x # [B, L]
                 

    @torch.no_grad()
    def generate_2(self, x_init, max_steps, temperature=1.0,
        sample_index=False, sample_token=False, topk=None):

        x = x_init.clone()
        B, L = x.shape
        M = x.eq(self.mask_id)
        steps = 0

        while M.any() and steps < max_steps:
            h = self.encode(x)
            order_scores = self.order_head(h).squeeze(-1)
            masked_scores = order_scores.masked_fill(~M, float('-inf'))
            if sample_index:
                z = torch.distributions.Categorical(logits=order_scores/temperature).sample()   # [B]
            else:
                z = masked_scores.argmax(dim=-1)

            tok_logits_all = self.token_head(h)  # [B, L, V]

            tok_logits = tok_logits_all[torch.arange(B, device=x.device), z] / temperature

            if topk is not None:
                topk_idx = tok_logits.topk(topk).indices
                mask = torch.full_like(tok_logits, float('-inf'))
                tok_logits = mask.scatter(1, topk_idx, tok_logits.gather(1, topk_idx))

            if sample_token:
                new_tok = torch.distributions.Categorical(logits=tok_logits).sample()  # [B]
            else:
                new_tok = tok_logits.argmax(dim=-1)                                   # [B]

            x[torch.arange(B, device=x.device), z] = new_tok
            M = x.eq(self.mask_id)

            steps  += 1
        return x # [B, L]
    
    def create_attention_mask_per_batch(self, x):
        """
        Create per-batch attention masks
        Returns: [B, L, L] or [B*num_heads, L, L]
        """
        B, L = x.shape
        is_masked = (x == self.mask_id)  # [B, L]
        
        # Create mask for each sample: [B, L, L]
        # mask[b, i, j] = can position i attend to position j?
        mask = is_masked.unsqueeze(1) & is_masked.unsqueeze(2)  # [B, L, L]
        
        # Reshape for multi-head attention: [B*num_heads, L, L]
        mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        mask = mask.view(B * self.n_heads, L, L)
        
        # Convert to float
        attn_mask_float = torch.zeros_like(mask, dtype=torch.float)
        attn_mask_float.masked_fill_(mask, float('-inf'))
        
        return attn_mask_float


        




