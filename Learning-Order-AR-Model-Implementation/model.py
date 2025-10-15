import torch
import torch.nn as nn
import torch.nn.functional as F

# orders = (B, seq, 10)
# time_step = int 

    

class Model(nn.Module):
    def __init__(self, device, emb_dim, seq_len, vocab_size, mask_id = 0, max_len = 1024):
        self.n_layers = 2
        self.n_heads = 2
        self.device = device
        self.emb_dim = emb_dim
        self.d_model = self.emb_dim
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

        self.value_head = nn.Linear(self.d_model, self.seq_len)
        self.order_head = nn.Linear(self.d_model, 1)

    def encode(self, batch):
        #Batch, seq_len
        B, L = batch.size
        pos = torch.arange(L, device = batch.device).unsqueeze(0).expand((B, L))
        h = self.transformer(self.tok_emb(batch) + self.pos_emb(pos))
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
        order_probs = masked_ords.log_softmax(-1) # [B, L]
        order_loss = F.kl_div(order_probs, target, reduction='batchmean')

        total_loss = tok_loss + self.order_weight * order_loss
        return {"loss": total_loss, "token_loss": tok_loss.detach(), "order_loss": order_loss.detach()}
    
    def generate(self, x_init, max_steps, temperature=1.0,
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
                z = torch.distributions.Categorical(logits=order_logits/temperature).sample()   # [B]
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


        




