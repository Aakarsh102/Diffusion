import torch
import torch.nn as nn
import torch.nn.functional as F

# orders = (B, seq, 10)
# time_step = int 

    

# class Modelthing(nn.Module):
#     def __init__(self, device, d_model, n_layers, n_heads, vocab_size, mask_id = 0, max_len = 1024):
#         super().__init__() 
#         self.n_layers = n_layers
#         self.n_heads = n_heads
#         self.device = device
#         # self.emb_dim = emb_dim
#         self.d_model = d_model
#         self.order_weight = 1
#         # self.orders = 10
#         #self.seq_len = seq_len
#         self.vocab_size = vocab_size
#         self.mask_id = mask_id
#         self.max_len = max_len
#         self.layer = nn.TransformerEncoderLayer(d_model=self.d_model, 
#                                                 nhead = self.n_heads, 
#                                                 dim_feedforward=128, 
#                                                 device=self.device,
#                                                 batch_first=True
#                                                 )
#         self.transformer = nn.TransformerEncoder(self.layer, self.n_layers)

#         self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
#         self.pos_emb = nn.Embedding(self.max_len, self.d_model)

#         self.token_head = nn.Linear(self.d_model, self.vocab_size)
#         self.order_head = nn.Linear(self.d_model, 1)
#         if variational_policy == "shared_torso":
#             # Shares transformer, just adds another head
#             self.variational_order_head = nn.Linear(self.d_model, 1)
#         elif variational_policy == "separate":
#             # Separate smaller transformer for qθ
#             var_layer = nn.TransformerEncoderLayer(
#                 d_model=self.d_model, 
#                 nhead=self.n_heads,
#                 dim_feedforward=128,
#                 device=self.device,
#                 batch_first=True
#             )
#             self.var_transformer = nn.TransformerEncoder(var_layer, num_layers=max(1, self.n_layers // 2))
#             self.variational_order_head = nn.Linear(self.d_model, 1) 

#     def get_variational_logits(self, x):
#         """
#         Get qθ logits. Input x should be FULLY UNMASKED.
#         Returns: [B, L] logits for all positions
#         """
#         B, L = x.shape
        
#         if self.variational_policy == "shared_torso":
#             # Use main transformer
#             h = self.encode(x, use_mask=False)
#             return self.variational_order_head(h).squeeze(-1)  # [B, L]
#         else:
#             # Use separate transformer
#             pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
#             emb = self.tok_emb(x) + self.pos_emb(pos)
#             h = self.var_transformer(emb)
#             return self.variational_order_head(h).squeeze(-1)  # [B, L]

#     def sample_ordering_gumbel(self, q_logits):
#         """
#         Sample a full permutation using Gumbel-top-k trick.
#         Args:
#             q_logits: [B, L] logits from variational policy
#         Returns:
#             ordering: [B, L] where ordering[b, :] is a permutation of [0, 1, ..., L-1]
#         """
#         gumbel = -torch.log(-torch.log(torch.rand_like(q_logits) + 1e-10) + 1e-10)
#         perturbed_logits = q_logits + gumbel
#         ordering = torch.argsort(perturbed_logits, dim=-1, descending=True)
#         return ordering 

#     def encode(self, batch, use_mask = True):
#         #Batch, seq_len
#         B, L = batch.shape
#         pos = torch.arange(L, device = batch.device).unsqueeze(0).expand((B, L))
#         emb = self.tok_emb(batch) + self.pos_emb(pos)  # [B, L, d_model]
#         if use_mask:
#             attn_mask = self.create_attention_mask_per_batch(batch)
#             #h = self.transformer(emb, mask=attn_mask)
#             h = self.transformer(emb)
#         else:
#             h = self.transformer(emb) 
#         return h

#     def forward(self, x_in, x_target):
#         h = self.encode(x_in)
#         logits_tok = self.token_head(h)             # [B, L, V]
#         scores_ord = self.order_head(h).squeeze(-1)
#         M = x_in.eq(self.mask_id)

#         tok_loss = F.cross_entropy(
#             logits_tok[M], x_target[M]
#         ) if M.any() else torch.zeros((), device=x_in.device)

#         with torch.no_grad():

# orders = (B, seq, 10)
# time_step = int 

    

class Model(nn.Module):
    def __init__(self, device, d_model, n_layers, n_heads, vocab_size, variational_policy = "shared_torso", mask_id = 0, max_len = 1024):
        super().__init__() 
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.device = device
        # self.emb_dim = emb_dim
        self.d_model = d_model
        self.order_weight = 1
        # self.orders = 10
        #self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.max_len = max_len
        self.variational_policy = variational_policy
        self.layer = nn.TransformerEncoderLayer(d_model=self.d_model, 
                                                nhead = self.n_heads, 
                                                dim_feedforward=128, 
                                                device=self.device,
                                                batch_first=True
                                                )
        self.transformer = nn.TransformerEncoder(self.layer, self.n_layers)

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)

        self.token_head = nn.Linear(self.d_model, self.vocab_size)
        self.order_head = nn.Linear(self.d_model, 1)
        if variational_policy == "shared_torso":
            # Shares transformer, just adds another head
            self.variational_order_head = nn.Linear(self.d_model, 1)
        elif variational_policy == "separate":
            # Separate smaller transformer for qθ
            var_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model, 
                nhead=self.n_heads,
                dim_feedforward=128,
                device=self.device,
                batch_first=True
            )
            self.var_transformer = nn.TransformerEncoder(var_layer, num_layers=max(1, self.n_layers // 2))
            self.variational_order_head = nn.Linear(self.d_model, 1) 

    def get_variational_logits(self, x):
        """
        Get qθ logits. Input x should be FULLY UNMASKED.
        Returns: [B, L] logits for all positions
        """
        B, L = x.shape
        
        if self.variational_policy == "shared_torso":
            # Use main transformer
            h = self.encode(x, use_mask=False)
            return self.variational_order_head(h).squeeze(-1)  # [B, L]
        else:
            # Use separate transformer
            pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
            emb = self.tok_emb(x) + self.pos_emb(pos)
            h = self.var_transformer(emb)
            return self.variational_order_head(h).squeeze(-1)  # [B, L]

    def sample_ordering_gumbel(self, q_logits):
        """
        Sample a full permutation using Gumbel-top-k trick.
        Args:
            q_logits: [B, L] logits from variational policy
        Returns:
            ordering: [B, L] where ordering[b, :] is a permutation of [0, 1, ..., L-1]
        """
        gumbel = -torch.log(-torch.log(torch.rand_like(q_logits) + 1e-10) + 1e-10)
        perturbed_logits = q_logits + gumbel
        ordering = torch.argsort(perturbed_logits, dim=-1, descending=True)
        return ordering 
    
    def encode(self, batch, use_mask = True):
        #Batch, seq_len
        B, L = batch.shape
        pos = torch.arange(L, device = batch.device).unsqueeze(0).expand((B, L))
        emb = self.tok_emb(batch) + self.pos_emb(pos)  # [B, L, d_model]
        if use_mask:
            attn_mask = self.create_attention_mask_per_batch(batch)
            #h = self.transformer(emb, mask=attn_mask)
            h = self.transformer(emb)
        else:
            h = self.transformer(emb)

    # def encode(self, batch, use_mask = True):
    #     #Batch, seq_len
    #     B, L = batch.shape
    #     pos = torch.arange(L, device = batch.device).unsqueeze(0).expand((B, L))
    #     emb = self.tok_emb(batch) + self.pos_emb(pos)  # [B, L, d_model]
    #     finished = finished | eos_generated
        
    #     # Replace remaining masks with pad token for unfinished sequences
    #     still_masked = (x == self.mask_id)
    #     x[still_masked] = pad_token_id
        
    #     # For sequences that never generated EOS, set length to max_len
    #     lengths[lengths == 0] = max_len
        
    #     return x, lengths
                 

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
            # z is the index to unmask in the current generation step
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