import torch
import torch.nn as nn
import torch.nn.functional as F

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
                                                dim_feedforward=4 * self.d_model, 
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
        return h 

    def encode_with_causal_mask(self, batch):
        """Encode with proper causal attention mask"""
        B, L = batch.shape
        pos = torch.arange(L, device=batch.device).unsqueeze(0).expand((B, L))
        emb = self.tok_emb(batch) + self.pos_emb(pos)
        
        # Create causal mask: position i can only attend to positions <= i
        causal_mask = torch.triu(torch.ones(L, L, device=batch.device), diagonal=1).bool()
        causal_mask = causal_mask.float().masked_fill(causal_mask, float('-inf'))
        
        h = self.transformer(emb, mask=causal_mask)
        return h

    @torch.no_grad()
    def generate(self, batch_size, seq_len, temperature=1.0, 
            sample_order=True, sample_token=True):

        x = torch.full((batch_size, seq_len), self.mask_id, dtype = torch.long, device = self.device)
        l = []
        for step in range(seq_len):
            h = self.encode(x)

            token_logits = self.token_head(h)
            order_logits = self.order_head(h).squeeze(dim = -1)
            mask = (x == self.mask_id)
            if not mask.any():
                break
            masked_order_logits = order_logits.masked_fill(~mask, float('-inf'))
            if sample_order:
                probs = F.softmax(masked_order_logits/temperature, dim = -1)
                next_pos = torch.multinomial(probs, num_samples = 1).squeeze(dim = -1)
            else:
                next_pos = masked_order_logits.argmax(dim = -1)
            #next_pos += 1

            selected_tokens = token_logits[torch.arange(batch_size), next_pos]

            if sample_token:
                probs = F.softmax(selected_tokens/temperature, dim = -1)
                next_token = torch.multinomial(probs, num_samples = 1).squeeze(dim = -1)
            else:
                next_token = selected_tokens.argmax(dim = -1)

            x[torch.arange(batch_size), next_pos] = next_token
            l.append(next_pos)


        return {"outputs": x, "orders": l}

    @torch.no_grad()
    def generate_ar(self, batch_size, seq_len, temperature=1.0):
        """
        Standard autoregressive generation for causal-trained models.
        """
        self.eval()
        
        # Start with BOS token or random token
        x = torch.full((batch_size, 1), self.eos_id, dtype=torch.long, device=self.device)
        
        for step in range(seq_len - 1):
            # Encode current sequence with causal mask
            h = self.encode_with_causal_mask(x)
            
            # Get logits for the last position only
            token_logits = self.token_head(h[:, -1, :])  # [batch_size, vocab_size]
            
            # Sample next token
            probs = F.softmax(token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            
            # Append to sequence
            x = torch.cat([x, next_token], dim=1)
        
        return {"outputs": x, "orders": list(range(seq_len))}
    
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