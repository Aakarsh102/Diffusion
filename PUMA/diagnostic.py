# ---------- DIAGNOSTIC SNIPPET ----------
import torch, sys, pprint

# objects from your script: `to` (tokenizer), `model` (MDM model instance),
# `model_config` or CONFIGS['tinygsm']['model'], and mask_id variable.
print("==== BASIC SIZES ====")
try:
    tok_vocab = getattr(to, "vocab_size", None) or len(to.get_vocab())
except Exception:
    tok_vocab = None
print("tokenizer vocab_size:", tok_vocab)
mc_vocab = CONFIGS["tinygsm"]["model"]["vocab_size"]
print("model config vocab_size:", mc_vocab)
print("CONFIGS tinygsm mask_id:", CONFIGS["tinygsm"]["mask_id"])
print("mask_id variable (if set):", globals().get("mask_id", "<not set>"))

print("\n==== SPECIAL TOKENS ====")
for attr in ("mask_token","mask_token_id","pad_token_id","unk_token_id","eos_token_id"):
    print(attr, "=", getattr(to, attr, None))

print("\n==== DECODE TROUBLE ID 151643 ====")
try:
    print("to.decode([151643]):", to.decode([151643], skip_special_tokens=False))
except Exception as e:
    print("decode failed:", e)
try:
    print("to.convert_ids_to_tokens(151643):", to.convert_ids_to_tokens(151643))
except Exception as e:
    print("convert_ids_to_tokens failed:", e)

print("\n==== CHECKPOINT KEYS MISMATCH ====")
# If you already have ckpt dict `ckpt` from torch.load; otherwise
# load a checkpoint path: ckpt = torch.load(path, map_location='cpu')
if "ckpt" in globals():
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing keys (when loading checkpoint into current model):")
    pprint.pprint(missing)
    print("unexpected keys:")
    pprint.pprint(unexpected)
else:
    print("No 'ckpt' object in globals — skip load-state check.")

print("\n==== LOGITS / TOP-K AT ONE MASKED POSITION ====")
model_to_sample = model.module if hasattr(model, "module") else model
device = next(model_to_sample.parameters()).device
L = getattr(model_to_sample, "max_position", getattr(model_to_sample, "config", {}).get("max_position", None)) or CONFIGS['tinygsm']['seq_len']
xt_debug = torch.full((1, L), CONFIGS['tinygsm']['mask_id'], dtype=torch.long, device=device)
with torch.no_grad():
    logits = model_to_sample(xt_debug)  # shape [1, L, V]
    pos0_logits = logits[0, 0]          # inspect first position
    topk = torch.topk(pos0_logits, k=20)
    print("top-20 token ids at pos0:", topk.indices.tolist())
    probs = torch.softmax(pos0_logits, dim=-1)
    print("top-20 probs:", probs[topk.indices].tolist())
    print("entropy:", -(probs * torch.log(probs + 1e-12)).sum().item())
print("==== END DIAGNOSTIC ====")
