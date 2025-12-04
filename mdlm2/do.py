# download_esm2.py
import os

# SET BEFORE ANY IMPORTS
os.environ['HF_HOME'] = '/scratch/gilbreth/rai53/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/gilbreth/rai53/hf_cache'
os.environ['HF_HUB_CACHE'] = '/scratch/gilbreth/rai53/hf_cache'

import transformers

save_dir = '/scratch/gilbreth/rai53/esm2_650M'
os.makedirs(save_dir, exist_ok=True)

print("Downloading ESM-2 model...")
model = transformers.EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

print(f"Saving to {save_dir}...")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("Done!")
