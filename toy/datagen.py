# Create a synthetic "checksum" dataset with 10 predefined orders (templates).
# We generate 50,000 examples total (5,000 per order). Each example is length-10:
# [a,b,c,d,e,[SEP],s1,s2,s3,s4] where s1=(a+b)%10, s2=(c+d)%10, s3=(a+c+e)%10, s4=(a+b+c+d+e)%10.
# We annotate each example with a template (order) id/name/permutation.

import json
import numpy as np
import pandas as pd
from pathlib import Path
# from caas_jupyter_tools import display_dataframe_to_user

rng = np.random.default_rng(20250919)

SEQ_LEN = 10

# Define 10 order templates over positions 1..10 (inclusive)
# Positions: 1..5 are bases, 6 is [SEP], 7..10 are sums
# We'll include some sensible and some "adversarial" orders
templates = [
    ("left_to_right",        [1,2,3,4,5,6,7,8,9,10]),
    ("right_to_left",        [10,9,8,7,6,5,4,3,2,1]),
    ("bases_then_sums_mix",  [1,3,2,5,4,6,7,9,8,10]),       # bases first (scrambled), then sums
    ("sums_then_bases",      [7,8,9,10,6,1,2,3,4,5]),       # "bad" order (hard first)
    ("interleave_spread",    [1,7,2,8,3,9,4,10,5,6]),
    ("reverse_interleave",   [10,5,9,4,8,3,7,2,6,1]),
    ("odds_then_evens",      [1,3,5,7,9,2,4,6,8,10]),
    ("evens_then_odds",      [2,4,6,8,10,1,3,5,7,9]),
]

# Add two fixed random permutations for diversity
def random_perm_fixed(seed_offset: int):
    perm = np.arange(1, SEQ_LEN+1)
    rng_local = np.random.default_rng(20250919 + seed_offset)
    rng_local.shuffle(perm)
    return perm.tolist()

templates.append(("random_perm_1", random_perm_fixed(1)))
templates.append(("random_perm_2", random_perm_fixed(2)))

assert len(templates) == 10

# Generator for one example
def sample_example():
    # Sample bases a..e uniformly from digits 0..9
    a, b, c, d, e = rng.integers(0, 10, size=5).tolist()
    s1 = (a + b) % 10
    s2 = (c + d) % 10
    s3 = (a + c + e) % 10
    s4 = (a + b + c + d + e) % 10
    tokens = [str(a), str(b), str(c), str(d), str(e), "[SEP]", str(s1), str(s2), str(s3), str(s4)]
    return dict(
        bases=[a,b,c,d,e],
        sums=[s1,s2,s3,s4],
        tokens=tokens,
        text=" ".join(tokens),
        task="checksum_5_4",
        vocab="digits+SEP",
        seq_len=SEQ_LEN,
    )

# Build dataset with equal counts per order
EXAMPLES_PER_ORDER = 5000
TOTAL = EXAMPLES_PER_ORDER * len(templates)

rows = []
for order_id, (name, perm) in enumerate(templates, start=1):
    for _ in range(EXAMPLES_PER_ORDER):
        ex = sample_example()
        rows.append({
            "order_id": order_id,
            "order_name": name,
            "order": perm,
            **ex
        })

df = pd.DataFrame(rows)

# Save to disk
out_dir = Path("data/")
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "checksum10_orders_50k.csv"
jsonl_path = out_dir / "checksum10_orders_50k.jsonl"
orders_path = out_dir / "orders_catalog.json"
readme_path = out_dir / "README_checksum10_orders.md"

df.to_csv(csv_path, index=False)

with open(jsonl_path, "w") as f:
    for _, row in df.iterrows():
        # Serialize row with 'order' as list, tokens as list, bases/sums as list
        obj = {
            "order_id": int(row["order_id"]),
            "order_name": row["order_name"],
            "order": list(row["order"] if isinstance(row["order"], (list, tuple, np.ndarray)) else eval(row["order"])),
            "tokens": row["tokens"] if isinstance(row["tokens"], list) else row["tokens"].split(" "),
            "text": row["text"],
            "bases": row["bases"] if isinstance(row["bases"], list) else eval(row["bases"]),
            "sums": row["sums"] if isinstance(row["sums"], list) else eval(row["sums"]),
            "task": row["task"],
            "vocab": row["vocab"],
            "seq_len": int(row["seq_len"]),
        }
        f.write(json.dumps(obj) + "\n")

with open(orders_path, "w") as f:
    catalog = [{"order_id": i+1, "order_name": name, "order": perm} for i, (name, perm) in enumerate(templates)]
    json.dump(catalog, f, indent=2)

with open(readme_path, "w") as f:
    f.write(
        "# Checksum Toy Dataset (10 Orders)\n\n"
        "## Sequence format (len=10)\n"
        "`a b c d e [SEP] s1 s2 s3 s4`\n\n"
        "where:\n"
        "- `a..e` ~ Uniform{0..9}\n"
        "- `s1=(a+b) mod 10`, `s2=(c+d) mod 10`, `s3=(a+c+e) mod 10`, `s4=(a+b+c+d+e) mod 10`\n\n"
        "## Orders\n"
        "10 template permutations over positions 1..10. See `orders_catalog.json`.\n\n"
        "## Files\n"
        "- `checksum10_orders_50k.csv` — 50,000 rows, one example per line\n"
        "- `checksum10_orders_50k.jsonl` — JSONL with lists preserved for tokens/order\n"
        "- `orders_catalog.json` — list of the 10 order templates\n\n"
        "## Columns (CSV/JSONL)\n"
        "- `order_id` (1..10), `order_name`, `order` (list[int])\n"
        "- `tokens` (list[str]), `text` (space-separated)\n"
        "- `bases` (list[int]), `sums` (list[int])\n"
        "- `task`, `vocab`, `seq_len`\n"
    )

# Show a small preview to the user
#display_dataframe_to_user("Preview: checksum10_orders (first 100 rows)", df.head(100))

csv_path.as_posix(), jsonl_path.as_posix(), orders_path.as_posix(), readme_path.as_posix()
