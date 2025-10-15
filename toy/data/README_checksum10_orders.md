# Checksum Toy Dataset (10 Orders)

## Sequence format (len=10)
`a b c d e [SEP] s1 s2 s3 s4`

where:
- `a..e` ~ Uniform{0..9}
- `s1=(a+b) mod 10`, `s2=(c+d) mod 10`, `s3=(a+c+e) mod 10`, `s4=(a+b+c+d+e) mod 10`

## Orders
10 template permutations over positions 1..10. See `orders_catalog.json`.

## Files
- `checksum10_orders_50k.csv` — 50,000 rows, one example per line
- `checksum10_orders_50k.jsonl` — JSONL with lists preserved for tokens/order
- `orders_catalog.json` — list of the 10 order templates

## Columns (CSV/JSONL)
- `order_id` (1..10), `order_name`, `order` (list[int])
- `tokens` (list[str]), `text` (space-separated)
- `bases` (list[int]), `sums` (list[int])
- `task`, `vocab`, `seq_len`
