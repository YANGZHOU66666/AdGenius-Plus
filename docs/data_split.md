# Office Products split v1

This split freezes the evaluation boundary before any generation, judging, or
training work:

- `gold.jsonl`: 100 products reserved for the shared human calibration set.
- `test.jsonl`: 1,000 products reserved for the final frozen test set.
- `artifacts/cache/office_products/train_pool.jsonl`: every remaining product (709,403 records), available as
  a candidate pool for later training experiments.

`train_pool` is deliberately larger than any planned training run. A later
experiment may derive a smaller `val` or training dataset from it, but must not
read `gold` or `test`. There is no validation split in v1 because validation
size and checkpoint-selection rules depend on the eventual training strategy.
The full train-pool manifest is generated locally and ignored by Git because it
is about 208 MiB; its exact source and output hashes are recorded in the
manifest, so it can be regenerated without changing the split.

The source is the Amazon Reviews 2023 `Office Products` metadata file. The
split uses seed `20261028` and groups records by a SHA-256 digest of the
generation-relevant fields (`title`, `features`, `description`, `details`,
and related product metadata), excluding `parent_asin`. This keeps exact
content duplicates in one split. The source has no reliable variant/family
relation field, so brands, manufacturers, and model numbers are not treated as
families. `data/splits/v1/manifest.json` records the source hash, counts,
missing-field statistics, grouping method, and output hashes.

The two files in `data/processed/v1/` contain the original product objects for
gold and test only. They do not contain generated copy or human labels. The
local train pool manifest points back to the immutable raw source by
`source_line` and `product_id`, so it does not duplicate the 2+ GB decompressed
dataset.

Rebuild from the repository root:

```sh
PYTHONPATH=src .venv/bin/python scripts/build_data_split.py
```

To reproduce a different explicit version, use a different output directory or
copy the script and record the new seed. Do not overwrite `v1` after evaluation
has started.

## Shared partition and local cache

Gold/test manifests stay under data/splits/v1/. The full train pool is a local cache under artifacts/cache/office_products/. Its project IDs are train_product_000000 onward, with source_id preserving the original ASIN; the split builder now writes the same representation to that cache location.

The pre-cleanup manifest recorded the hash of the original ASIN-only representation. The existing cache had already been materialized with project IDs and source_id before this move. Restoring those two fields to the original representation reproduces the historical hash exactly. Current metadata records the actual cache hash and retains the earlier hash in cache_provenance; no cache rows were regenerated or changed during organization. Original metadata and static audit details are in deprecated/layout_migrations/data_results_artifacts_20261114/.
