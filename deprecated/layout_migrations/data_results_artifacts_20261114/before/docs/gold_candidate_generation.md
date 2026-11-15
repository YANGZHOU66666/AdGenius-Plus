# Frozen Gold Candidate Generation

The shared dataset is `data/eval/gold/`, documented in [data/README.md](../data/README.md). This is the one current set of 400 candidates for human pairwise and rubric annotation.

## Collection

The input is 100 fixed Office Products ProductItems. Four generators each produce one English ad per product:

| Alias | Generator |
|---|---|
| `qwen_local` | Local Qwen3.5 2B (`qwen3.5:2b`) |
| `gemma_31b` | Gemma 4 31B through the Google API |
| `glm53_flash` | GLM-5.3-Flash through BigModel |
| `longcat` | LongCat-2.5 |

A shared generation prompt requests concise, product-grounded English promotional copy, selecting information useful for a short ad rather than covering every attribute. It requests one or two sentences, preferably fewer than 25 words and no more than 35 words. These are prompt instructions: complete responses are not automatically truncated to a word limit or quality-filtered. The frozen manifest stores the actual prompt, models, run IDs, inference controls, file hashes and counts.

The input rendering uses title, categories, features, description and selected details. It excludes prices, ratings, media and other unsuitable metadata. Rendering is implemented in `src/adgenius/shared/product.py` and reused by the annotation workbench. ProductItem records retain their complete original product_info regardless of which fields are rendered.

All four final files contain 100 successful GenerationResults. API failures were repaired using the saved request settings, while successful candidates and candidate IDs were preserved. Call journals keep original failed attempts as well as retries. Standard result files, not historical failures in journals, determine final completion.

## Frozen files

- `data/eval/gold/results_<alias>.jsonl`: four shared, frozen GenerationResult files; 400 candidates total.
- `artifacts/eval/gold_generation/v2/results_<alias>.jsonl`: original local runtime snapshots from the generation batch.
- `artifacts/eval/gold_generation/v2/calls_<alias>.jsonl`: local call journals with actual prompts, requests, responses, usage and retry traces.

The `v2` in the local runtime log directory is a historical batch name. There is no parallel v2 subdirectory in the formal gold dataset. Old generations and pre-retry backups are archived in deprecated.

## New generation and repair

The existing 400 candidates need no new API calls. To run a separate experiment, choose a new output directory:

```sh
PYTHONPATH=src .venv/bin/python scripts/generate_candidates.py   --input data/eval/gold/product_items.jsonl   --output-dir artifacts/eval/gold_generation/new_run
```

The script refuses existing results/calls paths. Its default input is the formal ProductItem file; its historical default output remains `artifacts/eval/gold_generation/v2/`. New runtime outputs do not overwrite or automatically replace the frozen data in data/eval/gold.

Repair reads actual historical settings from a runtime directory. The completed current batch has zero remaining failures:

```sh
PYTHONPATH=src .venv/bin/python scripts/retry_failed_candidates.py --dry-run
# For a separate unfinished generation run:
PYTHONPATH=src .venv/bin/python scripts/retry_failed_candidates.py   --source-dir artifacts/eval/gold_generation/new_run
```

## Pair construction

For each of 100 products, sample two distinct generator aliases uniformly with seed 42. Per-product random seeds are derived from canonical `[seed, product_id]`; candidate array order is canonical by candidate ID, not a visible A/B preference. The existing 100 frozen pairs are preserved byte-for-byte.

To reproduce pairs into a temporary output:

```sh
PYTHONPATH=src .venv/bin/python scripts/build_candidate_pairs.py   --input data/eval/gold/results_qwen_local.jsonl           data/eval/gold/results_gemma_31b.jsonl           data/eval/gold/results_glm53_flash.jsonl           data/eval/gold/results_longcat.jsonl   --products data/eval/gold/product_items.jsonl   --seed 42 --output /tmp/adgenius_candidate_pairs.jsonl
```

Both input provenance and output hash are saved in candidate_pairs.manifest.json. Repository inputs use portable relative paths. Pair sampling is a separate offline step; all 400 candidates remain available for rubric annotation.
