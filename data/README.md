# Current Data

The formal calibration dataset is [eval/gold/](eval/gold/). There is one current gold dataset, without parallel v1/v2 directories. Paths in the table below are relative to data/. Run commands from the repository root.

For the complete dataset descriptions, production lineage, ID joins and rebuild limitations, see [数据说明与生产链路](../docs/data_pipeline.md).

| Path | Contents | Shared in Git |
|---|---|---|
| `eval/gold/product_items.jsonl` | 100 frozen ProductItems | Yes |
| `eval/gold/results_<model>.jsonl` | Four complete GenerationResult files, 100 candidates each | Yes |
| `eval/gold/candidate_pairs.jsonl` | 100 frozen CandidatePairs, one per product | Yes |
| `eval/gold/pairwise_annotations_zhou.jsonl` | 100 current human labels | Yes |
| `eval/gold/rubric_sample.jsonl` | 130 candidates frozen for rubric human labels: all 100 qwen_local, plus 10 each from the other three models on 30 distinct products; shuffled; built by `scripts/build_rubric_sample.py` | Yes |
| `eval/gold/rubric_sample.manifest.json` | Rubric sample inputs and hashes, seed and generator counts | Yes |
| `eval/gold/rubric_annotations_zhou.jsonl` | 130 current human rubric labels on `rubric_sample` (`factual_support`, protocol `rubric-v0.2`); 17 false | Yes |
| `eval/gold/manifest.json` | Counts, schemas, hashes, generator models, prompt and inference settings | Yes |
| `eval/gold/candidate_pairs.manifest.json` | Pair sources and hashes, sampling seed and model combinations | Yes |
| `splits/v1/` | Frozen gold 100, test 1,000 and partition metadata | Yes |
| `processed/v1/` | Original source-format records for gold 100 and test 1,000 | Yes |
| `train/` | Future curated training datasets; currently empty | Yes when published |

The four candidate files are `results_qwen_local.jsonl`, `results_gemma_31b.jsonl`, `results_glm53_flash.jsonl`, and `results_longcat.jsonl`. All 400 candidates are available to both pairwise and rubric work. The 100 pairs contain only 200 of those candidates. Join by `product_id` and `candidate_id`; human presentation letters are randomized and are not canonical candidate identities.

Gold and test are disjoint evaluation products, and both are excluded from training. The 709,403-record train pool is a candidate pool, not a decision to train on every product. Validation will be selected from that pool. Test products are currently retained in source format; a standard ProductItem test-generation dataset has not yet been materialized.

The `v1` in splits/processed identifies the one frozen source partition. It does not mean that the old candidate generations are still active. Product, candidate and pair IDs, ad text and current human winners were preserved during organization.

## Raw source

The Office Products metadata contains 710,503 records. Its local download currently lives at `../data/raw/amazon_reviews_2023/meta_categories/meta_Office_Products.jsonl.gz`, outside the repository. Raw downloads can be kept outside the repo or under artifacts/cache/. The large train-pool index is now artifacts/cache/office_products/train_pool.jsonl; data/splits/v1/manifest.json records its current hash and original source representation. The split manifest records the raw gzip SHA-256 and source-line references. The large raw gzip and full train-pool index are not uploaded to ordinary Git. Use `scripts/build_data_split.py --help` and [the split method](../docs/data_split.md) to reconstruct the split from a matching source download.

## Offline validation

From the repository root, after installing dependencies:

```sh
PYTHONPATH=src .venv/bin/python scripts/validate_gold_data.py
```

This validates schemas, hashes, all 400 candidate identities, product coverage, pair sampling and human winner mapping. It also recomputes the four published judges' agreement against the current labels. It requires no API key, HTTP call log or original large raw download.

## Formal results, local runtime and archive

The current four-judge statistics are in [results/pairwise/comparison/](../results/pairwise/comparison/). Formal judge results and comparison files are shared under results/. Full calls.jsonl journals remain local at artifacts/pairwise/<run_id>/. Generation runtime snapshots/logs stay in `artifacts/generation/gold/`; the published frozen dataset is the copy here in data, protected by manifest hashes.

Original human labels, the four review batches covering 15 revised winners, older candidates/pairs, and prompt pilots are under [deprecated/](../deprecated/), following the project's archive convention. Original labels and review history are local only and ignored by Git; only the final 100 human labels are shared. Current labels include user-confirmed model-assisted review; this is one annotator's calibration reference, not independent two-person consensus. A later label edit must be reflected in the dataset manifest and comparison statistics before publishing it as a new frozen snapshot.

The Git boundary is directory-based: data/ contains curated shared inputs, results/ contains formal shared outcomes, and artifacts/ contains ignored runtime outputs, journals, caches and weights. No model-specific artifact whitelist is required.
