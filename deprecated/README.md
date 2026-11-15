# Deprecated Material

This directory contains material retired from the active experiment and documentation paths on 2026-11-14. The current dataset is `data/eval/gold/`; current judge comparison statistics are `results/pairwise/comparison/`.

| Location | What was retired |
|---|---|
| `data/eval/gold/v1/` | Previous long-ad CandidatePairs and duplicated ProductItems |
| `data/eval/gold/v2/candidate_pairs.manifest.json` | Original construction manifest with captured absolute paths |
| `data/eval/gold/pairwise_annotations_zhou.original.jsonl` | Original human labels before the 15 confirmed revisions; local only, ignored by Git |
| `artifacts/eval/gold_review/` | Four review batches, before-snapshots and user-confirmed corrections; local only, ignored by Git |
| `artifacts/eval/gold_generation/` | Previous generation, prompt pilots, pre-retry backups and retired retry manifest |
| `artifacts/eval/pairwise/` | Judge pilots, old Qwen protocol run, original-label metrics and the previous detailed review analysis |
| `prompts/eval/pairwise/` | Retired product-grounded judge v1 prompt |
| `docs/` | Retired generation guide and pre-cleanup documentation snapshots |
| `data/raw/amazon_reviews_2023/` | Unused All Beauty raw download, retained locally |

[migration_manifest.json](migration_manifest.json) maps old paths to preserved files and records SHA-256 hashes. Archived manifests, call logs and review records retain their originally captured paths and content; use the migration mapping when locating moved inputs. They are historical evidence, not current runnable configuration.

Only the final 100 human labels in data/eval/gold/pairwise_annotations_zhou.jsonl are shared. Original human labels, intermediate annotation snapshots and review records remain local and are ignored by Git. Small old inputs, archived docs and the migration index can be committed. Bulky obsolete generation/judge outputs and the unused raw download remain ignored. Current call logs are still under artifacts, not in the formal dataset. No generated ad, candidate identity, pair or current human winner was changed as part of the reorganization.

## Subsequent results/ split

The follow-up layout change separates shared formal results/ from ignored runtime artifacts/. Its before-snapshots, file relocation hashes and cache-format audit are in [layout_migrations/data_results_artifacts_20261114/](layout_migrations/data_results_artifacts_20261114/). Original run records were not regenerated. Historical captured paths in archived records are intentional; use the successive migration maps to resolve moved files.
