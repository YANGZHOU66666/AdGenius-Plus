# Formal Experiment Results

This directory contains standard per-sample results, decisions, configuration/provenance manifests and published comparison reports. It is shared in Git together with data/. Full HTTP calls, generation process files, caches and weights live under ignored artifacts/.

The [data pipeline guide](../docs/data_pipeline.md) connects these results to their input datasets and describes how judgments, decisions and comparison statistics were produced, including current rebuild limitations.

## Current pairwise outputs

| Judge | Standard result file | Local journal |
|---|---|---|
| Qwen Flash | pairwise/qwen37_gold_v2/results.jsonl | artifacts/pairwise/qwen37_gold_v2/calls.jsonl |
| DeepSeek | pairwise/deepseek41_gold_v2/results.jsonl | artifacts/pairwise/deepseek41_gold_v2/calls.jsonl |
| Claude Haiku | pairwise/claude55_gold_v2/results.jsonl | artifacts/pairwise/claude55_gold_v2/calls.jsonl |
| GPT-6 Luna | pairwise/gpt6_luna_gold_v2/results.jsonl | artifacts/pairwise/gpt6_luna_gold_v2/calls.jsonl |

Each result file has 100 PairwiseResult records with independent AB and BA judgments. decisions.jsonl contains the order-consistency outcome. Original run manifests preserve their captured provenance and were not rewritten during relocation. The runtime journals are optional for reading the shared results, but are required when resuming that exact run.

[Current comparison](pairwise/comparison/report.md) uses the current human labels from data/eval/gold. metrics.json contains the existing four-model metrics and uncertainty estimates; manifest.json records the current input/result locations and hashes. No scores or metrics were recomputed for the directory reorganization. Original labels/review records and earlier-label statistics remain under deprecated/.

## Training experiments

| Experiment | Contents | Report |
|---|---|---|
| training/opd_lr_sweep_v1 | OPD learning-rate and beta selection on 200 dev products: 7 GenerationResult files (untrained student and 6 runs), CandidatePairs, per-run manifests, selection | [report](training/opd_lr_sweep_v1/report.md) |
| training/sft_lr_sweep_v1 | SFT learning-rate selection on the same 200 dev products: 3 GenerationResult files (the untrained student is shared with opd_lr_sweep_v1), CandidatePairs, per-run manifests, selection | [report](training/sft_lr_sweep_v1/report.md) |
| training/sft_formal_v1, training/opd_formal_v1 | Formal runs to 8k with the selected settings: GenerationResult files at 1k/2k/4k/8k on test (1,000) and dev (200) under generations/{test,dev}/, the per-step training log, the Drive checkpoint listing (sizes and md5) and a manifest | [report](training/formal_v1_report.md) |
| training/test_references_v1 | Reference systems on the 1,000 test products: the 397B teacher and the untrained student (dev references are the existing files named in its manifest) | [report](training/formal_v1_report.md) |

Their DeepSeek judge runs are under pairwise/opd_sweep_dev_*/ and pairwise/sft_sweep_dev_*/ in the same layout as the gold judge runs, with call journals in artifacts/pairwise/<run_id>/. The scripts are in scripts/opd/ and scripts/sft/ (training code) with lr_sweep/ subfolders (sweep orchestration); the formal runs are orchestrated by scripts/formal/. Training logs, VM archives, frozen prompts or SFT examples and the exact code versions that ran stay in ignored artifacts/train/<experiment>/; each experiment manifest records their hashes. Sweep checkpoints were not retained; the formal runs' checkpoints are on Google Drive (gdrive:adgenius_checkpoints/<run>/), listed in each run's drive_checkpoints.json.

## Future runs

Pairwise evaluation defaults to results/pairwise/<run_id>/ for results, decisions, metrics and the run manifest, and artifacts/pairwise/<run_id>/ for calls.jsonl. --output-dir and --calls-dir can override those locations. Use a fresh run ID after changing inputs, labels, prompts, code or parameters. A frozen run is not silently resumed with a new protocol.

See [the judge guide](../docs/pairwise_llm_judge.md) and [data index](../data/README.md). All curated files under results/ are eligible for Git; artifacts/ is ignored by directory rather than by individual model paths.
