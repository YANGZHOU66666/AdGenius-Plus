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

## Current rubric outputs

| Judge | Standard result file | Local journal |
|---|---|---|
| DeepSeek (default) | rubric/deepseek41_rubric_v1/results.jsonl | artifacts/rubric/deepseek41_rubric_v1/calls.jsonl |
| Claude Haiku | rubric/claude55_rubric_v1/results.jsonl | artifacts/rubric/claude55_rubric_v1/calls.jsonl |
| GPT-6 Luna | rubric/gpt6_luna_rubric_v1/results.jsonl | artifacts/rubric/gpt6_luna_rubric_v1/calls.jsonl |
| Qwen Flash | rubric/qwen37_rubric_v1/results.jsonl | artifacts/rubric/qwen37_rubric_v1/calls.jsonl |

Each result file has 130 RubricResult model records (`factual_support`) on data/eval/gold/rubric_sample.jsonl. rubric/comparison/ holds agreement with the current human labels (metrics.json, manifest.json); it is rebuilt by scripts/compare_rubric.py without API calls. The report is [docs/rubric_judge_agreement.md](../docs/rubric_judge_agreement.md). New rubric runs default to results/rubric/<run_id>/ and artifacts/rubric/<run_id>/. With `--candidates`, scripts/evaluate_rubric.py judges every successful ad in the given GenerationResult files and writes metrics.json with the judged conflict rate per generator; see [the rubric judge guide](../docs/rubric_llm_judge.md).

## Bradley–Terry comparisons

scripts/summarize_bt.py writes results/bt/<design name>/ (metrics.json, report.md, manifest.json): win/tie/loss per compared system pair and Bradley–Terry scores with product-bootstrap intervals, computed from pairwise judge results without API calls. Designs live in configs/eval/bt/. See [the Bradley–Terry guide](../docs/bradley_terry_eval.md).

## Test evaluation (v1)

The final evaluation of the SFT and OPD checkpoints on the 1,000 test products. Report: [docs/test_evaluation_v1.md](../docs/test_evaluation_v1.md).

| Output | Contents |
|---|---|
| pairwise/deepseek41_test_sft_opd_v1/ | DeepSeek AB/BA judgments for the 16,000 pairs in data/eval/test/bt_pairs_test_sft_opd_v1.jsonl (16 comparisons, 10 systems) |
| bt/test_sft_opd_v1/ | Bradley–Terry scores and per-comparison win/tie/loss for configs/eval/bt/test_sft_opd_v1.toml |
| rubric/deepseek41_rubric_v1_test/ | factual_support judgments for all 10,000 test ads and the conflict rate per system |

Call journals are in artifacts/pairwise/deepseek41_test_sft_opd_v1/ and artifacts/rubric/deepseek41_rubric_v1_test/.

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
