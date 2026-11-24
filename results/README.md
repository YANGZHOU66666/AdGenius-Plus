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

## Future runs

Pairwise evaluation defaults to results/pairwise/<run_id>/ for results, decisions, metrics and the run manifest, and artifacts/pairwise/<run_id>/ for calls.jsonl. --output-dir and --calls-dir can override those locations. Use a fresh run ID after changing inputs, labels, prompts, code or parameters. A frozen run is not silently resumed with a new protocol.

See [the judge guide](../docs/pairwise_llm_judge.md) and [data index](../data/README.md). All curated files under results/ are eligible for Git; artifacts/ is ignored by directory rather than by individual model paths.
