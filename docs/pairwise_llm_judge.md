# Pairwise LLM Judge

The evaluator compares ad communication quality against current human labels using two fresh, independent requests per pair. It supports Qwen Flash, DeepSeek, Claude Haiku and GPT-6 Luna through the shared client.

**DeepSeek is the selected primary and default judge**, based on the combined agreement, order consistency, reliability and recorded latency in the [four-judge comparison](../results/pairwise/comparison/report.md). Omitting --eval-config loads configs/eval/pairwise.toml and requires DEEPSEEK_API_KEY. The explicit pairwise_deepseek.toml has the same judge settings; use pairwise_qwen.toml for a Qwen comparison.

## Inputs and protocol

- Pairs: `data/eval/gold/candidate_pairs.jsonl` (100).
- Human reference: `data/eval/gold/pairwise_annotations_zhou.jsonl` (100).
- Frozen candidates and full dataset provenance: [data/README.md](../data/README.md).
- Prompts: `prompts/eval/pairwise/system_v2.txt` and `user_v2.txt`.
- Protocol: `pairwise-v0.2`, ad text only. No product metadata, source model names, IDs or human labels are sent to the judge.

The prompt asks which ad communicates more effectively to a reader scrolling a social media feed, considering clarity, useful specificity, appeal, naturalness and focus. Factual accuracy is a separate rubric task; see [the rubric judge guide](rubric_llm_judge.md). More attributes, more words or complete sentences do not automatically imply better advertising.

Each pair is requested as AB and BA independently. Display labels are Ad 1 and Ad 2; the strict JSON choice is the string "1" or "2" with a nonempty reason. The evaluator maps this to the fixed candidate identity, so stored A/B choices always refer to candidate_mapping, even for reversed presentation. Neither request sees the other order's result.

- Same winner in both successful orders: `stable`.
- Different winners: `order_inconsistent`, not a quality tie; never retried to force agreement.
- Missing or failed order: `incomplete`.

Human reference protocol remains v0.1 for workbench compatibility. Current labels include 15 user-confirmed model-assisted revisions. Judge agreement measures calibration to this one annotator, not two-person consensus. Original labels and revision records are in deprecated under the agreed archive convention.

## Running

From the repository root, install dependencies and put the relevant key in `.env`. The script loads that file explicitly. A dry run needs no key and writes no output:

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairwise.py   --eval-config configs/eval/pairwise_deepseek.toml   --run-id deepseek_gold_new --dry-run
```

Choose a fresh run ID for new paid requests:

```sh
# Two pairs, four initial requests.
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairwise.py   --eval-config configs/eval/pairwise_deepseek.toml   --run-id deepseek_gold_new_pilot --limit 2

# All 100 pairs, 200 initial requests.
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairwise.py   --eval-config configs/eval/pairwise_deepseek.toml   --run-id deepseek_gold_new
```

| Configuration | Credential | Provider-specific controls |
|---|---|---|
| `configs/eval/pairwise.toml` (default) | `DEEPSEEK_API_KEY` | DeepSeek; native thinking disabled, temperature 0, 1,024 output tokens |
| `configs/eval/pairwise_deepseek.toml` | `DEEPSEEK_API_KEY` | DeepSeek; native thinking disabled, temperature 0, 1,024 output tokens |
| `configs/eval/pairwise_qwen.toml` | `DASHSCOPE_API_KEY` | Qwen; thinking disabled, temperature 0, 1,024 output tokens |
| `configs/eval/pairwise_claude.toml` | `ANTHROPIC_API_KEY` | Claude Messages; adaptive thinking, low effort, no temperature, 2,048 output tokens |
| `configs/eval/pairwise_gpt6_luna.toml` | `OPENAI_API_KEY` | GPT-6 Luna; reasoning_effort none, temperature 0, 1,024 max_completion_tokens |

Use the matching eval config rather than only overriding model_alias: provider-specific thinking/output options differ. Claude receives a native output_config.format JSON schema; the compatible APIs use JSON mode with local validation. The shared client omits authorization headers from persisted response records. See [OpenAI model details](https://developers.openai.com/api/docs/models/gpt-6-luna), [DeepSeek API](https://api-docs.deepseek.com/), and [Claude structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

## Retries and resume

There are at most four attempts per order with the default three retries. Retryable transport/provider failures include timeouts, 408/429 and 5xx. Invalid JSON, refusal and truncation are logged failures; rerunning the same unchanged run can retry those failed directions. Successful directions are reused. Only one process can write a run directory at a time.

Resume validates input, prompt, configuration, schema and code fingerprints. Existing archived or pre-organization run IDs retain their captured historical paths and hashes. File organization and revised human labels mean that a new experiment needs a fresh run ID. Do not rewrite old manifests to bypass this check.

## Concurrency

`--workers N` (default 8) judges N pairs at the same time, each worker with its own HTTP client. The AB and BA requests of one pair still run one after the other in the same worker. The worker count is not part of the run fingerprint, so a run can be resumed with a different value. Rate-limit responses (429) are retried with backoff and Retry-After as before; lower `--workers` if a provider keeps returning them.

Every attempt is appended to calls.jsonl as one fsync'd line, so the journal grows linearly with the number of calls instead of being rewritten each time. A line cut off by a crash during an append is dropped on resume and that order is retried. results.jsonl, decisions.jsonl and metrics.json are derived from the journal: they are refreshed at most every 15 seconds during a run and written in pair order when the run ends or is interrupted. On Ctrl+C, pending pairs are cancelled and requests already in flight are allowed to finish and be logged.

## Results and sharing

New evaluation runs separate formal outcomes from local runtime journals:

| Location | Files | Git |
|---|---|---|
| `results/pairwise/<run_id>/` | manifest.json, results.jsonl, decisions.jsonl, metrics.json | Shared |
| `artifacts/pairwise/<run_id>/` | calls.jsonl and runtime lock | Ignored |

Use --output-dir to override the standard-result directory and --calls-dir to override the journal directory. Defaults are the two paths above with the same run ID. The result manifest records calls_file, and resume requires the same journal location. Both directories are locked during a run. AB/BA judgments, retry limits, prompts and scoring logic are unchanged.

The four completed historical runs are shared under results/pairwise/qwen37_gold_v2, deepseek41_gold_v2, claude55_gold_v2, and gpt6_luna_gold_v2. Their original manifests, result rows and decisions were moved byte-for-byte. Historical run manifests retain their originally captured paths and hashes; they were not rewritten to make old runs resume under changed code. New experiments require a fresh run ID.

The authoritative comparison against current labels is [results/pairwise/comparison/report.md](../results/pairwise/comparison/report.md), with metrics.json and a current location manifest. Metric values and judge decisions were not recomputed during file organization. Original-label metrics remain archived in deprecated.

All curated files in data/ and formal files in results/ are shared; artifacts/ is ignored as a whole. The complete 400 generated ads are in data/eval/gold, while the generation runtime snapshots/journals are under artifacts/generation/gold. See [results/README.md](../results/README.md).

## Metrics

AB and BA agreement each use all successful, labeled pairs. Mean agreement averages the two orders within a product and then averages products. Stable-only human agreement uses only order-consistent pairs, so always report its denominator and coverage. Order consistency and first-position choice rate describe judge behavior and do not change when human labels change.

Token/latency summaries include all recorded attempts, including failed outputs and retries. Unknown usage remains unknown, not zero. AB/BA are related observations; current confidence estimates resample whole products. The published comparison records that Claude inference controls differ from the three non-thinking runs and that current labels include model-assisted review.

Offline validation of the shared data and recomputed judge agreement:

```sh
PYTHONPATH=src .venv/bin/python scripts/validate_gold_data.py
```
