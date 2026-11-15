# Pairwise LLM Judge

The evaluator compares ad communication quality against current human labels using two fresh, independent requests per pair. It supports Qwen Flash, DeepSeek, Claude Haiku and GPT-6 Luna through the shared client.

## Inputs and protocol

- Pairs: `data/eval/gold/candidate_pairs.jsonl` (100).
- Human reference: `data/eval/gold/pairwise_annotations_zhou.jsonl` (100).
- Frozen candidates and full dataset provenance: [data/README.md](../data/README.md).
- Prompts: `prompts/eval/pairwise/system_v2.txt` and `user_v2.txt`.
- Protocol: `pairwise-v0.2`, ad text only. No product metadata, source model names, IDs or human labels are sent to the judge.

The prompt asks which ad communicates more effectively to a reader scrolling a social media feed, considering clarity, useful specificity, appeal, naturalness and focus. Factual accuracy is a separate rubric task. More attributes, more words or complete sentences do not automatically imply better advertising.

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
| `configs/eval/pairwise.toml` | `DASHSCOPE_API_KEY` | Qwen; thinking disabled, temperature 0, 1,024 output tokens |
| `configs/eval/pairwise_deepseek.toml` | `DEEPSEEK_API_KEY` | DeepSeek; native thinking disabled, temperature 0, 1,024 output tokens |
| `configs/eval/pairwise_claude.toml` | `ANTHROPIC_API_KEY` | Claude Messages; adaptive thinking, low effort, no temperature, 2,048 output tokens |
| `configs/eval/pairwise_gpt6_luna.toml` | `OPENAI_API_KEY` | GPT-6 Luna; reasoning_effort none, temperature 0, 1,024 max_completion_tokens |

Use the matching eval config rather than only overriding model_alias: provider-specific thinking/output options differ. Claude receives a native output_config.format JSON schema; the compatible APIs use JSON mode with local validation. The shared client omits authorization headers from persisted response records. See [OpenAI model details](https://developers.openai.com/api/docs/models/gpt-6-luna), [DeepSeek API](https://api-docs.deepseek.com/), and [Claude structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

## Retries and resume

There are at most four attempts per order with the default three retries. Retryable transport/provider failures include timeouts, 408/429 and 5xx. Invalid JSON, refusal and truncation are logged failures; rerunning the same unchanged run can retry those failed directions. Successful directions are reused. Only one process can write a run directory at a time.

Resume validates input, prompt, configuration, schema and code fingerprints. Existing archived or pre-organization run IDs retain their captured historical paths and hashes. File organization and revised human labels mean that a new experiment needs a fresh run ID. Do not rewrite old manifests to bypass this check. Current CLI execution is synchronous.

## Results and sharing

Each runtime directory is `artifacts/eval/pairwise/<run_id>/`:

| File | Role |
|---|---|
| `manifest.json` | Safe configuration, selected pair IDs, input/prompt/code hashes |
| `results.jsonl` | Standard PairwiseResult, exactly AB and BA per pair |
| `decisions.jsonl` | Stable, order-inconsistent or incomplete outcomes |
| `metrics.json` | Agreement and operational statistics for that run's human reference |
| `calls.jsonl` | Local full request/response journal, errors, usage and latency |

The four complete historical judge outputs are shared at qwen37_gold_v2, deepseek41_gold_v2, claude55_gold_v2, and gpt6_luna_gold_v2. Their manifests preserve original request provenance. Their original-label metrics have been archived. The authoritative comparison against current labels is [artifacts/eval/pairwise/comparison/report.md](../artifacts/eval/pairwise/comparison/report.md), backed by metrics.json and a manifest with current label/data hashes.

Git includes those four runs' standard results, decisions and manifests, plus the current comparison. Calls and other runtime outputs remain local. The complete 400 generated ads are shared in data/eval/gold independently of call logs.

## Metrics

AB and BA agreement each use all successful, labeled pairs. Mean agreement averages the two orders within a product and then averages products. Stable-only human agreement uses only order-consistent pairs, so always report its denominator and coverage. Order consistency and first-position choice rate describe judge behavior and do not change when human labels change.

Token/latency summaries include all recorded attempts, including failed outputs and retries. Unknown usage remains unknown, not zero. AB/BA are related observations; current confidence estimates resample whole products. The published comparison records that Claude inference controls differ from the three non-thinking runs and that current labels include model-assisted review.

Offline validation of the shared data and recomputed judge agreement:

```sh
PYTHONPATH=src .venv/bin/python scripts/validate_gold_data.py
```
