# Pairwise LLM Judge

The pairwise evaluator runs a blind Qwen judge over the frozen `CandidatePair` set. Each pair receives two separate requests: canonical order AB and reversed order BA. The two responses are never shown to one another. Under the current `pairwise-v0.2` protocol, the prompt shows only `Ad 1` / `Ad 2` and the communication-quality instructions. It does not show product metadata, candidate IDs, source model, generation run, or the human label.

The judge chooses the more effective ad for a reader scrolling a social media feed. It considers immediate clarity, useful specificity, appeal, naturalness, and focus together. More attributes or fewer words do not automatically make an ad better; short phrases and complete sentences can both work. Factual accuracy is evaluated separately by the rubric judge. This pairwise prompt instructs the model not to fact-check or speculate about product claims, and not to reward generic hype that adds little meaningful content.

The judge must choose one ad even when they are close in quality. The prompt requires a JSON object with a forced `"1"` or `"2"` choice and a short reason. Invalid, truncated, or refused output is recorded as a failed call; it is not silently repaired or treated as a tie.

The canonical `CandidatePair.candidates[0]` is identity A and `[1]` is identity B. AB displays A as Ad 1 and B as Ad 2; BA displays B as Ad 1 and A as Ad 2. The evaluator converts the returned slot into the candidate ID, then into the fixed A/B identity in `PairwiseResult`. A stable winner requires both calls to select the same candidate ID. Different winners are `order_inconsistent`, never a tie. A missing or failed direction makes the pair incomplete.

## Run

Install the project dependencies and make the DashScope key available as `DASHSCOPE_API_KEY` in the environment or in `.env`. A worktree can point to an existing local env file with `--env-file`; the file is read locally and is never copied into the repository.

Validate the frozen inputs and planned call count without a key, output files, or API calls:

```sh
PYTHONPATH=src python scripts/evaluate_pairwise.py --dry-run --limit 2
```

Run a small paid pilot first:

```sh
PYTHONPATH=src python scripts/evaluate_pairwise.py \
  --run-id qwen37_gold_pilot_v2 \
  --limit 2
```

Then evaluate the full 100-pair set with a new run ID:

```sh
PYTHONPATH=src python scripts/evaluate_pairwise.py \
  --run-id qwen37_gold_v2
```

The default model alias is `qwen`, mapped to `qwen3.7-flash` in `configs/models.toml`. Evaluation settings are in `configs/eval/pairwise.toml`; the active prompts are `prompts/eval/pairwise/system_v2.txt` and `user_v2.txt`. Thinking remains disabled. Inputs require only `CandidatePair` records, which already include both ad texts; there is no `--products` argument or `ProductItem` dependency.

The default 100 human labels are loaded for agreement analysis. Supply one annotator file with `--human-annotations PATH`, or opt out with `--no-human-annotations`. `human_annotation_protocol_version` explicitly selects the accepted human-label protocol and defaults to the judge protocol when omitted. The default config explicitly uses the existing `pairwise-v0.1` labels as a calibration reference while the model uses `pairwise-v0.2`. Label records are not rewritten. The manifest records both protocol versions and the metrics flag the cross-protocol comparison; the existing labels were collected in a workbench with product information visible and are not newly collected ad-text-only labels.

The former `system_v1.txt` and `user_v1.txt` are retained as historical prompts. Existing `qwen37_gold_v1` results remain separate from the new run; use a new run ID as shown above. The current evaluator refuses to resume earlier runs with changed inputs, prompts, configuration, or code.

For a DeepSeek comparison, set `DEEPSEEK_API_KEY` in the repository `.env` and select the separate judge configuration. It uses the same v2 prompts, inputs, temperature, output budget, and AB/BA procedure, with DeepSeek's native `thinking = { type = "disabled" }` option. The official API model ID `deepseek-flash` currently serves DeepSeek-V4.1-Flash-1017 ([official model details](https://api-docs.deepseek.com/quick_start/pricing/)). The call journal records the model returned by the provider. Use `--eval-config` rather than only changing `--model-alias` on the Qwen config, because their thinking options differ.

```sh
# No API calls; validates configuration and the planned 200 initial calls.
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairwise.py \
  --eval-config configs/eval/pairwise_deepseek.toml \
  --run-id deepseek41_gold_v2 --dry-run

# Paid pilot: 2 pairs, 4 initial API calls.
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairwise.py \
  --eval-config configs/eval/pairwise_deepseek.toml \
  --run-id deepseek41_gold_pilot_v2 --limit 2

# Full comparison: 100 pairs, 200 initial API calls.
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairwise.py \
  --eval-config configs/eval/pairwise_deepseek.toml \
  --run-id deepseek41_gold_v2
```

For Claude Haiku 5.5, set `ANTHROPIC_API_KEY` in the repository `.env` and use `configs/eval/pairwise_claude.toml`. The pipeline reuses the same v2 prompts and AB/BA mapping, calls the Anthropic Messages API, and supplies a native `output_config.format` JSON schema for `choice` and `reason`. The shared client reads only text blocks as the answer; any thinking blocks remain in the raw call response. Refusal, truncation, and empty output remain failures.

Haiku 5.5 has different inference controls: the config omits `temperature`, requests adaptive thinking with `output_config.effort = "low"`, and allows 2,048 output tokens for thinking plus the JSON answer. Qwen and DeepSeek runs use temperature 0, disable thinking, and allow 1,024 output tokens. These differences are recorded in the run manifest and call journal, so the runs compare the same evaluation protocol with provider-specific inference settings; they are not a strictly matched non-thinking experiment. See [Haiku migration guidance](https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide) and [native structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

```sh
# Paid pilot: 2 pairs, 4 initial API calls.
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairwise.py \
  --eval-config configs/eval/pairwise_claude.toml \
  --run-id claude55_gold_pilot_v2 --limit 2

# Full comparison: 100 pairs, 200 initial API calls.
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairwise.py \
  --eval-config configs/eval/pairwise_claude.toml \
  --run-id claude55_gold_v2
```

For GPT-6 Luna, put `OPENAI_API_KEY=...` in the repository `.env` and select `configs/eval/pairwise_gpt6_luna.toml`. This configuration calls the official OpenAI Chat Completions endpoint with `model = "gpt-6-luna"`, `reasoning_effort = "none"`, temperature 0, and a 1,024-token output cap. It reuses the same ad-text-only v2 prompts, frozen pairs, human labels, AB/BA procedure, and result schema. See the [official OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna).

The model config sets `max_tokens_parameter = "max_completion_tokens"`; the shared client's `max_output_tokens` argument is translated to that request field. Other models retain their existing `max_tokens` default. Provider options cannot override either token-limit field. This integration has offline HTTP contract coverage; a dry run does not verify credentials or account model access.

```sh
# No API calls or output files; no key required.
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairwise.py \
  --eval-config configs/eval/pairwise_gpt6_luna.toml \
  --run-id gpt6_luna_gold_pilot_v2 --limit 2 --dry-run

# Paid pilot: 2 pairs, 4 initial API calls.
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairwise.py \
  --eval-config configs/eval/pairwise_gpt6_luna.toml \
  --run-id gpt6_luna_gold_pilot_v2 --limit 2

# Full comparison: 100 pairs, 200 initial API calls.
PYTHONPATH=src .venv/bin/python scripts/evaluate_pairwise.py \
  --eval-config configs/eval/pairwise_gpt6_luna.toml \
  --run-id gpt6_luna_gold_v2
```

Each pair needs two initial API calls. With the configured three retries, one direction can make at most four attempts, and retries occur only for retryable transport/provider errors such as timeouts, 408/429, and 5xx responses. Invalid JSON or a refusal is logged as failed and can be retried by rerunning with the same run ID. Previously successful directions are reused. Resume is refused if inputs, prompts, configuration, schemas, or relevant code changed; use a new run ID for a changed experiment. Only one process may write a run directory at a time.

## Artifacts

By default, outputs are written to `artifacts/eval/pairwise/<run_id>/`:

| File | Contents |
| --- | --- |
| `manifest.json` | Run ID, status, safe model configuration, prompt/input/code hashes, and protocol settings. |
| `calls.jsonl` | One row per actual API attempt, including request, response, usage, latency, error details, and trace ID. |
| `results.jsonl` | One schema-validated model `PairwiseResult` per pair with AB and BA outputs. |
| `decisions.jsonl` | Derived `stable`, `order_inconsistent`, or `incomplete` outcome for each completed pair record. |
| `metrics.json` | Call success, stable coverage, order consistency, first-position preference, per-order and stable human agreement, token usage, and latency. |

The manifest stores no key value. The call log includes ad text, judge instructions, and provider responses, so treat the run directory as experiment data. It is not committed automatically.

## Interpreting Results

Human agreement is aligned by candidate ID, so random human A/B presentation is handled correctly. Stable agreement uses only pairs where both model orders succeeded and chose the same candidate; its denominator and coverage over all valid human labels are reported explicitly. AB-only and BA-only agreement are also reported. One annotator's labels measure agreement with that annotator, not consensus or inter-annotator reliability. Agreement with earlier-protocol labels is a calibration diagnostic; changing the prompt does not prove that communication preference and factual accuracy are now fully separated in those labels.

The first-position choice rate is a diagnostic for order preference. AB and BA are repeated judgments of the same pair and are not independent evaluation examples. The confidence interval in the metrics is descriptive; final comparisons should resample at product level.

Qwen is intentionally enabled as the first judge here. Because some candidates may also come from Qwen, use this run as evaluator calibration against human labels, not as independent evidence that Qwen candidates are better. The judge model, prompt, and protocol are frozen in the manifest so another judge can later be evaluated on the same inputs.

## Current Data and Calibration Snapshot — 2026-11-13

The current branch contains the short-ad generation flow, ad-text-only pairwise v2 protocol, and 100 human preference labels. Fifteen labels were revised after the annotator reviewed model disagreements and explicitly confirmed the changes.

### Current inputs and local outputs

| Path | Role | Version control |
|---|---|---|
| `data/splits/v1/` | Frozen 100-product gold and 1,000-product test splits; 709,403-product candidate training pool. The train-pool JSONL is local and reproducible from the original metadata. | Small split files tracked; train pool ignored. |
| `data/processed/v1/` | Original metadata records for the 100 gold and 1,000 test products. | Tracked. |
| `data/eval/gold/v2/` | Current 100 ProductItems, 100 CandidatePairs, pair provenance, and 100 reviewed human labels. | Tracked. |
| `data/eval/gold/v1/product_items.jsonl` | Byte-identical to the v2 ProductItems; still the default generation/retry input. Retain this path while scripts reference it. | Tracked. |
| `artifacts/eval/gold_generation/v2/` | Four successful 100-candidate GenerationResult files (400 total), plus original call history and retry provenance. | Local experiment output; ignored. |
| `artifacts/eval/pairwise/{qwen37,deepseek41,claude55,gpt6_luna}_gold_v2/` | Four complete 100-pair judge runs; each pair has AB and BA judgments. | Local experiment output; ignored. |
| `artifacts/eval/pairwise/reviewed_comparison_20261113_150614/` | Offline recalculated metrics, original/reviewed label snapshots, source hashes, and per-pair comparisons. | Local experiment output; ignored. |
| `artifacts/eval/gold_review/` | Four user-confirmed review batches with before-snapshots and corrections. | Local backup/audit output; ignored. |

The original Office Products metadata is currently stored outside the repository at `../data/raw/amazon_reviews_2023/meta_categories/meta_Office_Products.jsonl.gz`; the repository’s `data/raw/` is empty. The frozen split manifest records the source hash and line references. `data/train/`, `artifacts/train/`, and `artifacts/checkpoints/` have no training datasets or model outputs yet.

The v1/v2 suffixes are stage-specific: the product split remains v1; candidate generation and CandidatePairs are v2; judge prompt/protocol is v0.2. Earlier candidate generations, prompt pilots, pre-retry backups and Qwen v1 judge runs remain local historical material. Calls containing failed attempts are valid history; use the corresponding final results to determine completion.

### Label-review provenance

- Annotator: `zhou`; annotation set: `gold_pairwise_v2`; record protocol tag remains `pairwise-v0.1` for compatibility with the workbench.
- The revision procedure was model-assisted, not a second independent blind annotation. Both alternatives were shown, recommendations were discussed, and the user confirmed the following 15 changes.
- A/B below are canonical `CandidatePair.candidates[0/1]`, not the randomized A/B positions shown in the workbench. Stored `choice` values remain aligned with each record’s saved `presentation`.
- The previous committed annotation file contains the original labels. The local comparison directory also freezes both original and reviewed snapshots. Existing judge run metrics still refer to original labels; reviewed metrics are separate.

Original annotation SHA-256: `231890fdcf4dc9db8b95800f9d5823a0cb2b6f691d6519d48c139ee9e5436236`.

Reviewed annotation SHA-256: `16894ca15b4c3d7373c338278ad31970dbda6c511e8840cb500d93c3a79af62f`.

| Discussion case | Product ID | Original canonical choice | Reviewed canonical choice |
|---|---|---|---|
| 1 | `gold_product_000013` | A | B |
| 2 | `gold_product_000052` | B | A |
| 3 | `gold_product_000046` | B | A |
| 6 | `gold_product_000047` | B | A |
| 7 | `gold_product_000089` | B | A |
| 8 | `gold_product_000004` | B | A |
| 11 | `gold_product_000051` | A | B |
| 12 | `gold_product_000050` | A | B |
| 13 | `gold_product_000055` | A | B |
| 14 | `gold_product_000059` | B | A |
| 15 | `gold_product_000087` | B | A |
| 16 | `gold_product_000003` | A | B |
| 17 | `gold_product_000028` | A | B |
| 18 | `gold_product_000045` | A | B |
| 20 | `gold_product_000084` | B | A |

### Four-judge comparison

All runs share the same 100 pairs and v2 ad-text-only prompts. No new API calls were made for label-review recalculation.

| Judge | Original mean agreement | Reviewed mean agreement | AB | BA | Stable-only human agreement | Order consistency |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek | 68.5% | 83.5% | 82.0% | 85.0% | 77/87 (88.5%) | 87.0% |
| GPT-6 Luna | 65.0% | 80.0% | 79.0% | 81.0% | 76/92 (82.6%) | 92.0% |
| Qwen Flash | 65.0% | 79.0% | 76.0% | 82.0% | 65/72 (90.3%) | 72.0% |
| Claude Haiku | 59.5% | 73.5% | 72.0% | 75.0% | 67/87 (77.0%) | 87.0% |

Mean agreement averages the two orders within each product and then averages products. Stable-only agreement uses different subsets across judges and must be read alongside coverage. All final runs have 200 successful judgments; Claude made 202 total attempts because two truncated outputs were later replaced by successful calls.

DeepSeek has the highest point estimate for mean human agreement before and after review, while GPT has the highest order consistency. The paired product-bootstrap 95% interval for DeepSeek minus GPT is −2.0 to +9.0 percentage points, so this batch does not establish a clear difference between them. These are calibration statistics: the increase follows model-assisted label revisions with unchanged judge outputs, rather than a model capability improvement. Claude’s inference settings also differ from the three non-thinking runs, as described above.
