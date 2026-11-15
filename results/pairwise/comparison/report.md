# Current Pairwise Judge Comparison

Calculated on 2026-11-13; file organization updated on 2026-11-14. Uses the frozen shared dataset in `data/eval/gold/`. No new API calls.

| Judge | AB agreement | BA agreement | Mean agreement | Stable-only agreement | Order consistency |
|---|---:|---:|---:|---:|---:|
| DeepSeek | 82.0% | 85.0% | 83.5% | 77/87 (88.5%) | 87.0% |
| GPT-6 Luna | 79.0% | 81.0% | 80.0% | 76/92 (82.6%) | 92.0% |
| Qwen Flash | 76.0% | 82.0% | 79.0% | 65/72 (90.3%) | 72.0% |
| Claude Haiku | 72.0% | 75.0% | 73.5% | 67/87 (77.0%) | 87.0% |

Mean agreement averages the two orders per product and then averages 100 products. Stable-only agreement includes only pairs with the same winner in AB and BA, with different coverage per judge. All final runs have 200 successful judgments. Claude used 202 attempts including two failed/truncated outputs.

The current human labels contain 15 user-confirmed revisions made after model-assisted review. This is calibration against one annotator, not an independent consensus or new blind evaluation. Earlier labels, review records and original-label statistics are under `deprecated/`.

DeepSeek has the highest mean agreement point estimate; GPT has the highest order consistency. The paired product-bootstrap 95% interval for DeepSeek minus GPT is -2.0 to +9.0 percentage points, so these 100 products do not establish a clear difference between them.

## Primary judge selection

**We select DeepSeek as the primary and default pairwise judge. Considering agreement with the current human reference, order consistency, successful completion and observed latency together, it is the best overall choice for this project's current evaluation protocol.**

- Its mean human agreement is highest at 83.5%, with the highest number of pairs that are both order-consistent and human-aligned: 77/100. Its 88.5% stable-only agreement covers 87/100 pairs. Qwen's higher conditional agreement of 90.3% covers 72/100 pairs.
- Order consistency is 87%, above Qwen's 72% and equal to Claude's 87%. GPT has the strongest order consistency at 92% and remains a useful comparison judge.
- This run completed all 200 DeepSeek judgments in 200 attempts with no failed attempts, and recorded the lowest mean request latency at about 1.25 seconds.
- DeepSeek belongs to a different model family from the planned Qwen students, which supports its use as the primary judge when comparing Qwen generations. This family separation does not establish freedom from judge bias.

The operational figures supporting this choice are recorded in metrics.json:

| Judge | Stable and human-aligned / all 100 | First-presented choice rate | Attempts (failed) | Mean request latency | Input / output tokens |
|---|---:|---:|---:|---:|---:|
| DeepSeek | 77/100 | 44.5% | 200 (0) | 1.25 s | 92,028 / 12,362 |
| GPT-6 Luna | 76/100 | 51.0% | 200 (0) | 1.66 s | 88,310 / 11,310 |
| Qwen Flash | 65/100 | 40.0% | 200 (0) | 1.56 s | 94,256 / 13,744 |
| Claude Haiku | 67/100 | 48.5% | 202 (2) | 1.68 s | 195,136 / 26,915 |

Latency and token totals describe these recorded runs, including failed attempts and retries. They are not a monetary cost comparison. The selection is limited to the current task and reviewed single-annotator calibration set; the bootstrap interval above still allows either DeepSeek or GPT to have the higher underlying agreement.

The default configs/eval/pairwise.toml now uses DeepSeek with thinking disabled, temperature 0 and 1,024 output tokens. The explicit configs/eval/pairwise_deepseek.toml is retained, and configs/eval/pairwise_qwen.toml provides the Qwen comparison settings. This selection changes future defaults without rerunning or rewriting the four recorded judge runs or metrics.

Qwen, DeepSeek and GPT use temperature 0 with reasoning/thinking disabled and 1,024 output tokens. Claude uses adaptive thinking at low effort, no temperature, and 2,048 output tokens. Recorded usage and inference settings are included in metrics and run manifests.

The JSON files here and the four full-run results/manifests/decisions are shared in Git. Full calls.jsonl logs remain local under artifacts/pairwise/<run_id>/. Run manifests retain historical paths and original input hashes for provenance; new runs use the current data/result/journal paths and a fresh run ID.
