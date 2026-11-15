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

Qwen, DeepSeek and GPT use temperature 0 with reasoning/thinking disabled and 1,024 output tokens. Claude uses adaptive thinking at low effort, no temperature, and 2,048 output tokens. Recorded usage and inference settings are included in metrics and run manifests.

The JSON files here and the four full-run results/manifests/decisions are shared in Git. Full calls.jsonl logs remain local under artifacts/pairwise/<run_id>/. Run manifests retain historical paths and original input hashes for provenance; new runs use the current data/result/journal paths and a fresh run ID.
