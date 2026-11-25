# Bradley–Terry: gold_generators

Anchor `qwen_local` = 0. Natural-log odds relative to the anchor; a difference of 1 is a 73% expected win rate. 95% intervals: product bootstrap, 2000 iterations, 100 products. Judged pairs scored: 100; forfeits: 0; incomplete: 0; not judged: 0.

| System | Score | 95% CI | Pairs |
|---|---:|---|---:|
| glm53_flash | 3.23 | [2.36, 4.72] | 57 |
| longcat | 2.89 | [2.09, 4.31] | 36 |
| gemma_31b | 2.23 | [1.36, 3.62] | 53 |
| qwen_local | 0.00 | [0.00, 0.00] | 54 |

Win rate is for the first system; ties are pairs whose two orders disagree. Forfeits are pairs decided by a failed generation and are included in W / T / L.

| Comparison | Pairs | W / T / L | Forfeits | Win rate [95% CI] | BT predicted |
|---|---:|---|---:|---|---:|
| gemma_31b vs qwen_local | 16 | 15 / 0 / 1 | 0 | 0.938 [0.79, 1.00] | 0.903 |
| glm53_flash vs qwen_local | 27 | 25 / 1 / 1 | 0 | 0.944 [0.86, 1.00] | 0.962 |
| longcat vs qwen_local | 11 | 11 / 0 / 0 | 0 | 1.000 [1.00, 1.00] | 0.948 (n/t) |
| gemma_31b vs glm53_flash | 21 | 3 / 4 / 14 | 0 | 0.238 [0.09, 0.41] | 0.268 |
| gemma_31b vs longcat | 16 | 4 / 3 / 9 | 0 | 0.344 [0.14, 0.57] | 0.339 |
| glm53_flash vs longcat | 9 | 3 / 5 / 1 | 0 | 0.611 [0.38, 0.82] | 0.584 |

Fit check: 0 comparison(s) with the BT prediction outside the observed interval (⚠); 1 not testable (n/t). Cycle rank 3. With cycle_rank 0 the comparisons form a tree and the fit reproduces every observed rate, so this check only has power when cycle_rank > 0. A comparison whose observed interval is a single value is not testable.
