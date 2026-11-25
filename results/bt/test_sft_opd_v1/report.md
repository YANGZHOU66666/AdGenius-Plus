# Bradley–Terry: test_sft_opd_v1

Anchor `base` = 0. Natural-log odds relative to the anchor; a difference of 1 is a 73% expected win rate. 95% intervals: product bootstrap, 2000 iterations, 1000 products. Judged pairs scored: 16000; forfeits: 0; incomplete: 0; not judged: 0.

| System | Score | 95% CI | Pairs |
|---|---:|---|---:|
| teacher_397b | 3.53 | [3.38, 3.70] | 2000 |
| sft_8k | 2.80 | [2.66, 2.96] | 4000 |
| opd_8k | 2.61 | [2.47, 2.77] | 4000 |
| sft_4k | 2.53 | [2.39, 2.67] | 3000 |
| sft_2k | 2.46 | [2.33, 2.62] | 3000 |
| opd_4k | 2.46 | [2.32, 2.61] | 3000 |
| opd_2k | 2.35 | [2.22, 2.49] | 3000 |
| sft_1k | 1.99 | [1.86, 2.14] | 3000 |
| opd_1k | 1.94 | [1.80, 2.08] | 3000 |
| base | 0.00 | [0.00, 0.00] | 4000 |

Win rate is for the first system; ties are pairs whose two orders disagree. Forfeits are pairs decided by a failed generation and are included in W / T / L.

| Comparison | Pairs | W / T / L | Forfeits | Win rate [95% CI] | BT predicted |
|---|---:|---|---:|---|---:|
| opd_1k vs sft_1k | 1000 | 355 / 263 / 382 | 0 | 0.486 [0.46, 0.51] | 0.486 |
| opd_2k vs sft_2k | 1000 | 308 / 318 / 374 | 0 | 0.467 [0.44, 0.49] | 0.471 |
| opd_4k vs sft_4k | 1000 | 320 / 347 / 333 | 0 | 0.493 [0.47, 0.52] | 0.484 |
| opd_8k vs sft_8k | 1000 | 281 / 329 / 390 | 0 | 0.446 [0.42, 0.47] | 0.452 |
| sft_2k vs sft_1k | 1000 | 459 / 310 / 231 | 0 | 0.614 [0.59, 0.64] | 0.616 |
| sft_4k vs sft_2k | 1000 | 346 / 344 / 310 | 0 | 0.518 [0.49, 0.54] | 0.515 |
| sft_8k vs sft_4k | 1000 | 390 / 343 / 267 | 0 | 0.561 [0.54, 0.58] | 0.569 |
| opd_2k vs opd_1k | 1000 | 444 / 311 / 245 | 0 | 0.600 [0.57, 0.62] | 0.602 |
| opd_4k vs opd_2k | 1000 | 344 / 356 / 300 | 0 | 0.522 [0.50, 0.55] | 0.528 |
| opd_8k vs opd_4k | 1000 | 363 / 355 / 282 | 0 | 0.540 [0.52, 0.56] | 0.537 |
| sft_1k vs base | 1000 | 820 / 119 / 61 | 0 | 0.879 [0.86, 0.90] | 0.880 |
| opd_1k vs base | 1000 | 804 / 134 / 62 | 0 | 0.871 [0.85, 0.89] | 0.874 |
| sft_8k vs base | 1000 | 907 / 71 / 22 | 0 | 0.943 [0.93, 0.95] | 0.943 |
| opd_8k vs base | 1000 | 898 / 76 / 26 | 0 | 0.936 [0.92, 0.95] | 0.931 |
| teacher_397b vs sft_8k | 1000 | 529 / 289 / 182 | 0 | 0.673 [0.65, 0.70] | 0.674 |
| teacher_397b vs opd_8k | 1000 | 591 / 251 / 158 | 0 | 0.717 [0.69, 0.74] | 0.715 |

Fit check: 0 comparison(s) with the BT prediction outside the observed interval (⚠); 0 not testable (n/t). Cycle rank 7. With cycle_rank 0 the comparisons form a tree and the fit reproduces every observed rate, so this check only has power when cycle_rank > 0. A comparison whose observed interval is a single value is not testable.
