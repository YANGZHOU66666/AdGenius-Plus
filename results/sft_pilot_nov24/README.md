# Qwen3.5 advertising copy before and after supervised fine-tuning

A local pilot trained Qwen3.5-0.8B on 128 frozen product–ad examples from the project's 397B teacher. On 32 held-out development products, the adapted model met the prompt's 35-word limit on 25 ads, up from 9 before training. It also assigned higher likelihood to the teacher's reference ad on each of the 32 products.

| Development metric | Baseline | After SFT |
|---|---:|---:|
| Nonempty ads of at most 35 words | 9/32 (28.1%) | 25/32 (78.1%) |
| Mean output length, words | 48.6 | 31.0 |
| Ads reaching the 96-token generation cap | 12/32 | 3/32 |
| Mean teacher-response negative log-likelihood, averaged over products | 2.098 | 1.808 |

On the same 32 products, 19 ads moved from over the word limit to within it after training, 3 moved in the opposite direction, 6 met the limit in both versions, and 4 exceeded it in both.

Both versions used the same 4-bit base weights, product inputs, prompt, greedy decoder and output cap. A single training pass updated 200,704 LoRA parameters in the final transformer block, with no checkpoint or hyperparameter selection. Across matched products, the increase in length compliance was 50 percentage points (paired product bootstrap 95% interval, +25.0 to +71.9). Mean teacher-response loss fell by 0.290 nats per token (95% interval for the decrease, 0.237 to 0.344). The interval measures variation across development products for this single training run.

## DeepSeek evaluation on the same 32 products

DeepSeek Flash compared each pair using only the two ads, then judged the same pair in reverse order. It preferred the tuned ad in both orders for 24 products and the baseline ad in both orders for 5. Its choice changed with presentation order for 3. Across the 64 individual decisions, the tuned ad received 51 votes (79.7%). The sample consists of 32 paired products. The two votes per product are dependent.

In a separate assessment, the judge read each ad alongside its source product information and checked for direct contradictions. It flagged 7 of 32 baseline ads (21.9%) and 6 of 32 tuned ads (18.8%). Four products lost a baseline conflict flag after tuning, three gained one, three had flags in both versions, and 22 had none in either. The one-ad net difference provides little evidence of a change in factual consistency. Claims the source neither confirms nor contradicts pass this rubric.

All 32 pairs and 64 single ads received scores. The preference result describes this judge on the development sample. Shopper response and click-through rate were not measured. This pilot compares one supervised fine-tuning run against the base model and leaves the effects of data filtering and OPD open.

[Judge summary](judge_summary.json) · [Pair decisions](pairwise/decisions.jsonl) · [Factual judgments and reasons](rubric/results.jsonl)

[Reproduction steps](REPRODUCE.md) · [Metrics](summary.json) · [Run configuration](manifest.json) · [Paired ads](candidate_pairs.jsonl)
