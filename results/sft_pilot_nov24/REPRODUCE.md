# Reproduce the paired Qwen SFT pilot

Use the AdGenius-Plus repository at the November 24 evidence branch, including the three pilot scripts. The frozen training inputs and teacher ads are on the training branch at f98a45ffd05300f9206ddff8f777fc85a461f7fa. Product IDs and file hashes are recorded with the outputs.

The run used Apple M5 (24 GB), Python 3.12.14, mlx-lm 0.32.2, mlx 0.32.4, mlx-metal 0.32.4, transformers 5.24.1 and tokenizers 0.23.3. The model is the MLX 4-bit text checkpoint of Qwen3.5-0.8B at Hugging Face revision da28692b5f139cb0ec58a356b437486b7dac7462. Download the safetensors, its index, config.json, tokenizer.json, tokenizer_config.json and chat_template.jinja from this revision. The run script verifies the weight SHA-256 before training.

From the repository root, use an isolated environment with these dependencies plus the project's Python requirements. The following commands use a fresh output path so the saved run remains available for review.

```sh
PYTHONPATH=src HF_HUB_OFFLINE=1 python scripts/run_sft_pilot_nov24.py \
  --model artifacts/pilot_nov24/base \
  --data ../tmp/adgenius-train-nov24/data/train/v1 \
  --out artifacts/pilot_reproduction/results \
  --adapters artifacts/pilot_reproduction/adapter
python scripts/summarize_sft_pilot_nov24.py --out artifacts/pilot_reproduction/results
```

The first 128 frozen training products form the training subset and the first 32 frozen dev products form the evaluation subset. Each training product appears once, in order shuffled with seed 20261124. Both systems receive the repository's identical frozen English product-card prompt and the full rendered product information. Thinking is disabled. Generation is greedy, with a 96-token cap. The same 4-bit base weights are used throughout. The tuned version adds rank-8 adapters to linear projections in the final full-attention block. No dropout or learning-rate search is used. AdamW takes 128 updates at 0.0001. Loss is computed on teacher response tokens, including the end-of-response token.

Each dev teacher reference is used to compute its own response-token mean negative log-likelihood. The primary change is the paired, product-weighted mean difference (tuned minus base). The uncertainty interval resamples 32 whole matched products with replacement 10,000 times. The corresponding 35-word check uses nonempty output and whitespace-separated word counts. Score comparisons involving different prompts, tokenizers, or samples require a separate run.

To validate or run the project's official DeepSeek judges on the saved run:

```sh
python scripts/judge_sft_pilot_nov24.py --dry-run
# With DEEPSEEK_API_KEY configured in the project .env:
python scripts/judge_sft_pilot_nov24.py
```

One complete evaluation sends two presentation orders per eligible product pair and one factual-conflict check per successful ad. The runner keeps individual decisions and can resume incomplete runs.
