"""Greedy generation with vLLM for one model on a frozen prompt file (dev or test). Runs on the VM.

Same settings as dev_generate.py used in the learning-rate sweeps (temperature 0, at most 200 new tokens, bf16,
seed 0, thinking disabled); only max_model_len is raised to 4096 so the longest test prompt (1968 tokens) fits.
Writes one JSON line per product: product_id, text, finish_reason, completion_tokens.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--limit", type=int, default=None, help="smoke tests only")
    p.add_argument("--gpu-mem", type=float, default=0.5)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.prompts.open(encoding="utf-8")][:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    texts = [tokenizer.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False) for r in rows]
    llm = LLM(model=args.model_dir, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem, max_model_len=4096, seed=0)
    outputs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    tmp = args.out.with_suffix(".partial")
    with tmp.open("w", encoding="utf-8") as f:
        for row, out in zip(rows, outputs):
            choice = out.outputs[0]
            f.write(json.dumps({"product_id": row["product_id"], "text": choice.text.strip(),
                                "finish_reason": choice.finish_reason,
                                "completion_tokens": len(choice.token_ids)}, ensure_ascii=False) + "\n")
    tmp.rename(args.out)
    print(f"GEN_DONE {args.out} {len(outputs)}")


if __name__ == "__main__":
    main()
