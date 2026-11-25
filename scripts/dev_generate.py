"""Greedy dev-set generation with vLLM for one checkpoint (or the untrained student).

Writes one JSON line per dev product: product_id, text, finish_reason, completion_tokens.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=200)
    args = p.parse_args()
    rows = [json.loads(line) for line in (HERE / "prompts_dev_v1.jsonl").open()]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    texts = [tokenizer.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False) for r in rows]
    llm = LLM(model=args.model_dir, dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=2048, seed=0)
    outputs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
    with args.out.open("w") as f:
        for row, out in zip(rows, outputs):
            choice = out.outputs[0]
            f.write(json.dumps({"product_id": row["product_id"], "text": choice.text.strip(),
                                "finish_reason": choice.finish_reason,
                                "completion_tokens": len(choice.token_ids)}, ensure_ascii=False) + "\n")
    print(f"DEV_DONE {args.out} {len(outputs)}")


if __name__ == "__main__":
    main()
