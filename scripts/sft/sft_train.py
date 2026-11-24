"""One SFT run for the learning-rate sweep (runs on the Colab VM, next to its data files).

Student Qwen3.5-0.8B (text-only checkpoint), full-parameter SFT on the frozen 397B teacher ads with TRL SFTTrainer;
loss on the assistant ad only (completion_only_loss), thinking disabled in the chat template.

Data order matches OPD exactly: rank order, shuffled only inside the blocks [0,1k) [1k,2k) [2k,4k) [4k,8k) with the
seed (same function as scripts/opd/opd_train.py), then the first --prompts-end examples. One pass, warmup for a
fixed number of steps then a constant LR, so a checkpoint after N examples equals a run on the first N only.

Outputs (under --out): final/ (model + tokenizer, for dev generation) and run_report.json (per-step log, timings,
peak memory, first product ids and code/data hashes taken before training).
"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

HERE = Path(__file__).resolve().parent
BLOCKS = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 8000))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_rows(rows: list[dict], seed: int) -> list[dict]:
    """Identical to scripts/opd/opd_train.py ordered_prompts."""
    rows = sorted(rows, key=lambda r: r["rank"])
    out = []
    for lo, hi in BLOCKS:
        block = rows[lo:hi]
        random.Random(f"{seed}:{lo}-{hi}").shuffle(block)
        out.extend(block)
    return out


class StepLog(TrainerCallback):
    def __init__(self):
        self.times, self._t = [], None

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self._t = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.times.append(round(time.time() - self._t, 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--prompts-end", type=int, default=1000, help="train on ordered examples [0, end)")
    p.add_argument("--per-step", type=int, default=40)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=None, help="override for smoke tests only")
    p.add_argument("--student", default="/content/qwen35_08b_text")
    args = p.parse_args()
    if args.prompts_end % args.per_step or args.per_step % args.micro_batch:
        raise SystemExit("prompts_end must divide by per_step, and per_step by micro_batch")

    data_file = HERE / "sft_train_v1.jsonl"
    hashes = {name: sha256(HERE / name) for name in ("sft_train.py", "sft_train_v1.jsonl")}
    rows = ordered_rows([json.loads(line) for line in data_file.open(encoding="utf-8")], args.seed)[:args.prompts_end]
    max_steps = args.max_steps or args.prompts_end // args.per_step
    dataset = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"],
                                  "chat_template_kwargs": {"enable_thinking": False}} for r in rows])

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    model = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.float32).to("cuda")
    config = SFTConfig(
        output_dir=str(args.out / "trainer"), max_steps=max_steps, learning_rate=args.lr,
        per_device_train_batch_size=args.micro_batch, gradient_accumulation_steps=args.per_step // args.micro_batch,
        lr_scheduler_type="constant_with_warmup", warmup_steps=args.warmup_steps, optim="adamw_torch_fused",
        bf16=True, max_length=2048, completion_only_loss=True, gradient_checkpointing=True,
        logging_steps=1, save_strategy="no", report_to="none", shuffle_dataset=False,
        # transformers defaults to a RandomSampler even with shuffle_dataset=False; keep the planned order.
        train_sampling_strategy="sequential",
        seed=args.seed, data_seed=args.seed)
    log = StepLog()
    trainer = SFTTrainer(model=model, args=config, processing_class=tokenizer, train_dataset=dataset, callbacks=[log])
    sampler = type(trainer._get_train_sampler(trainer.train_dataset)).__name__
    if sampler != "SequentialSampler":
        raise SystemExit(f"expected a SequentialSampler, got {sampler}; the data order would be lost")
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    train_seconds = round(time.time() - start, 1)
    final = args.out / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    hist = [h for h in trainer.state.log_history if "loss" in h]
    losses = [h["loss"] for h in hist]
    report = {
        "method": "sft", "lr": args.lr, "seed": args.seed, "prompts_end": args.prompts_end, "per_step": args.per_step,
        "micro_batch": args.micro_batch, "gradient_accumulation": args.per_step // args.micro_batch,
        "warmup_steps": args.warmup_steps, "max_steps": max_steps, "sampler": sampler, "final_model": str(final),
        "train_seconds": train_seconds, "step_seconds": log.times,
        "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "diverged": any(not math.isfinite(x) for x in losses),
        "log": [{k: h.get(k) for k in ("step", "loss", "grad_norm", "learning_rate", "num_tokens",
                                       "mean_token_accuracy")} for h in hist],
        "first_product_ids": [r["product_id"] for r in rows[:5]],
        "examples": len(rows), "hashes": hashes,
    }
    (args.out / "run_report.json").write_text(json.dumps(report, indent=1))
    print("RUN_REPORT " + json.dumps({k: report[k] for k in ("lr", "train_seconds", "peak_gpu_allocated_gb", "sampler")}))


if __name__ == "__main__":
    main()
