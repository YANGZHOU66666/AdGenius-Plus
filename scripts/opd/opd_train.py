"""OPD training run for the learning-rate sweep (and later the 1k->8k continuation).

Student Qwen3.5-0.8B (text-only checkpoint), teacher Qwen3.5-397B through the local Tinker proxy, TRL
ServerDistillationTrainer with completion-only logits (bit-identical loss to TRL, see check_completion_logits.py).

Data order follows docs/sft_opd_experiment_plan.md: the frozen train ranks are split into blocks
[0,1k) [1k,2k) [2k,4k) [4k,8k); each seed shuffles inside a block only. Every prompt is repeated
`samples` times next to each other. Warmup is a fixed number of steps followed by a constant LR, so a
checkpoint after N prompts equals a run trained on the first N prompts only.

Outputs (under --out): checkpoint-<step>/ with model, tokenizer, optimizer, scheduler, RNG and trainer state
(resumable), run_report.json with the per-step log and the code/data hashes.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl.experimental.server_distillation import ServerDistillationConfig

from completion_logits_trainer import CompletionLogitsServerDistillationTrainer
from drive_sync import DriveSync

HERE = Path(__file__).resolve().parent
BLOCKS = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 8000))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_prompts(rows: list[dict], seed: int) -> list[dict]:
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
        self._t = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.times.append(round(time.time() - self._t, 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--prompts-end", type=int, default=1000, help="train on ordered prompts [0, end)")
    p.add_argument("--prompts-per-step", type=int, default=40)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--vllm-mem", type=float, default=0.15)
    p.add_argument("--teacher-url", default="http://127.0.0.1:18001")
    p.add_argument("--student", default="/content/qwen35_08b_text")
    p.add_argument("--resume-from", default=None)
    p.add_argument("--save-steps", type=int, default=25, help="full resumable checkpoint every N steps")
    p.add_argument("--drive-root", default=None, help="rclone destination for checkpoints, e.g. gdrive:adgenius_checkpoints/x")
    p.add_argument("--stop-after", type=int, default=None, help="smoke tests only: exit after this step")
    p.add_argument("--vllm-max-len", type=int, default=4096, help="headroom for the longest prompts (2307 tokens)")
    args = p.parse_args()

    prompts_file = HERE / "prompts_train_v1.jsonl"
    rows = [json.loads(line) for line in prompts_file.open()]
    ordered = ordered_prompts(rows, args.seed)[:args.prompts_end]
    seqs_per_step = args.prompts_per_step * args.samples
    if args.prompts_end % args.prompts_per_step or seqs_per_step % args.micro_batch:
        raise SystemExit("prompts_end must divide by prompts_per_step, and step size by micro_batch")
    max_steps = args.prompts_end // args.prompts_per_step
    dataset = Dataset.from_list([{"prompt": r["prompt"]} for r in ordered for _ in range(args.samples)])

    # Hash the code and data before training, so the report names what actually ran.
    hashes = {name: sha256(HERE / name) for name in
              ("opd_train.py", "completion_logits_trainer.py", "teacher_proxy.py", "drive_sync.py",
               "prompts_train_v1.jsonl")}
    tokenizer = AutoTokenizer.from_pretrained(args.student, padding_side="left")
    # On resume, load the weights with from_pretrained: checkpoints store them under the hub names
    # (model.language_model.*), which Trainer's own resume loading does not map, so it would silently keep
    # the untrained weights. Trainer still restores optimizer, scheduler, RNG and step from the checkpoint.
    model = AutoModelForCausalLM.from_pretrained(args.resume_from or args.student, dtype=torch.float32).to("cuda")
    config = ServerDistillationConfig(
        output_dir=str(args.out), teacher_model_server_url=args.teacher_url,
        beta=args.beta, loss_top_k=1, temperature=1.0, max_completion_length=128,
        chat_template_kwargs={"enable_thinking": False},
        per_device_train_batch_size=args.micro_batch, gradient_accumulation_steps=seqs_per_step // args.micro_batch,
        max_steps=max_steps, learning_rate=args.lr, lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps, optim="adamw_torch_fused", bf16=True, gradient_checkpointing=True,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=args.vllm_mem, vllm_max_model_length=args.vllm_max_len,
        shuffle_dataset=False, logging_steps=1, save_strategy="steps", save_steps=args.save_steps, save_only_model=False,
        report_to="none", seed=args.seed, data_seed=args.seed)
    log = StepLog()
    sync = DriveSync(args.drive_root, args.out / "progress.json", args.out / "uploads.log", args.stop_after)
    trainer = CompletionLogitsServerDistillationTrainer(model=model, args=config, train_dataset=dataset,
                                                        processing_class=tokenizer, callbacks=[log, sync])
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from)
    tokenizer.save_pretrained(str(args.out / f"checkpoint-{max_steps}"))
    hist = [h for h in trainer.state.log_history if "loss" in h]
    report = {
        "beta": args.beta, "lr": args.lr, "seed": args.seed, "prompts_end": args.prompts_end,
        "prompts_per_step": args.prompts_per_step, "samples": args.samples, "sequences_per_step": seqs_per_step,
        "micro_batch": args.micro_batch, "warmup_steps": args.warmup_steps, "max_steps": max_steps,
        "checkpoint": str(args.out / f"checkpoint-{max_steps}"), "resumed_from": args.resume_from,
        "save_steps": args.save_steps, "vllm_max_model_length": args.vllm_max_len,
        "train_seconds": round(time.time() - start, 1), "step_seconds": log.times,
        "peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "log": [{k: h.get(k) for k in ("step", "loss", "grad_norm", "learning_rate", "completions/mean_length",
                                       "completions/clipped_ratio")} for h in hist],
        "first_product_ids": [r["product_id"] for r in ordered[:5]],
        "hashes": hashes,
    }
    (args.out / "run_report.json").write_text(json.dumps(report, indent=1))
    with (args.out / "attempts.jsonl").open("a") as f:  # one line per (re)started training process
        f.write(json.dumps({k: report[k] for k in ("resumed_from", "train_seconds", "step_seconds",
                                                   "peak_gpu_allocated_gb")}) + "\n")
    print("RUN_REPORT " + json.dumps({k: report[k] for k in ("beta", "lr", "train_seconds", "peak_gpu_allocated_gb")}))


if __name__ == "__main__":
    main()
