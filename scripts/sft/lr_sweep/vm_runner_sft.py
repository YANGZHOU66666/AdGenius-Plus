"""Runs the SFT learning-rate sweep on a Colab VM (called by sweep_cell_sft.py inside a blocking notebook cell).

1. builds the text-only student checkpoint if missing;
2. for each learning rate: trains on the first 1k examples, then generates greedy dev outputs from the final model.
A run whose loss is not finite is marked diverged and gets no dev outputs (excluded by the selection rule).

Progress goes to the status file next to this script; each run logs to runs/<name>/.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDENT = Path("/content/qwen35_08b_text")


def run(cmd, log_path):
    with open(log_path, "w") as log:
        return subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lrs", type=float, nargs="+", required=True)
    p.add_argument("--status-file", default="status_sft.json")
    p.add_argument("--name-prefix", default="sft")
    p.add_argument("--max-steps", type=int, default=None, help="smoke tests only")
    args = p.parse_args()
    status_path = HERE / args.status_file
    status = {"lrs": args.lrs, "started": time.strftime("%H:%M:%S"), "runs": {}, "done": False}

    def write():
        status_path.write_text(json.dumps(status, indent=1))

    write()
    if not (STUDENT / "config.json").exists():
        code = ("import torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n"
                "m=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B', dtype=torch.bfloat16)\n"
                f"m.save_pretrained('{STUDENT}'); AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B').save_pretrained('{STUDENT}')")
        subprocess.run([sys.executable, "-c", code], check=True)

    (HERE / "runs").mkdir(exist_ok=True)
    for lr in args.lrs:
        name = f"{args.name_prefix}_lr{lr:g}"
        out = HERE / "runs" / name
        out.mkdir(parents=True, exist_ok=True)
        status["runs"][name] = {"phase": "train", "started": time.strftime("%H:%M:%S")}
        write()
        cmd = [sys.executable, "sft_train.py", "--lr", str(lr), "--out", str(out)]
        if args.max_steps:
            cmd += ["--max-steps", str(args.max_steps)]
        code = run(cmd, out / "train.log")
        if code != 0:
            status["runs"][name].update(phase="train_failed", code=code)
            write()
            continue
        report = json.loads((out / "run_report.json").read_text())
        if report["diverged"]:
            status["runs"][name].update(phase="diverged", train_done=time.strftime("%H:%M:%S"))
            write()
            continue
        status["runs"][name].update(phase="dev", train_done=time.strftime("%H:%M:%S"))
        write()
        code = run([sys.executable, "dev_generate.py", "--model-dir", report["final_model"], "--out", str(out / "dev.jsonl")],
                   out / "dev.log")
        status["runs"][name].update(phase="done" if code == 0 else "dev_failed", finished=time.strftime("%H:%M:%S"))
        write()
    status.update(done=True, finished=time.strftime("%H:%M:%S"))
    write()


if __name__ == "__main__":
    main()
