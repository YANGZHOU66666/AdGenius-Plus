"""Runs one beta group of the OPD learning-rate sweep on a Colab VM, unattended.

1. builds the text-only student checkpoint if missing;
2. starts the Tinker proxy (actual mode for beta=1, top1 mode otherwise) and waits until it is ready;
3. optionally generates dev outputs for the untrained student;
4. for each learning rate: trains on the first 1k prompts, then generates greedy dev outputs from the checkpoint.

Progress goes to status.json next to this file; each run logs to runs/<name>/.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDENT = Path("/content/qwen35_08b_text")


STATUS = HERE / "status.json"


def write_status(status):
    STATUS.write_text(json.dumps(status, indent=1))


def run(cmd, log_path):
    with open(log_path, "w") as log:
        return subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--lrs", type=float, nargs="+", required=True)
    p.add_argument("--port", type=int, default=18001)
    p.add_argument("--max-tokens", type=int, default=15_000_000)
    p.add_argument("--base-dev", action="store_true", help="also generate dev outputs for the untrained student")
    p.add_argument("--status-file", default="status.json")
    args = p.parse_args()
    global STATUS
    STATUS = HERE / args.status_file
    status = {"beta": args.beta, "lrs": args.lrs, "started": time.strftime("%H:%M:%S"), "runs": {}, "done": False}
    write_status(status)

    if not (STUDENT / "config.json").exists():
        code = ("import torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n"
                "m=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B', dtype=torch.bfloat16)\n"
                f"m.save_pretrained('{STUDENT}'); AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B').save_pretrained('{STUDENT}')")
        subprocess.run([sys.executable, "-c", code], check=True)

    mode = "actual" if args.beta == 1.0 else "top1"
    proxy_log_path = HERE / f"proxy_{args.port}.log"
    proxy_log = open(proxy_log_path, "w")
    subprocess.Popen([sys.executable, "teacher_proxy.py", "--port", str(args.port), "--mode", mode,
                      "--max-tokens", str(args.max_tokens), "--usage-file", str(HERE / f"tinker_usage_{args.port}.json")],
                     cwd=HERE, stdout=proxy_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     start_new_session=True)
    for _ in range(100):
        text = proxy_log_path.read_text()
        if "ready on" in text or "Traceback" in text:
            break
        time.sleep(3)
    status["proxy"] = {"mode": mode, "ready": "ready on" in text}
    write_status(status)
    if "ready on" not in text:
        status["error"] = "proxy failed to start"
        write_status(status)
        return

    (HERE / "runs").mkdir(exist_ok=True)
    if args.base_dev:
        code = run([sys.executable, "dev_generate.py", "--model-dir", str(STUDENT), "--out", "runs/base_dev.jsonl"],
                   HERE / "runs" / "base_dev.log")
        status["base_dev"] = "ok" if code == 0 else f"failed ({code})"
        write_status(status)

    for lr in args.lrs:
        name = f"beta{args.beta:g}_lr{lr:g}"
        out = HERE / "runs" / name
        out.mkdir(parents=True, exist_ok=True)
        status["runs"][name] = {"phase": "train", "started": time.strftime("%H:%M:%S")}
        write_status(status)
        code = run([sys.executable, "opd_train.py", "--beta", str(args.beta), "--lr", str(lr), "--out", str(out),
                    "--teacher-url", f"http://127.0.0.1:{args.port}"], out / "train.log")
        if code != 0:
            status["runs"][name].update(phase="train_failed", code=code)
            write_status(status)
            continue
        ckpt = json.loads((out / "run_report.json").read_text())["checkpoint"]
        status["runs"][name].update(phase="dev", train_done=time.strftime("%H:%M:%S"))
        write_status(status)
        code = run([sys.executable, "dev_generate.py", "--model-dir", ckpt, "--out", str(out / "dev.jsonl")],
                   out / "dev.log")
        status["runs"][name].update(phase="done" if code == 0 else "dev_failed", finished=time.strftime("%H:%M:%S"))
        write_status(status)
    status["done"] = True
    status["finished"] = time.strftime("%H:%M:%S")
    write_status(status)


if __name__ == "__main__":
    main()
