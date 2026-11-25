"""Smoke test of the formal pipeline on real GPUs (runs on the Mac; blocks until the VM cell finishes).

    python3 scripts/formal/smoke.py --method sft     # L4
    python3 scripts/formal/smoke.py --method opd     # A100 (or H100)

4 training steps (save every 2), a simulated crash after step 3, local checkpoints deleted so the resume must come
from Drive, then resume to step 4, bf16 export at steps 2 and 4, and 10 dev + 10 test generations per export.
Drive destination: gdrive:adgenius_checkpoints/_smoke/<method> (deleted after the check). The session is kept
running for the formal run.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import watchdog_formal as w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["opd", "sft"], required=True)
    p.add_argument("--first-gpu-minutes", type=float, default=30)
    p.add_argument("--work-dir", type=Path, default=w.REPO / "artifacts/train")
    p.add_argument("--prompts-dir", type=Path, default=w.REPO / "artifacts/train/prompts/v1")
    p.add_argument("--sft-data-dir", type=Path, default=w.REPO / "artifacts/train/sft_data/v1")
    a = p.parse_args()
    w.PROMPTS, w.SFT_DATA = a.prompts_dir, a.sft_data_dir
    base = dict(w.METHODS[a.method], key=a.method)
    name = f"smoke_{a.method}"
    drive = f"gdrive:adgenius_checkpoints/_smoke/{a.method}"
    args = list(base["args"])
    args[args.index("--name") + 1] = name
    args[args.index("--drive-root") + 1] = drive
    args += ["--prompts-end", "160", "--save-steps", "2", "--milestones", "2", "4", "--stop-after", "3",
             "--drop-local-after-stop", "--gen-limit", "10"]
    w.M = dict(base, name=name, args=args, first_gpu_minutes=a.first_gpu_minutes)
    w.RESULTS = a.work_dir / name
    (w.RESULTS / "vm").mkdir(parents=True, exist_ok=True)
    w.LOG = w.RESULTS / f"smoke_{a.method}.log"
    if not w.alive():
        if not w.acquire() or not w.install():
            sys.exit("could not prepare the session")
    w.upload_code()
    w.upload_secrets()
    w.log("smoke: running the blocking cell")
    r = subprocess.run(["colab", "--auth=oauth2", "exec", "-s", w.M["session"], "-f", str(w.HERE / "formal_cell.py"),
                        "--timeout", "0"], capture_output=True, text=True)
    w.log("smoke: cell finished: " + r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "no output")
    w.sync()
    status = json.loads((w.RESULTS / "vm" / f"status_{a.method}.json").read_text())
    print(json.dumps({k: status.get(k) for k in ("phase", "attempts", "drive_full_steps", "events")}, indent=1))


if __name__ == "__main__":
    main()
