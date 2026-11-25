"""Orchestrates one formal run (OPD or SFT) on a Colab VM; called by formal_cell.py inside a blocking cell.

Idempotent and resumable: every phase checks what already exists locally or on Drive, so a restarted cell (after a
crash, an idle kernel or a reclaimed VM) continues where the previous one stopped.
1. build the text-only student checkpoint if missing; for OPD start the Tinker proxy;
2. train to max_steps, resuming from the newest complete checkpoint (local, else downloaded from Drive); a failed
   training process is retried up to --train-retries times, each time from the newest checkpoint;
3. wait until every checkpoint upload has finished;
4. for each milestone step: export bf16 inference weights (config, weights, tokenizer) to weights/<scale>/, upload
   them to <drive>/weights_bf16/<scale>/, then generate dev and test outputs into outputs/<scale>/;
5. optionally generate untrained-student test outputs (outputs/base/test.jsonl);
6. upload outputs and reports to Drive, write DONE.
Status goes to status_<method>.json next to this script.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import drive_sync

HERE = Path(__file__).resolve().parent
STUDENT = Path("/content/qwen35_08b_text")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["opd", "sft"], required=True)
    p.add_argument("--name", required=True, help="run name, e.g. opd_formal_v1")
    p.add_argument("--lr", required=True)
    p.add_argument("--beta", default="0.5")
    p.add_argument("--prompts-end", type=int, default=8000)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--milestones", type=int, nargs="+", default=[25, 50, 100, 200])
    p.add_argument("--drive-root", required=True)
    p.add_argument("--train-retries", type=int, default=3)
    p.add_argument("--base-test", action="store_true", help="also generate untrained-student test outputs")
    p.add_argument("--gen-limit", type=int, default=None, help="smoke tests only")
    p.add_argument("--stop-after", type=int, default=None, help="smoke tests only: first attempt exits after this step")
    p.add_argument("--drop-local-after-stop", action="store_true", help="smoke: force the resume to use Drive")
    p.add_argument("--max-tokens-cap", type=int, default=32_000_000)
    p.add_argument("--upload-wait-minutes", type=int, default=60, help="cap on waiting for checkpoint uploads at the end")
    args = p.parse_args()

    run_dir = HERE / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer_dir = run_dir if args.method == "opd" else run_dir / "trainer"
    status_path = HERE / f"status_{args.method}.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"method": args.method, "name": args.name,
                                                                              "attempts": [], "events": []}
    status.update(phase="starting", cell_started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def save():
        status_path.write_text(json.dumps(status, indent=1))

    def event(msg):
        status["events"].append(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}")
        save()
        print(msg, flush=True)

    save()
    if not (STUDENT / "config.json").exists():
        code = ("import torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n"
                "m=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B', dtype=torch.bfloat16)\n"
                f"m.save_pretrained('{STUDENT}'); AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B').save_pretrained('{STUDENT}')")
        subprocess.run([sys.executable, "-c", code], check=True)

    max_steps = args.prompts_end // 40
    final_step = lambda: max(drive_sync.local_complete_steps(trainer_dir), default=0)  # noqa: E731

    # --- training -------------------------------------------------------------------------------------------------
    reached = max(drive_sync.local_complete_steps(trainer_dir) + drive_sync.drive_complete_steps(args.drive_root), default=0)
    if reached < max_steps:
        if args.method == "opd":
            proxy_log = HERE / "proxy.log"
            subprocess.run(["pkill", "-f", "teacher_proxy.py"])
            if proxy_log.exists():  # keep the previous cell's proxy log
                proxy_log.rename(HERE / f"proxy.{int(time.time())}.log")
            subprocess.Popen([sys.executable, "teacher_proxy.py", "--port", "18001", "--mode", "top1",
                              "--max-tokens", str(args.max_tokens_cap),
                              "--usage-file", str(HERE / "tinker_usage.json")],
                             cwd=HERE, stdout=open(proxy_log, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
            for _ in range(100):
                if "ready on" in proxy_log.read_text() or "Traceback" in proxy_log.read_text():
                    break
                time.sleep(3)
            if "ready on" not in proxy_log.read_text():
                event("FAILED: Tinker proxy did not start")
                status["phase"] = "failed"
                save()
                return 1
        for attempt in range(1, args.train_retries + 2):
            if final_step() >= max_steps:
                break
            ckpt = drive_sync.latest_checkpoint(trainer_dir, args.drive_root, log=event)
            stop_after = args.stop_after if (attempt == 1 and ckpt is None) else None
            status.update(phase="train", resume_from=str(ckpt) if ckpt else None)
            event(f"training attempt {attempt}, resume from {ckpt.name if ckpt else 'scratch'}")
            if args.method == "opd":
                cmd = [sys.executable, "opd_train.py", "--beta", args.beta, "--lr", args.lr, "--out", str(run_dir),
                       "--prompts-end", str(args.prompts_end), "--save-steps", str(args.save_steps),
                       "--drive-root", args.drive_root, "--teacher-url", "http://127.0.0.1:18001"]
            else:
                cmd = [sys.executable, "sft_train.py", "--lr", args.lr, "--out", str(run_dir),
                       "--prompts-end", str(args.prompts_end), "--save-steps", str(args.save_steps),
                       "--drive-root", args.drive_root]
            if ckpt:
                cmd += ["--resume-from", str(ckpt)]
            if stop_after:
                cmd += ["--stop-after", str(stop_after)]
            with open(run_dir / "train.log", "a") as log:
                log.write(f"\n===== attempt {attempt} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} =====\n")
                log.flush()
                code = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
            status["attempts"].append({"attempt": attempt, "exit": code, "resumed_from": ckpt.name if ckpt else None,
                                       "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            event(f"training attempt {attempt} exited with {code}")
            if code != 0 and args.drop_local_after_stop and stop_after:
                drive_sync.wait_for_uploads()
                for c in trainer_dir.glob("checkpoint-*"):
                    shutil.rmtree(c)
                event("smoke: removed local checkpoints so the resume must come from Drive")
        if final_step() < max_steps:
            event(f"FAILED: training stopped at step {final_step()} of {max_steps} after all retries")
            status["phase"] = "failed"
            save()
            return 1
    # Checkpoint uploads keep running in the background; export and generation use the local checkpoints.

    # --- export and generation ------------------------------------------------------------------------------------
    for step in args.milestones:
        scale = f"scale_{step * 40 // 1000}k" if step * 40 % 1000 == 0 else f"step_{step:04d}"
        out_dir = run_dir / "outputs" / scale
        out_dir.mkdir(parents=True, exist_ok=True)
        weights = run_dir / "weights" / scale
        if not (weights / "config.json").exists():
            status.update(phase=f"export_{scale}")
            save()
            src = trainer_dir / f"checkpoint-{step}"
            if not (src / "config.json").exists():
                src.mkdir(parents=True, exist_ok=True)
                drive_sync.rclone("copy", f"{args.drive_root}/full/step_{step:04d}", str(src), "--exclude", "optimizer.*",
                                  "--exclude", "UPLOAD_OK", check=True)
            code = ("import sys, torch\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n"
                    "m = AutoModelForCausalLM.from_pretrained(sys.argv[1], dtype=torch.float32).to(torch.bfloat16)\n"
                    "m.save_pretrained(sys.argv[2]); AutoTokenizer.from_pretrained(sys.argv[3]).save_pretrained(sys.argv[2])\n")
            subprocess.run([sys.executable, "-c", code, str(src), str(weights), str(STUDENT)], check=True)
            # Upload in the background (the shared rclone client can stall on quota retries); the end-of-run
            # wait cell copies every export again, so generation never waits for Drive.
            with open(run_dir / "uploads.log", "a") as ulog:
                subprocess.Popen(["rclone", "copy", str(weights), f"{args.drive_root}/weights_bf16/{scale}",
                                  *drive_sync.RETRY, "-q"], stdout=ulog, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
            event(f"exported bf16 weights for {scale}; Drive upload started in the background")
        for split in ("dev", "test"):
            target = out_dir / f"{split}.jsonl"
            if target.exists():
                continue
            status.update(phase=f"generate_{scale}_{split}")
            save()
            cmd = [sys.executable, "gen_outputs.py", "--model-dir", str(weights), "--prompts", str(HERE / f"prompts_{split}_v1.jsonl"),
                   "--out", str(target)] + (["--limit", str(args.gen_limit)] if args.gen_limit else [])
            with open(out_dir / f"{split}.log", "w") as log:
                code = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
            if code != 0:
                event(f"FAILED: generation {scale} {split} exited with {code}")
                status["phase"] = "failed"
                save()
                return 1
            event(f"generated {scale} {split}")
    if args.base_test and not (run_dir / "outputs/base/test.jsonl").exists():
        (run_dir / "outputs/base").mkdir(parents=True, exist_ok=True)
        status.update(phase="generate_base_test")
        save()
        cmd = [sys.executable, "gen_outputs.py", "--model-dir", str(STUDENT), "--prompts", str(HERE / "prompts_test_v1.jsonl"),
               "--out", str(run_dir / "outputs/base/test.jsonl")] + (["--limit", str(args.gen_limit)] if args.gen_limit else [])
        with open(run_dir / "outputs/base/test.log", "w") as log:
            subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        event("generated base test")

    # --- final upload ---------------------------------------------------------------------------------------------
    status["phase"] = "final_upload"
    save()
    if not drive_sync.wait_for_uploads(max_seconds=args.upload_wait_minutes * 60):
        event(f"WARNING: checkpoint uploads still running after {args.upload_wait_minutes} min; stopping the wait")
    status["drive_full_steps"] = drive_sync.drive_complete_steps(args.drive_root)
    missing = [s for s in range(args.save_steps, max_steps + 1, args.save_steps) if s not in status["drive_full_steps"]]
    if missing:
        event(f"WARNING: full checkpoints not on Drive: {missing}")
    for item in ["outputs", "run_report.json", "attempts.jsonl", "train.log", "uploads.log", "progress.json"]:
        src = run_dir / item
        if src.exists():
            dest = f"{args.drive_root}/{item}" if src.is_dir() else f"{args.drive_root}/logs/"
            drive_sync.rclone("copy", str(src), dest)
    for extra in [HERE / "proxy.log", HERE / "tinker_usage.json", status_path]:
        if extra.exists():
            drive_sync.rclone("copy", str(extra), f"{args.drive_root}/logs/")
    status.update(phase="done", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    save()
    drive_sync.rclone("copy", str(status_path), f"{args.drive_root}/logs/")
    (HERE / f"DONE_{args.method}").write_text("done")
    event("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
