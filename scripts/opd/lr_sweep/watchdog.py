"""Detached watchdog for an OPD sweep on Colab (runs on the Mac, survives Claude sessions).

Start it detached, for example:
    nohup caffeinate -i python3 scripts/opd/lr_sweep/watchdog.py --work-dir artifacts/train/<experiment> > /dev/null 2>&1 &

Every 10 minutes:
- if the Colab session is alive: copy finished runs (dev.jsonl, run_report.json, train.log) and the base dev
  outputs into results/, log progress and Tinker usage;
- if the session is gone (reclaimed) and runs are still missing: acquire an A100 (retry each minute), install,
  upload, write pending.json with only the missing runs, upload the one-time key and start the blocking kernel cell;
- when every run is in results/: stop the session and exit.
If Colab cannot be reached (for example the Mac is offline) the check is skipped; only an explicit
"no such session" answer counts as a reclaim. All lines go to <work-dir>/watchdog.log. Never prints the key or
session URLs. Code is uploaded flat to /content/sweep/ from scripts/, scripts/opd/ and this directory, and the
frozen prompts from --prompts-dir.
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # scripts/opd/lr_sweep
OPD_DIR = HERE.parent                           # scripts/opd: training code shared with the formal runs
SCRIPTS = HERE.parents[1]                       # scripts: dev generation shared with SFT
REPO_ROOT = HERE.parents[2]
ENV_FILE = REPO_ROOT / ".env"
CODE_FILES = ["opd_train.py", "dev_generate.py", "vm_runner.py", "teacher_proxy.py", "completion_logits_trainer.py",
              "sweep_cell.py"]
PROMPT_FILES = ["prompts_train_v1.jsonl", "prompts_dev_v1.jsonl"]
CHECK_SECONDS = 600
# Set from the command line in main().
WORK = RESULTS = LOG = PROMPTS = None
SESSION, GPU = "sweep", "A100"
BETAS, LRS = ("1.0", "0.5"), ("5e-7", "1e-6", "2e-6")
DEADLINE = 0.0


def source(name):
    if name in PROMPT_FILES:
        return PROMPTS / name
    if name in ("opd_train.py", "completion_logits_trainer.py", "teacher_proxy.py"):
        return OPD_DIR / name
    if name == "dev_generate.py":
        return SCRIPTS / name
    return HERE / name


def upload_files():
    for f in CODE_FILES + PROMPT_FILES:
        colab("upload", "-s", SESSION, str(source(f)), f"/content/sweep/{f}", timeout=600)


def log(msg):
    with LOG.open("a") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + msg + "\n")


def colab(*args, timeout=None):
    try:
        r = subprocess.run(["colab", "--auth=oauth2", *args], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def run_name(beta, lr):
    return f"beta{float(beta):g}_lr{float(lr):g}"


def missing_runs():
    return [(b, lr) for b in BETAS for lr in LRS if not (RESULTS / run_name(b, lr) / "dev.jsonl").exists()]


def base_missing():
    return not (RESULTS / "base_dev.jsonl").exists()


def alive():
    """True/False from an answer of the Colab server; None when it could not be reached."""
    code, out = colab("sessions", timeout=120)
    if code != 0:
        return None
    return f"[{SESSION}]" in out


def listdir(path):
    """Names in a VM directory (last token of each line, trailing slash removed); empty set on error."""
    code, out = colab("ls", "-s", SESSION, path, timeout=120)
    if code != 0:
        return set()
    return {line.split()[-1].rstrip("/").split("/")[-1] for line in out.splitlines() if line.split()}


def complete(path):
    """Keep a downloaded dev file only if it has all 200 rows (it may have been mid-write)."""
    try:
        ok = sum(1 for line in path.open() if line.strip()) == 200
    except OSError:
        ok = False
    if not ok:
        path.unlink(missing_ok=True)
    return ok


def sync():
    """Copy finished runs and base dev outputs from the VM."""
    entries = listdir("/content/sweep/runs")
    if "base_dev.jsonl" in entries and base_missing():
        RESULTS.mkdir(exist_ok=True)
        colab("download", "-s", SESSION, "/content/sweep/runs/base_dev.jsonl", str(RESULTS / "base_dev.jsonl"),
              timeout=300)
        if complete(RESULTS / "base_dev.jsonl"):
            log("synced base_dev.jsonl")
    for beta, lr in missing_runs():
        name = run_name(beta, lr)
        if name not in entries:
            continue
        if "dev.jsonl" not in listdir(f"/content/sweep/runs/{name}"):
            continue
        dest = RESULTS / name
        dest.mkdir(parents=True, exist_ok=True)
        for f in ("run_report.json", "train.log", "dev.log"):
            colab("download", "-s", SESSION, f"/content/sweep/runs/{name}/{f}", str(dest / f), timeout=300)
        colab("download", "-s", SESSION, f"/content/sweep/runs/{name}/dev.jsonl", str(dest / "dev.jsonl"), timeout=300)
        if complete(dest / "dev.jsonl"):
            log(f"synced {name}")
    top = listdir("/content/sweep")
    for f in sorted(top):
        if f.startswith(("tinker_usage_", "status_beta", "runner_", "proxy_", "env_")):
            colab("download", "-s", SESSION, f"/content/sweep/{f}", str(RESULTS / "vm" / f), timeout=120)
    return "ALL_DONE" in top


def archive(tag):
    """Keep VM-only logs that a restarted cell would overwrite (proxy logs, status, failed runs' train.log)."""
    dest = RESULTS / f"vm_archive_{tag}"
    (dest / "runs").mkdir(parents=True, exist_ok=True)
    for f in listdir("/content/sweep"):
        if f.startswith(("tinker_usage_", "status_beta", "runner_", "proxy_", "env_", "pending")):
            colab("download", "-s", SESSION, f"/content/sweep/{f}", str(dest / f), timeout=300)
    for beta, lr in missing_runs():
        name = run_name(beta, lr)
        if "train.log" in listdir(f"/content/sweep/runs/{name}"):
            (dest / "runs" / name).mkdir(exist_ok=True)
            colab("download", "-s", SESSION, f"/content/sweep/runs/{name}/train.log", str(dest / "runs" / name / "train.log"),
                  timeout=300)
    log(f"archived VM logs to {dest.name}")


def progress_line():
    parts = []
    for local in sorted((RESULTS / "vm").glob("status_beta*.json")):
        try:
            s = json.loads(local.read_text())
        except json.JSONDecodeError:
            continue
        runs = ", ".join(f"{k}={v.get('phase')}" for k, v in s.get("runs", {}).items())
        parts.append(f"{local.stem}: base_dev={s.get('base_dev', '-')} {runs} error={s.get('error')}")
    for local in sorted((RESULTS / "vm").glob("tinker_usage_*.json")):
        try:
            parts.append(f"{local.stem}: {json.loads(local.read_text())}")
        except json.JSONDecodeError:
            pass
    return " | ".join(parts) or "no status yet"


def launch():
    for attempt in range(1, 31):
        code, out = colab("new", "-s", SESSION, "--gpu", GPU, timeout=600)
        if alive():
            log(f"acquired {GPU} on attempt {attempt}")
            break
        log(f"no {GPU} (attempt {attempt})")
        time.sleep(60)
    else:
        return False
    install = ("%%bash\nmkdir -p /content/sweep/runs && cd /content\n"
               "python -m pip install -q \"trl[vllm]==1.18.0\" \"transformers==5.23.0\" datasets accelerate tinker "
               "flash-linear-attention bitsandbytes > /content/pip_install.log 2>&1; echo \"pip exit $?\"\n")
    r = subprocess.run(["colab", "--auth=oauth2", "exec", "-s", SESSION, "--timeout", "0"], input=install,
                       capture_output=True, text=True)
    log("install: " + ("ok" if "pip exit 0" in r.stdout else "FAILED"))
    upload_files()
    start_cell()
    return True


def kernel_busy():
    code, out = colab("status", "-s", SESSION, timeout=120)
    return code != 0 or "Status: BUSY" in out  # treat errors as busy so a flaky call never double-starts


def start_cell():
    colab("upload", "-s", SESSION, str(HERE / "sweep_cell.py"), "/content/sweep/sweep_cell.py", timeout=120)
    groups = []
    for beta in BETAS:
        lrs = [lr for b, lr in missing_runs() if b == beta]
        if lrs:
            groups.append({"beta": beta, "lrs": lrs, "base_dev": beta == "1.0" and base_missing()})
    if base_missing() and not any(g.get("base_dev") for g in groups) and groups:
        groups[0]["base_dev"] = True
    pending = Path(tempfile.mkdtemp()) / "pending.json"
    pending.write_text(json.dumps({"groups": groups}))
    colab("upload", "-s", SESSION, str(pending), "/content/sweep/pending.json", timeout=120)
    with tempfile.NamedTemporaryFile("w", delete=False) as k:
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("TINKER_API_KEY="):
                k.write(line.split("=", 1)[1].strip().strip('"').strip("'"))
    Path(k.name).chmod(0o600)
    colab("upload", "-s", SESSION, k.name, "/content/sweep/.tinker_key", timeout=120)
    Path(k.name).unlink()
    # The blocking cell keeps the kernel busy; start it detached so this watchdog keeps checking.
    cell_log = open(WORK / "cell_exec.log", "a")
    subprocess.Popen(["colab", "--auth=oauth2", "exec", "-s", SESSION, "-f", str(HERE / "sweep_cell.py"),
                      "--timeout", "0"], stdout=cell_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     start_new_session=True)
    log(f"started kernel cell for {json.dumps(groups)}")


MAX_LAUNCHES = 4
MAX_RETRIES = 1


def main():
    global WORK, RESULTS, LOG, PROMPTS, SESSION, GPU, BETAS, LRS, DEADLINE
    parser = argparse.ArgumentParser(description="Watch an OPD sweep on Colab and relaunch missing runs.")
    parser.add_argument("--work-dir", type=Path, required=True, help="results/, logs and VM archives go here")
    parser.add_argument("--prompts-dir", type=Path, default=REPO_ROOT / "artifacts/train/prompts/v1",
                        help="output of build_training_prompts.py")
    parser.add_argument("--session", default=SESSION)
    parser.add_argument("--gpu", default=GPU)
    parser.add_argument("--betas", nargs="+", default=list(BETAS))
    parser.add_argument("--lrs", nargs="+", default=list(LRS))
    parser.add_argument("--hours", type=float, default=9.0, help="stop the session after this many hours")
    args = parser.parse_args()
    WORK, PROMPTS = args.work_dir.resolve(), args.prompts_dir.resolve()
    RESULTS, LOG = WORK / "results", WORK / "watchdog.log"
    SESSION, GPU, BETAS, LRS = args.session, args.gpu, tuple(args.betas), tuple(args.lrs)
    DEADLINE = time.time() + args.hours * 3600
    for f in PROMPT_FILES:
        if not source(f).exists():
            raise SystemExit(f"missing {source(f)}; run build_training_prompts.py first")
    (RESULTS / "vm").mkdir(parents=True, exist_ok=True)
    log("watchdog started; missing runs: " + ", ".join(run_name(*r) for r in missing_runs()))
    launches = 0
    retries = 0
    while time.time() < DEADLINE:
        if not missing_runs() and not base_missing():
            archive("final")
            log("ALL RESULTS SYNCED; stopping session")
            colab("stop", "-s", SESSION, timeout=300)
            log("DONE")
            return
        state = alive()
        if state is None:
            log("cannot reach Colab (network?); skipping this check")
            time.sleep(CHECK_SECONDS)
            continue
        if state:
            all_done = sync()
            log("alive; " + progress_line())
            if all_done and (missing_runs() or base_missing()):
                if retries < MAX_RETRIES:
                    retries += 1
                    archive(f"before_retry{retries}")
                    log(f"VM finished with runs missing; retry {retries}/{MAX_RETRIES} for: "
                        + ", ".join(run_name(*r) for r in missing_runs()))
                    upload_files()
                    start_cell()
                    time.sleep(CHECK_SECONDS)
                    continue
                log("FAILED: VM finished but runs are missing: " + ", ".join(run_name(*r) for r in missing_runs())
                    + "; stopping session to save compute units")
                colab("stop", "-s", SESSION, timeout=300)
                return
            if not missing_runs() and not base_missing():
                continue
            if not all_done and not kernel_busy():
                # Kernel went idle without finishing (e.g. it crashed): an idle VM gets reclaimed in ~20 min,
                # so restart the cell now for the runs still missing instead of waiting for the reclaim.
                time.sleep(60)
                if not kernel_busy():
                    log("kernel IDLE before finishing; restarting the cell for: "
                        + ", ".join(run_name(*r) for r in missing_runs()))
                    start_cell()
        else:
            if launches >= MAX_LAUNCHES:
                log(f"FAILED: session lost {launches} times; giving up")
                return
            launches += 1
            log(f"session not found (reclaimed or not started); launch {launches} for: "
                + ", ".join(run_name(*r) for r in missing_runs()))
            if not launch():
                log(f"could not acquire an {GPU} for 30 minutes; will retry next cycle")
        time.sleep(CHECK_SECONDS)
    log("DEADLINE reached; stopping session")
    colab("stop", "-s", SESSION, timeout=300)


if __name__ == "__main__":
    main()
