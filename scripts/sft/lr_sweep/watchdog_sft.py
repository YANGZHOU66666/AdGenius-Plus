"""Detached watchdog for the SFT learning-rate sweep on a Colab L4 (runs on the Mac, survives Claude sessions).

Start (from the repository root, after scripts/build_training_prompts.py and scripts/sft/build_sft_data.py):
    nohup caffeinate -i python3 scripts/sft/lr_sweep/watchdog_sft.py --work-dir artifacts/train/<experiment> \
        > /dev/null 2>&1 &

Every 10 minutes:
- Colab unreachable (e.g. the Mac is offline): skip this check; only an explicit "no such session" is a reclaim;
- session alive: copy finished runs (dev.jsonl, run_report.json, logs) into results/; if the kernel went idle before
  finishing, restart the blocking cell for the missing runs; if the VM finished with runs missing, archive the VM
  logs and retry them once;
- session gone: acquire an L4 (retry each minute), install, upload, start the blocking cell for the missing runs;
- every run finished (dev outputs synced, or diverged): archive VM logs, stop the session, exit.
All lines go to <work-dir>/watchdog.log. No API key is needed or uploaded. Files are uploaded flat to /content/sweep/
from scripts/, scripts/sft/, this directory and the data directories.
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # scripts/sft/lr_sweep
SFT_DIR = HERE.parent                           # scripts/sft: training code shared with the formal runs
SCRIPTS = HERE.parents[1]                       # scripts: dev generation shared with OPD
REPO_ROOT = HERE.parents[2]
SESSION, GPU = "sft", "L4"
LRS = ("5e-6", "1e-5", "2e-5")
FILES = ["sft_train.py", "vm_runner_sft.py", "sweep_cell_sft.py", "dev_generate.py", "sft_train_v1.jsonl",
         "prompts_dev_v1.jsonl", "drive_sync.py"]
# Set from the command line in main().
WORK = RESULTS = LOG = PROMPTS = SFT_DATA = None


def source(name):
    return {"sft_train.py": SFT_DIR, "dev_generate.py": SCRIPTS, "sft_train_v1.jsonl": SFT_DATA,
            "prompts_dev_v1.jsonl": PROMPTS, "drive_sync.py": SCRIPTS / "formal"}.get(name, HERE) / name
INSTALL = ("%%bash\nmkdir -p /content/sweep/runs && cd /content\n"
           "python -m pip install -q \"trl[vllm]==1.18.0\" \"transformers==5.23.0\" datasets accelerate "
           "flash-linear-attention > /content/pip_install.log 2>&1; echo \"pip exit $?\"\n")
CHECK_SECONDS = 600
DEADLINE = 0.0
MAX_LAUNCHES, MAX_RETRIES = 4, 1


def log(msg):
    with LOG.open("a") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + msg + "\n")


def colab(*args, timeout=None, stdin=None):
    try:
        r = subprocess.run(["colab", "--auth=oauth2", *args], capture_output=True, text=True, timeout=timeout, input=stdin)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def name(lr):
    return f"sft_lr{float(lr):g}"


def finished(lr):
    d = RESULTS / name(lr)
    if (d / "dev.jsonl").exists():
        return True
    report = d / "run_report.json"
    return report.exists() and json.loads(report.read_text()).get("diverged") is True


def missing():
    return [lr for lr in LRS if not finished(lr)]


def alive():
    """True/False from an answer of the Colab server; None when it could not be reached."""
    code, out = colab("sessions", timeout=120)
    if code != 0:
        return None
    return f"[{SESSION}]" in out


def listdir(path):
    code, out = colab("ls", "-s", SESSION, path, timeout=120)
    if code != 0:
        return set()
    return {line.split()[-1].rstrip("/").split("/")[-1] for line in out.splitlines() if line.split()}


def complete(path):
    try:
        ok = sum(1 for line in path.open() if line.strip()) == 200
    except OSError:
        ok = False
    if not ok:
        path.unlink(missing_ok=True)
    return ok


def sync():
    """Copy finished runs from the VM; return True when the VM wrote ALL_DONE."""
    entries = listdir("/content/sweep/runs")
    for lr in missing():
        n = name(lr)
        if n not in entries:
            continue
        files = listdir(f"/content/sweep/runs/{n}")
        if "run_report.json" not in files:
            continue
        dest = RESULTS / n
        dest.mkdir(parents=True, exist_ok=True)
        for f in ("run_report.json", "train.log", "dev.log"):
            if f in files:
                colab("download", "-s", SESSION, f"/content/sweep/runs/{n}/{f}", str(dest / f), timeout=300)
        if json.loads((dest / "run_report.json").read_text()).get("diverged"):
            log(f"synced {n} (diverged: excluded, no dev outputs)")
            continue
        if "dev.jsonl" in files:
            colab("download", "-s", SESSION, f"/content/sweep/runs/{n}/dev.jsonl", str(dest / "dev.jsonl"), timeout=300)
            if complete(dest / "dev.jsonl"):
                log(f"synced {n}")
                continue
        (dest / "run_report.json").unlink(missing_ok=True)  # training done, dev not yet: fetch again next time
    top = listdir("/content/sweep")
    for f in sorted(top):
        if f.startswith(("status_", "runner", "env_")):
            colab("download", "-s", SESSION, f"/content/sweep/{f}", str(RESULTS / "vm" / f), timeout=120)
    return "ALL_DONE" in top


def archive(tag):
    dest = RESULTS / f"vm_archive_{tag}"
    (dest / "runs").mkdir(parents=True, exist_ok=True)
    for f in listdir("/content/sweep"):
        if f.startswith(("status_", "runner", "env_", "pending")):
            colab("download", "-s", SESSION, f"/content/sweep/{f}", str(dest / f), timeout=300)
    for lr in missing():
        n = name(lr)
        if "train.log" in listdir(f"/content/sweep/runs/{n}"):
            (dest / "runs" / n).mkdir(exist_ok=True)
            colab("download", "-s", SESSION, f"/content/sweep/runs/{n}/train.log", str(dest / "runs" / n / "train.log"),
                  timeout=300)
    log(f"archived VM logs to {dest.name}")


def progress():
    s = RESULTS / "vm" / "status_sft.json"
    try:
        st = json.loads(s.read_text())
        return ", ".join(f"{k}={v.get('phase')}" for k, v in st.get("runs", {}).items()) or "starting"
    except (OSError, json.JSONDecodeError):
        return "no status yet"


def upload():
    for f in FILES:
        colab("upload", "-s", SESSION, str(source(f)), f"/content/sweep/{f}", timeout=900)


def kernel_busy():
    code, out = colab("status", "-s", SESSION, timeout=120)
    return code != 0 or "Status: BUSY" in out  # errors count as busy so a flaky call never double-starts


def start_cell():
    colab("upload", "-s", SESSION, str(HERE / "sweep_cell_sft.py"), "/content/sweep/sweep_cell_sft.py", timeout=120)
    pending = Path(tempfile.mkdtemp()) / "pending.json"
    pending.write_text(json.dumps({"lrs": missing()}))
    colab("upload", "-s", SESSION, str(pending), "/content/sweep/pending.json", timeout=120)
    cell_log = open(WORK / "cell_exec.log", "a")
    subprocess.Popen(["colab", "--auth=oauth2", "exec", "-s", SESSION, "-f", str(HERE / "sweep_cell_sft.py"),
                      "--timeout", "0"], stdout=cell_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     start_new_session=True)
    log(f"started kernel cell for {missing()}")


def launch():
    for attempt in range(1, 31):
        colab("new", "-s", SESSION, "--gpu", GPU, timeout=600)
        if alive():
            log(f"acquired {GPU} on attempt {attempt}")
            break
        log(f"no {GPU} (attempt {attempt})")
        time.sleep(60)
    else:
        return False
    code, out = colab("exec", "-s", SESSION, "--timeout", "0", stdin=INSTALL)
    log("install: " + ("ok" if "pip exit 0" in out else "FAILED"))
    upload()
    start_cell()
    return True


def main():
    global WORK, RESULTS, LOG, PROMPTS, SFT_DATA, SESSION, GPU, LRS, DEADLINE
    parser = argparse.ArgumentParser(description="Watch the SFT sweep on Colab and relaunch missing runs.")
    parser.add_argument("--work-dir", type=Path, required=True, help="results/, logs and VM archives go here")
    parser.add_argument("--prompts-dir", type=Path, default=REPO_ROOT / "artifacts/train/prompts/v1")
    parser.add_argument("--sft-data-dir", type=Path, default=REPO_ROOT / "artifacts/train/sft_data/v1")
    parser.add_argument("--session", default=SESSION)
    parser.add_argument("--gpu", default=GPU)
    parser.add_argument("--lrs", nargs="+", default=list(LRS))
    parser.add_argument("--hours", type=float, default=4.0)
    args = parser.parse_args()
    WORK, PROMPTS, SFT_DATA = args.work_dir.resolve(), args.prompts_dir.resolve(), args.sft_data_dir.resolve()
    RESULTS, LOG = WORK / "results", WORK / "watchdog.log"
    SESSION, GPU, LRS = args.session, args.gpu, tuple(args.lrs)
    DEADLINE = time.time() + args.hours * 3600
    for f in FILES:
        if not source(f).exists():
            raise SystemExit(f"missing {source(f)}")
    (RESULTS / "vm").mkdir(parents=True, exist_ok=True)
    log(f"watchdog started; missing: {[name(lr) for lr in missing()]}")
    launches = retries = 0
    while time.time() < DEADLINE:
        if not missing():
            archive("final")
            log("ALL RESULTS SYNCED; stopping session")
            colab("stop", "-s", SESSION, timeout=300)
            log("DONE")
            return
        state = alive()
        if state is None:
            log("cannot reach Colab (network?); skipping this check")
        elif state:
            all_done = sync()
            log("alive; " + progress())
            if not missing():
                continue
            if all_done:
                if retries < MAX_RETRIES:
                    retries += 1
                    archive(f"before_retry{retries}")
                    log(f"VM finished with runs missing; retry {retries}/{MAX_RETRIES} for {missing()}")
                    upload()
                    start_cell()
                else:
                    log(f"FAILED: runs still missing after retry: {missing()}; stopping session")
                    archive("failed")
                    colab("stop", "-s", SESSION, timeout=300)
                    return
            elif not kernel_busy():
                time.sleep(60)
                if not kernel_busy():
                    log(f"kernel IDLE before finishing; restarting the cell for {missing()}")
                    start_cell()
        else:
            if launches >= MAX_LAUNCHES:
                log(f"FAILED: session lost {launches} times; giving up")
                return
            launches += 1
            log(f"session not found (reclaimed or not started); launch {launches} for {missing()}")
            if not launch():
                log(f"could not acquire an {GPU} for 30 minutes; will retry next cycle")
        time.sleep(CHECK_SECONDS)
    log("DEADLINE reached; stopping session")
    colab("stop", "-s", SESSION, timeout=300)


if __name__ == "__main__":
    main()
