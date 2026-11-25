"""Detached watchdog for one formal run on Colab (runs on the Mac; one process per method).

    nohup caffeinate -i python3 scripts/formal/watchdog_formal.py --method opd > /dev/null 2>&1 &
    nohup caffeinate -i python3 scripts/formal/watchdog_formal.py --method sft > /dev/null 2>&1 &

Every 10 minutes:
- Colab unreachable (e.g. the Mac is offline): skip; only an explicit "no such session" counts as a reclaim;
- session alive: copy status, logs and finished outputs into <work-dir>/<name>/; if the VM wrote DONE, verify the
  outputs, stop the session and exit; if the cell ended without DONE, restart it (it resumes by itself) up to
  MAX_CELL_RESTARTS times; if the kernel went idle, restart the cell at once;
- session gone: acquire a GPU (fallbacks in order, ~30 min each), install, upload code, data and secrets, start the
  blocking cell. The VM side (run_formal.py) resumes from the newest checkpoint on the VM or on Drive.
All lines go to <work-dir>/<name>/watchdog_<method>.log (default work dir: artifacts/train/). Secrets (Tinker key, rclone config) are passed as owner-only temp files and
never printed; session URLs are not logged.
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # scripts/formal
SCRIPTS = HERE.parent
REPO = HERE.parents[1]
RCLONE_CONF = Path.home() / ".config/rclone/rclone.conf"
COMMON = ["drive_sync.py", "run_formal.py", "formal_cell.py", "gen_outputs.py", "prompts_dev_v1.jsonl", "prompts_test_v1.jsonl"]
METHODS = {
    "opd": {"session": "opd", "gpus": ["A100", "H100"], "name": "opd_formal_v1",
            "files": COMMON + ["opd_train.py", "completion_logits_trainer.py", "teacher_proxy.py", "prompts_train_v1.jsonl"],
            "pip": "\"trl[vllm]==1.18.0\" \"transformers==5.23.0\" datasets accelerate tinker flash-linear-attention",
            "args": ["--method", "opd", "--name", "opd_formal_v1", "--lr", "2e-6", "--beta", "0.5",
                     "--drive-root", "gdrive:adgenius_checkpoints/opd_formal_v1"], "tinker": True},
    "sft": {"session": "sft", "gpus": ["L4", "A100"], "name": "sft_formal_v1",
            "files": COMMON + ["sft_train.py", "sft_train_v1.jsonl"],
            "pip": "\"trl[vllm]==1.18.0\" \"transformers==5.23.0\" datasets accelerate flash-linear-attention",
            "args": ["--method", "sft", "--name", "sft_formal_v1", "--lr", "5e-6",
                     "--drive-root", "gdrive:adgenius_checkpoints/sft_formal_v1", "--base-test"], "tinker": False},
}
CHECK_SECONDS = 600
MAX_LAUNCHES, MAX_CELL_RESTARTS = 5, 3
GPU_MINUTES = 30
M = None  # current method config
LOG = RESULTS = PROMPTS = SFT_DATA = None


def source(name):
    """Local path of a file uploaded to /content/run (the VM keeps them flat)."""
    if name.startswith("prompts_"):
        return PROMPTS / name
    if name == "sft_train_v1.jsonl":
        return SFT_DATA / name
    if name in ("opd_train.py", "completion_logits_trainer.py", "teacher_proxy.py"):
        return SCRIPTS / "opd" / name
    if name == "sft_train.py":
        return SCRIPTS / "sft" / name
    return HERE / name


def log(msg):
    with LOG.open("a") as f:
        f.write(time.strftime("%m-%d %H:%M:%S ") + msg + "\n")


def colab(*args, timeout=None, stdin=None):
    try:
        r = subprocess.run(["colab", "--auth=oauth2", *args], capture_output=True, text=True, timeout=timeout, input=stdin)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def alive():
    code, out = colab("sessions", timeout=120)
    if code != 0:
        return None
    return f"[{M['session']}]" in out


def listdir(path):
    code, out = colab("ls", "-s", M["session"], path, timeout=120)
    if code != 0:
        return set()
    return {line.split()[-1].rstrip("/").split("/")[-1] for line in out.splitlines() if line.split()}


def fetch(remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    return colab("download", "-s", M["session"], remote, str(local), timeout=600)[0] == 0


def secret_file(text):
    with tempfile.NamedTemporaryFile("w", delete=False) as k:
        k.write(text)
    Path(k.name).chmod(0o600)
    return Path(k.name)


def upload_secrets():
    conf = secret_file(RCLONE_CONF.read_text())
    colab("upload", "-s", M["session"], str(conf), "/content/run/.rclone.conf", timeout=120)
    conf.unlink()
    if M["tinker"]:
        key = ""
        for line in (REPO / ".env").read_text().splitlines():
            if line.startswith("TINKER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
        k = secret_file(key)
        colab("upload", "-s", M["session"], str(k), "/content/run/.tinker_key", timeout=120)
        k.unlink()


def upload_code():
    for f in M["files"]:
        if colab("upload", "-s", M["session"], str(source(f)), f"/content/run/{f}", timeout=900)[0] != 0:
            log(f"upload failed: {f}")
    cfg = secret_file(json.dumps({"method": M["key"], "args": M["args"]}))
    colab("upload", "-s", M["session"], str(cfg), "/content/run/run_config.json", timeout=120)
    cfg.unlink()


def install():
    script = ("%%bash\nmkdir -p /content/run && cd /content\n"
              "(apt-get update -qq && apt-get install -y -qq rclone) > /content/apt_install.log 2>&1; echo \"apt exit $?\"\n"
              f"python -m pip install -q {M['pip']} > /content/pip_install.log 2>&1; echo \"pip exit $?\"\n")
    code, out = colab("exec", "-s", M["session"], "--timeout", "0", stdin=script)
    ok = "apt exit 0" in out and "pip exit 0" in out
    log("install: " + ("ok" if ok else "FAILED " + out[-300:].replace("\n", " ")))
    return ok


def start_cell():
    upload_secrets()
    cell_log = open(RESULTS / "cell_exec.log", "a")
    subprocess.Popen(["colab", "--auth=oauth2", "exec", "-s", M["session"], "-f", str(HERE / "formal_cell.py"),
                      "--timeout", "0"], stdout=cell_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     start_new_session=True)
    log("started the blocking cell")


def kernel_busy():
    code, out = colab("status", "-s", M["session"], timeout=120)
    return code != 0 or "Status: BUSY" in out


def acquire():
    """Try each GPU in order for GPU_MINUTES (a failed request alone can take ~4 minutes)."""
    for i, gpu in enumerate(M["gpus"]):
        budget = M.get("first_gpu_minutes", GPU_MINUTES) if i == 0 else GPU_MINUTES
        start, attempt = time.time(), 0
        while time.time() - start < budget * 60:
            attempt += 1
            colab("new", "-s", M["session"], "--gpu", gpu, timeout=600)
            if alive():
                log(f"acquired {gpu} on attempt {attempt}")
                return gpu
            log(f"no {gpu} (attempt {attempt}, {round((time.time() - start) / 60)} min)")
            time.sleep(60)
    return None


def launch():
    gpu = acquire()
    if not gpu:
        return False
    if not install():
        return False
    upload_code()
    start_cell()
    return True


def sync():
    """Copy status, logs and outputs; return (done, cell_exited)."""
    top = listdir("/content/run")
    for f in sorted(top):
        if f.startswith(("status_", "runner_", "env_", "proxy", "tinker_usage", "CELL_EXIT", "DONE")):
            fetch(f"/content/run/{f}", RESULTS / "vm" / f)
    run = f"/content/run/runs/{M['name']}"
    for f in listdir(run):
        if f in ("run_report.json", "attempts.jsonl", "train.log", "uploads.log", "progress.json"):
            fetch(f"{run}/{f}", RESULTS / f)
    for scale in listdir(f"{run}/outputs"):
        for f in listdir(f"{run}/outputs/{scale}"):
            if f.endswith((".jsonl", ".log")):
                fetch(f"{run}/outputs/{scale}/{f}", RESULTS / "outputs" / scale / f)
    return f"DONE_{M['key']}" in top, f"CELL_EXIT_{M['key']}" in top


def progress():
    parts = []
    try:
        st = json.loads((RESULTS / "vm" / f"status_{M['key']}.json").read_text())
        parts.append(f"phase={st.get('phase')} attempts={len(st.get('attempts', []))}")
    except (OSError, json.JSONDecodeError):
        parts.append("no status yet")
    try:
        pr = json.loads((RESULTS / "progress.json").read_text())
        parts.append(f"step {pr['step']}/{pr['max_steps']} loss {pr['loss']:.4f} lr {pr['learning_rate']:.2e}")
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass
    try:
        parts.append(f"tinker {json.loads((RESULTS / 'vm' / 'tinker_usage.json').read_text()).get('tokens_sent')}")
    except (OSError, json.JSONDecodeError):
        pass
    return " | ".join(parts)


UPLOAD_WAIT = {"opd": 120, "sft": 240}  # minutes the VM may stay up after DONE to finish Drive uploads
WAIT = {"started": None}


def uploads_finished():
    """True when the VM reported its upload pass, or every checkpoint and bf16 export is already on Drive."""
    if f"UPLOADS_{M['key']}.json" in listdir("/content/run"):
        fetch(f"/content/run/UPLOADS_{M['key']}.json", RESULTS / "vm" / f"UPLOADS_{M['key']}.json")
        return True
    root = M["args"][M["args"].index("--drive-root") + 1]
    r = subprocess.run(["rclone", "lsf", f"{root}/full/", "--dirs-only"], capture_output=True, text=True)
    steps = [d for d in r.stdout.split() if d.startswith("step_")]
    complete = sum("UPLOAD_OK" in subprocess.run(["rclone", "lsf", f"{root}/full/{d}"], capture_output=True,
                                                  text=True).stdout for d in steps)
    w = subprocess.run(["rclone", "lsf", f"{root}/weights_bf16/", "--dirs-only"], capture_output=True, text=True)
    return complete >= 8 and len(w.stdout.split()) >= 4


def wait_for_drive():
    """Start (once) a busy cell that finishes the uploads; return True while we should keep waiting."""
    if WAIT["started"] is None:
        cfg = secret_file(json.dumps({"name": M["name"], "method": M["key"], "max_minutes": UPLOAD_WAIT[M["key"]],
                                      "drive_root": M["args"][M["args"].index("--drive-root") + 1]}))
        colab("upload", "-s", M["session"], str(cfg), "/content/run/wait_config.json", timeout=120)
        cfg.unlink()
        colab("upload", "-s", M["session"], str(HERE / "wait_uploads_cell.py"), "/content/run/wait_uploads_cell.py",
              timeout=120)
        cell_log = open(RESULTS / "cell_exec.log", "a")
        subprocess.Popen(["colab", "--auth=oauth2", "exec", "-s", M["session"], "-f", str(HERE / "wait_uploads_cell.py"),
                          "--timeout", "0"], stdout=cell_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
        WAIT["started"] = time.time()
        log(f"DONE on the VM; keeping it busy up to {UPLOAD_WAIT[M['key']]} min to finish the Drive uploads")
        return True
    if time.time() - WAIT["started"] > (UPLOAD_WAIT[M["key"]] + 15) * 60:
        log("upload wait exceeded; stopping anyway")
        return False
    log("waiting for Drive uploads to finish")
    return True


def verify_outputs():
    expected = {"dev.jsonl": 200, "test.jsonl": 1000}
    problems = []
    for scale in ("scale_1k", "scale_2k", "scale_4k", "scale_8k"):
        for f, n in expected.items():
            p = RESULTS / "outputs" / scale / f
            got = sum(1 for line in p.open() if line.strip()) if p.exists() else 0
            if got != n:
                problems.append(f"{scale}/{f}: {got}/{n}")
    if M["key"] == "sft":
        p = RESULTS / "outputs" / "base" / "test.jsonl"
        if not p.exists() or sum(1 for line in p.open() if line.strip()) != 1000:
            problems.append("base/test.jsonl incomplete")
    return problems


def main():
    global M, LOG, RESULTS, PROMPTS, SFT_DATA
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=list(METHODS), required=True)
    p.add_argument("--hours", type=float, default=12)
    p.add_argument("--work-dir", type=Path, default=REPO / "artifacts/train",
                   help="local copies of the VM status, logs and outputs go to <work-dir>/<name>/")
    p.add_argument("--prompts-dir", type=Path, default=REPO / "artifacts/train/prompts/v1")
    p.add_argument("--sft-data-dir", type=Path, default=REPO / "artifacts/train/sft_data/v1")
    a = p.parse_args()
    M = dict(METHODS[a.method], key=a.method)
    PROMPTS, SFT_DATA = a.prompts_dir, a.sft_data_dir
    missing = [f for f in M["files"] if not source(f).exists()]
    if missing:
        raise SystemExit(f"missing local files: {missing}")
    RESULTS = a.work_dir / M["name"]
    (RESULTS / "vm").mkdir(parents=True, exist_ok=True)
    LOG = RESULTS / f"watchdog_{a.method}.log"
    deadline = time.time() + a.hours * 3600
    launches = restarts = 0
    log(f"watchdog started for {M['name']}")
    while time.time() < deadline:
        state = alive()
        if state is None:
            log("cannot reach Colab (network?); skipping this check")
        elif state:
            done, exited = sync()
            log("alive; " + progress())
            if done:
                sync()
                if not uploads_finished():
                    continue_waiting = wait_for_drive()
                    if continue_waiting:
                        time.sleep(CHECK_SECONDS)
                        continue
                problems = verify_outputs()
                log("DONE on the VM; outputs " + ("complete" if not problems else f"INCOMPLETE {problems}"))
                colab("stop", "-s", M["session"], timeout=300)
                log("session stopped; watchdog exiting")
                return
            if exited:
                if restarts >= MAX_CELL_RESTARTS:
                    log(f"FAILED: the run ended without DONE {restarts + 1} times; stopping the session")
                    colab("stop", "-s", M["session"], timeout=300)
                    return
                restarts += 1
                log(f"cell ended without DONE; restart {restarts}/{MAX_CELL_RESTARTS} (it resumes from the newest checkpoint)")
                start_cell()
            elif not kernel_busy():
                time.sleep(60)
                if not kernel_busy():
                    log("kernel IDLE; restarting the cell")
                    start_cell()
        else:
            if launches >= MAX_LAUNCHES:
                log(f"FAILED: session lost {launches} times; giving up")
                return
            launches += 1
            log(f"session not found (reclaimed or not started); launch {launches}")
            if not launch():
                log("launch failed; will retry next cycle")
        time.sleep(CHECK_SECONDS)
    log("DEADLINE reached; stopping the session")
    colab("stop", "-s", M["session"], timeout=300)


if __name__ == "__main__":
    main()
