"""Google Drive checkpoint sync for the formal runs (used on the Colab VM; rclone remote "gdrive").

- DriveSync: a TrainerCallback that, after every checkpoint save, starts a background `rclone copy` of the full
  checkpoint to <drive>/full/step_NNNN and writes an UPLOAD_OK marker only when the copy succeeded. It also writes
  progress.json (step, loss, lr, elapsed) after every log, and can stop training at a given step (smoke tests).
- latest_checkpoint(): the newest complete checkpoint, local or on Drive (downloaded if only on Drive).
- wait_for_uploads(): block until no rclone process is running.
The rclone config path comes from RCLONE_CONFIG; the token is never printed.
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path

from transformers import TrainerCallback

STEP_DIR = re.compile(r"^step_(\d+)/?$")


# rclone's shared client_id hits Google's per-minute quota (403 rateLimitExceeded): fewer, larger requests and
# patient retries. Transfers are best-effort; training never waits for Drive.
RETRY = ["--retries", "10", "--low-level-retries", "30", "--tpslimit", "4", "--drive-chunk-size", "512M"]


def rclone(*args, check=False, capture=True, timeout=None):
    extra = RETRY if args and args[0] in ("copy", "lsf") else []
    try:
        r = subprocess.run(["rclone", *args, *extra], capture_output=capture, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if check:
            raise RuntimeError(f"rclone {args[0]} timed out")
        return subprocess.CompletedProcess(args, 124, "", "timeout")
    if check and r.returncode != 0:
        raise RuntimeError(f"rclone {args[0]} failed: {r.stderr[-500:]}")
    return r


def wait_for_uploads(poll=10, max_seconds=None):
    """Wait for background uploads; return True if none are left, False if max_seconds passed first."""
    start = time.time()
    while subprocess.run(["pgrep", "-f", "rclone copy"], capture_output=True).returncode == 0:
        if max_seconds is not None and time.time() - start > max_seconds:
            return False
        time.sleep(poll)
    return True


def drive_complete_steps(drive_root):
    r = rclone("lsf", f"{drive_root}/full/", "--dirs-only")
    steps = []
    for line in (r.stdout or "").splitlines():
        m = STEP_DIR.match(line.strip())
        if m and "UPLOAD_OK" in (rclone("lsf", f"{drive_root}/full/{line.strip()}").stdout or ""):
            steps.append(int(m.group(1)))
    return sorted(steps)


def local_complete_steps(trainer_dir):
    out = []
    for p in Path(trainer_dir).glob("checkpoint-*"):
        if (p / "trainer_state.json").exists() and any(p.glob("optimizer.*")):
            out.append(int(p.name.split("-")[1]))
    return sorted(out)


def latest_checkpoint(trainer_dir, drive_root, log=print):
    """Path of the newest complete checkpoint (downloading it from Drive if needed), or None."""
    local = local_complete_steps(trainer_dir)
    remote = drive_complete_steps(drive_root) if drive_root else []
    best = max(local + remote, default=None)
    if best is None:
        return None
    path = Path(trainer_dir) / f"checkpoint-{best}"
    if best not in local:
        log(f"downloading checkpoint step {best} from Drive")
        path.mkdir(parents=True, exist_ok=True)
        rclone("copy", f"{drive_root}/full/step_{best:04d}", str(path), "--exclude", "UPLOAD_OK", check=True)
    return path


class DriveSync(TrainerCallback):
    def __init__(self, drive_root, progress_path, log_path, stop_after=None):
        self.drive_root, self.progress_path, self.log_path, self.stop_after = drive_root, Path(progress_path), log_path, stop_after
        self.start = time.time()

    def on_save(self, args, state, control, **kwargs):
        if not self.drive_root:
            return
        step = state.global_step
        src = Path(args.output_dir) / f"checkpoint-{step}"
        dest = f"{self.drive_root}/full/step_{step:04d}"
        cmd = (f"rclone copy '{src}' '{dest}' {' '.join(RETRY)} --transfers 2 -q && "
               f"echo ok | rclone rcat '{dest}/UPLOAD_OK' && echo \"$(date -u +%FT%TZ) uploaded step {step}\" "
               f"|| echo \"$(date -u +%FT%TZ) UPLOAD FAILED step {step}\"")
        with open(self.log_path, "a") as log:
            subprocess.Popen(["bash", "-c", cmd], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.progress_path.write_text(json.dumps({
                "step": state.global_step, "max_steps": state.max_steps, "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"), "elapsed_s": round(time.time() - self.start, 1),
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_after and state.global_step >= self.stop_after:
            os._exit(75)  # simulate a crash (smoke test of resume)
