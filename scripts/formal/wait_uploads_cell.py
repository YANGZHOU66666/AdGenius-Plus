# Runs in a Colab cell after DONE: keeps the kernel busy until every local checkpoint and bf16 export is complete on
# Drive (or a time limit passes), re-uploading any that failed. Reads wait_config.json: {"name", "method", "drive_root",
# "max_minutes"}.
import json, os, subprocess, sys, time
root = "/content/run"
sys.path.insert(0, root)
os.environ["RCLONE_CONFIG"] = f"{root}/.rclone.conf"
import drive_sync  # noqa: E402
cfg = json.load(open(f"{root}/wait_config.json"))
run = f"{root}/runs/{cfg['name']}"
trainer = run if cfg["method"] == "opd" else f"{run}/trainer"
drive = cfg["drive_root"]
start = time.time()
while time.time() - start < cfg["max_minutes"] * 60:
    drive_sync.wait_for_uploads(max_seconds=600)
    on_drive = set(drive_sync.drive_complete_steps(drive))
    pending = [s for s in drive_sync.local_complete_steps(trainer) if s not in on_drive]
    if not pending:
        break
    step = pending[0]
    dest = f"{drive}/full/step_{step:04d}"
    r = drive_sync.rclone("copy", f"{trainer}/checkpoint-{step}", dest)
    if r.returncode == 0:
        subprocess.run(["bash", "-c", f"echo ok | rclone rcat '{dest}/UPLOAD_OK'"], env=os.environ)
    print(f"re-upload step {step}: {'ok' if r.returncode == 0 else 'failed'}", flush=True)
for scale in sorted(os.listdir(f"{run}/weights")) if os.path.isdir(f"{run}/weights") else []:
    drive_sync.rclone("copy", f"{run}/weights/{scale}", f"{drive}/weights_bf16/{scale}")
steps = drive_sync.drive_complete_steps(drive)
json.dump({"drive_full_steps": steps, "minutes": round((time.time() - start) / 60, 1)},
          open(f"{root}/UPLOADS_{cfg['method']}.json", "w"))
print("UPLOADS_FINISHED", steps, flush=True)
