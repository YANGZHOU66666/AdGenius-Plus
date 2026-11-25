# Runs inside the Colab kernel and blocks until the formal run is finished, so the session never looks idle
# (an idle kernel is reclaimed after ~20 minutes even while a detached process uses the GPU).
# Reads run_config.json (written by the watchdog). The Tinker key arrives as a one-time file and is deleted at once;
# the rclone config stays in /content/run/.rclone.conf (0600) for the whole run and goes away with the VM.
import importlib.metadata as md
import json, os, subprocess, sys, time
root = "/content/run"
cfg = json.load(open(f"{root}/run_config.json"))
method = cfg["method"]
# A restart after a crash may leave an old trainer, generator or proxy holding the GPU or the port.
for name in ("run_formal.py", "opd_train.py", "sft_train.py", "gen_outputs.py", "teacher_proxy.py"):
    subprocess.run(["pkill", "-f", name])
for f in (f"{root}/DONE_{method}", f"{root}/CELL_EXIT_{method}"):
    if os.path.exists(f):
        os.remove(f)
env = dict(os.environ, RCLONE_CONFIG=f"{root}/.rclone.conf")
key_path = f"{root}/.tinker_key"
if os.path.exists(key_path):
    env["TINKER_API_KEY"] = open(key_path).read().strip()
    os.remove(key_path)
os.chmod(f"{root}/.rclone.conf", 0o600)
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
disk = subprocess.run(["df", "-h", "/content"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
versions = {}
for pkg in ("torch", "vllm", "trl", "transformers", "accelerate", "datasets", "tinker", "flash-linear-attention"):
    try:
        versions[pkg] = md.version(pkg)
    except md.PackageNotFoundError:
        versions[pkg] = None
json.dump({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": gpu, "disk": disk,
           "python": sys.version, "packages": versions, "config": cfg},
          open(f"{root}/env_{method}_{int(time.time())}.json", "w"), indent=1)
with open(f"{root}/runner_{method}.log", "a") as log:
    code = subprocess.run([sys.executable, "run_formal.py", *cfg["args"]], cwd=root, env=env,
                          stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
open(f"{root}/CELL_EXIT_{method}", "w").write(str(code))
print("run_formal exit", code, flush=True)
