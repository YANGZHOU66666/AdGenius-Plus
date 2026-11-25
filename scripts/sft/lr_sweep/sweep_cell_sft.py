# Runs inside the Colab kernel and blocks until every pending SFT run is finished, so the session never looks idle
# (an idle kernel is reclaimed after ~20 minutes even while a detached process uses the GPU).
# Reads pending.json written by the watchdog: {"lrs": [...], "args": [...]}.
import importlib.metadata as md
import json, os, subprocess, sys, time
root = "/content/sweep"
# A restart after a kernel crash may leave an old trainer or generator holding the GPU.
for name in ("vm_runner_sft.py", "sft_train.py", "dev_generate.py"):
    subprocess.run(["pkill", "-f", name])
if os.path.exists(f"{root}/ALL_DONE"):
    os.remove(f"{root}/ALL_DONE")
pending = json.load(open(f"{root}/pending.json"))
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
versions = {}
for pkg in ("torch", "vllm", "trl", "transformers", "accelerate", "datasets", "flash-linear-attention"):
    try:
        versions[pkg] = md.version(pkg)
    except md.PackageNotFoundError:
        versions[pkg] = None
json.dump({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": gpu, "python": sys.version,
           "packages": versions, "pending": pending}, open(f"{root}/env_{int(time.time())}.json", "w"), indent=1)
with open(f"{root}/runner.log", "a") as log:
    code = subprocess.run([sys.executable, "vm_runner_sft.py", "--lrs", *pending["lrs"], *pending.get("args", [])],
                          cwd=root, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
print("runner exit", code, flush=True)
if not pending.get("smoke"):
    open(f"{root}/ALL_DONE", "w").write("done")
print("CELL_DONE", flush=True)
