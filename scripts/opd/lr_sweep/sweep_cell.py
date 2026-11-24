# Runs inside the Colab kernel (blocking) so the session never looks idle.
# Reads pending.json (written by the watchdog) for the runs still missing, moves the one-time key file into the
# child environment, and runs the beta groups one after the other on this VM.
import json, os, subprocess, sys
root = "/content/sweep"
# A restart after a kernel crash may leave the old proxy or trainer holding the port and GPU.
for name in ("vm_runner.py", "opd_train.py", "dev_generate.py", "teacher_proxy.py"):
    subprocess.run(["pkill", "-f", name])
if os.path.exists(f"{root}/ALL_DONE"):
    os.remove(f"{root}/ALL_DONE")
pending = json.load(open(f"{root}/pending.json"))
# Record the software and hardware this cell ran on (lost when the VM goes away).
import importlib.metadata as md, time
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
versions = {}
for pkg in ("torch", "vllm", "trl", "transformers", "accelerate", "datasets", "tinker", "flash-linear-attention"):
    try:
        versions[pkg] = md.version(pkg)
    except md.PackageNotFoundError:
        versions[pkg] = None
json.dump({"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": gpu, "python": sys.version,
           "packages": versions, "pending": pending}, open(f"{root}/env_{int(time.time())}.json", "w"), indent=1)
key_path = f"{root}/.tinker_key"
key = open(key_path).read().strip()
os.remove(key_path)
env = dict(os.environ, TINKER_API_KEY=key)
del key
for i, group in enumerate(pending["groups"]):
    args = ["--beta", group["beta"], "--lrs", *group["lrs"], "--port", "18001" if float(group["beta"]) == 1.0 else "18002",
            "--status-file", f"status_beta{group['beta']}.json"]
    if group.get("base_dev"):
        args.append("--base-dev")
    with open(f"{root}/runner_{i}.log", "w") as log:
        code = subprocess.run([sys.executable, "vm_runner.py", *args], cwd=root, env=env,
                              stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
    print("group", group["beta"], "exit", code, flush=True)
open(f"{root}/ALL_DONE", "w").write("done")
print("ALL_DONE", flush=True)
