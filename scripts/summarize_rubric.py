"""Recompute metrics.json of a finished --candidates rubric run from its committed results. No API calls.

The candidate and product files are taken from the run's manifest.json and checked against the hashes recorded
there. Call totals come from the call journal, which stays local; they are copied from the existing metrics.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from adgenius.eval import rubric
from adgenius.eval.rubric_metrics import generator_rates

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="results/rubric/<run_id>")
    parser.add_argument("--generator-names", nargs="+", metavar="NAME",
                        help="The names the run used, one per candidates file (default: derived from run IDs)")
    parser.add_argument("--output", type=Path, help="Default: <run-dir>/metrics.json")
    args = parser.parse_args(argv)

    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    if manifest["specification"]["input_mode"] != "candidates":
        parser.error("only --candidates runs have generator metrics")
    files = manifest["specification"]["input_files"]
    for entry in files.values():
        path = REPO_ROOT / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            parser.error(f"{entry['path']} differs from the file the run judged")
    candidates = [REPO_ROOT / files[key]["path"]
                  for key in sorted((k for k in files if k.startswith("candidates_")), key=lambda k: int(k.split("_")[1]))]
    items, extra = rubric.load_candidates(candidates, REPO_ROOT / files["product_items"]["path"],
                                          names=args.generator_names)
    if [it["candidate_id"] for it in items] != manifest["specification"]["candidate_ids"]:
        parser.error("candidates differ from the run's candidate list")
    results = {r["candidate_id"]: r for r in map(json.loads, (args.run_dir / "results.jsonl").read_text().splitlines())
               if r}
    previous_path = args.run_dir / "metrics.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if previous and set(previous["by_generator"]) != {it["generator"] for it in items}:
        parser.error(f"generator names differ from the run's: {sorted(previous['by_generator'])}")

    rates = generator_rates([results[it["candidate_id"]] for it in items], items)
    rates["calls"] = previous.get("calls", {"attempts": 0, "failed_attempts": 0, "input_tokens": 0, "output_tokens": 0})
    metrics = {"schema_version": rubric.METRICS_SCHEMA_VERSION, "run_id": manifest["run_id"],
               "input_mode": "candidates", **rates, **extra}
    rubric._validator("rubric_generator_metrics.schema.json").validate(metrics)
    output = args.output or previous_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(rubric._json_bytes(metrics))
    for name, row in metrics["by_generator"].items():
        low, high = row["false_rate_ci95"]
        print(f"{name:16} {row['false_rate']:.3f}  [{low}, {high}]")
    print(f"Wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
