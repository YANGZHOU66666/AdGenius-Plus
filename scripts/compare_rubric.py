"""Compare rubric judge runs with the human factual_support labels.

Writes results/rubric/comparison/metrics.json and manifest.json. Reads call journals from
artifacts/rubric/<run_id>/calls.jsonl when present (token counts only). No API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from adgenius.eval.rubric_metrics import summarize

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "eval" / "gold"
RUNS = ["deepseek41_rubric_v1", "claude55_rubric_v1", "gpt6_luna_rubric_v1", "qwen37_rubric_v1"]


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=RUNS)
    parser.add_argument("--human", type=Path, default=GOLD / "rubric_annotations_zhou.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "rubric" / "comparison")
    args = parser.parse_args()

    sample_path = GOLD / "rubric_sample.jsonl"
    generator = {r["candidate_id"]: r["generator"] for r in _rows(sample_path)}
    human = {}
    for row in _rows(args.human):
        criterion = next(c for c in row["rubric_result"]["criteria"] if c["criterion_id"] == "factual_support")
        human[row["candidate_id"]] = criterion["value"]
    order = [r["candidate_id"] for r in _rows(sample_path)]
    human = {cid: human[cid] for cid in order}

    metrics, results_files = {}, []
    for run_id in args.runs:
        path = ROOT / "results" / "rubric" / run_id / "results.jsonl"
        calls_path = ROOT / "artifacts" / "rubric" / run_id / "calls.jsonl"
        calls = _rows(calls_path) if calls_path.exists() else None
        metrics[run_id] = summarize(_rows(path), human, generator, calls)
        results_files.append({"path": str(path.relative_to(ROOT)), "sha256": _sha(path)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "analysis_kind": "rubric_judge_human_agreement",
        "computed_on": date.today().isoformat(),
        "new_api_calls": 0,
        "positive_class": "factual_support = false (conflict)",
        "bootstrap": {"unit": "item", "iterations": 10000, "seed": 20261124},
        "human_annotations": {"path": str(args.human.relative_to(ROOT)), "sha256": _sha(args.human)},
        "rubric_sample": {"path": str(sample_path.relative_to(ROOT)), "sha256": _sha(sample_path)},
        "results": results_files,
        "metrics_sha256": _sha(metrics_path),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
    for run_id, m in metrics.items():
        print(f"{run_id:<22} acc {m['accuracy']:.3f}  kappa {m['kappa']:.3f}  "
              f"caught {m['caught']}/{m['human_false']}  extra {m['extra_false']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
