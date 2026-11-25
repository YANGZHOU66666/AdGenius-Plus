"""Win rates and Bradley–Terry scores from pairwise judge results for a design. No API calls.

Writes metrics.json, report.md and manifest.json to --output-dir (default results/bt/<design name>/).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

from adgenius.eval import bradley_terry as bt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True,
                        help="PairwiseResult files (results/pairwise/<run_id>/results.jsonl)")
    parser.add_argument("--pairs", type=Path,
                        help="Default: the design's pairs path, checked against the design. An override is not checked")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--pseudo", type=float, default=0.5,
                        help="Virtual tied pairs added to each comparison (default 0.5)")
    args = parser.parse_args(argv)
    try:
        design = bt.load_design(args.design.resolve())
        pairs_path = (args.pairs or design["pairs"]).resolve()
        pairs = bt.read_jsonl(pairs_path)
        results = [row for path in args.results for row in bt.read_jsonl(path)]
        planned = bt.plan(design)
        if args.pairs is None and {p["pair_id"] for p in pairs} != {p["pair_id"] for p in planned["pairs"]}:
            raise ValueError("The pairs file does not match the design and system files; rebuild it")
        rows, counts = bt.outcomes(design, pairs, results)
        if not rows:
            raise ValueError("No pair has two successful judged orders")
        counts["forfeits"] = len(planned["forfeits"])
        metrics = bt.summarize(design, rows, args.iterations, args.seed, args.pseudo, planned["forfeits"])
        metrics["pair_counts"] = counts
        bt.validate_metrics(metrics)
    except (OSError, ValueError, KeyError, jsonschema.ValidationError) as error:
        parser.error(str(error))
    output = (args.output_dir or bt.REPO_ROOT / "results" / "bt" / design["name"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(bt.render_report(metrics, counts), encoding="utf-8")
    code = [Path(bt.__file__), Path(__file__), bt.SCHEMA_DIR / "bt_metrics.schema.json"]
    manifest = {"computed_at": datetime.now(timezone.utc).isoformat(), "new_api_calls": 0,
                "design": {"path": bt.relative(design["path"]), "sha256": bt.sha256(design["path"])},
                "pairs": {"path": bt.relative(pairs_path), "sha256": bt.sha256(pairs_path)},
                "results": [{"path": bt.relative(p), "sha256": bt.sha256(p)} for p in args.results],
                "systems": {n: {"path": bt.relative(p), "sha256": bt.sha256(p)} for n, p in design["systems"].items()},
                "code_sha256": {bt.relative(p): bt.sha256(p) for p in code},
                "metrics_sha256": bt.sha256(output / "metrics.json")}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in metrics["systems"]:
        print(f"{row['system']:<16} {row['score']:+.3f}  {row['score_ci95']}")
    print(f"Wrote {bt.relative(output)}/ (judged {counts['scored']}, forfeits {counts['forfeits']}, "
          f"incomplete {counts.get('incomplete', 0)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
