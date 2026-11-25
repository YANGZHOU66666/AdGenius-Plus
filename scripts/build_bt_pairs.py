"""Build the CandidatePair file for a Bradley–Terry design (configs/eval/bt/*.toml).

Checks that the comparisons connect every system and that each compared system has a usable ad per
product, then writes the pairs (interleaved by product) and a manifest next to them. Offline; refuses to
overwrite. Judge the pairs with scripts/evaluate_pairwise.py --pairs <file> --no-human-annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from adgenius.eval.bradley_terry import build_pairs, load_design, relative, sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Default: the design's pairs path")
    parser.add_argument("--check", action="store_true", help="Validate and print the plan without writing")
    args = parser.parse_args(argv)
    try:
        design = load_design(args.design.resolve())
        rows, report = build_pairs(design)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    output = (args.output or design["pairs"]).resolve()
    report = {"design_file": {"path": relative(design["path"]), "sha256": sha256(design["path"])}, **report,
              "output": relative(output)}
    if args.check:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    report["output_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest_path = output.with_name(output.stem + ".manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(text)
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(rows)} pairs ({2 * len(rows)} judge calls) to {relative(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
