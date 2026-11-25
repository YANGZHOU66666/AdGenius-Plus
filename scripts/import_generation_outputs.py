"""Convert raw generation outputs (product_id, text, finish_reason) into GenerationResult v1.1 records.

For model outputs produced outside scripts/generate_candidates.py, e.g. trained-student checkpoints
generated on a GPU VM. finish_reason "stop" with text -> success; "length" with text -> truncated
(error output_too_long, text kept); no text -> failed (empty_output). Candidate IDs are derived from the
run ID and product, so re-importing the same run gives the same IDs. Offline; refuses to overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]


def convert(rows: list[dict], run_id: str, product_ids: set[str]) -> list[dict]:
    seen, out = set(), []
    for row in rows:
        product = row["product_id"]
        if product not in product_ids:
            raise ValueError(f"Output refers to a product not in the ProductItem file: {product}")
        if product in seen:
            raise ValueError(f"Two outputs for one product: {product}")
        seen.add(product)
        text = row.get("text")
        text = text if isinstance(text, str) and text.strip() else None
        if text is None:
            result = {"status": "failed", "text": None, "error": "empty_output"}
        elif row.get("finish_reason") == "length":
            result = {"status": "truncated", "text": text, "error": "output_too_long"}
        elif row.get("finish_reason") == "stop":
            result = {"status": "success", "text": text, "error": None}
        else:
            raise ValueError(f"Unexpected finish_reason {row.get('finish_reason')!r} for {product}")
        candidate = "cand_" + hashlib.sha256(json.dumps([run_id, product, 0]).encode()).hexdigest()[:24]
        out.append({"candidate_id": candidate, "product_id": product, "run_id": run_id, "sample_index": 0,
                    "generation_result": result, "trace_id": None})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL with product_id, text, finish_reason")
    parser.add_argument("--output", type=Path, required=True, help="GenerationResult JSONL to create")
    parser.add_argument("--system", required=True, help="System alias, e.g. sft_4k; becomes part of the run ID")
    parser.add_argument("--run-id", help="Default gen_<system>_<input file UTC mtime>")
    parser.add_argument("--products", type=Path, required=True, help="ProductItem file the outputs were generated for")
    parser.add_argument("--note", default="", help="Free-text provenance stored in the manifest")
    args = parser.parse_args(argv)
    if re.fullmatch(r"[A-Za-z0-9_]+", args.system) is None:
        parser.error("--system may contain only letters, digits and underscores")
    stamp = datetime.fromtimestamp(args.input.stat().st_mtime, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"gen_{args.system}_{stamp}"
    with args.products.open(encoding="utf-8") as handle:
        product_ids = {json.loads(line)["product_id"] for line in handle if line.strip()}
    with args.input.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    records = convert(rows, run_id, product_ids)
    validator = jsonschema.Draft202012Validator(json.loads(
        (ROOT / "metadata/schemas/generation_result.schema.json").read_text(encoding="utf-8")))
    for record in records:
        validator.validate(record)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    counts = {status: sum(r["generation_result"]["status"] == status for r in records)
              for status in ("success", "truncated", "failed")}
    manifest = {"run_id": run_id, "system": args.system, "imported_at": datetime.now(timezone.utc).isoformat(),
                "input": {"path": str(args.input.resolve()), "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()},
                "products": {"path": str(args.products), "sha256": hashlib.sha256(args.products.read_bytes()).hexdigest(),
                             "count": len(product_ids)},
                "records": len(records), "status_counts": counts,
                "missing_products": len(product_ids) - len(records), "note": args.note,
                "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_name(args.output.stem + ".manifest.json")
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(text)
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "run_id": run_id, **counts,
                      "missing_products": manifest["missing_products"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
