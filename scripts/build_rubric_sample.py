"""Freeze the gold subset used to check rubric judges against human factual_support labels.

The sample has 130 candidates: every candidate of the full-census model (qwen_local by default),
plus a fixed number from each other model, drawn from distinct products. Items are shuffled, so
item order does not reveal the generator. This is an offline transformation: no API requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "eval" / "gold"
FULL_MODEL = "qwen_local"
OTHER_MODELS = ("gemma_31b", "glm53_flash", "longcat")


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records found in {path}")
    return records, hashlib.sha256(raw).hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)


def build_sample(product_ids: list[str], candidates: dict[str, dict[str, dict[str, Any]]],
                 full_model: str, other_models: list[str], per_other: int, seed: int
                 ) -> list[dict[str, Any]]:
    """Return shuffled sample rows. `candidates` maps model -> product_id -> GenerationResult."""
    products = sorted(product_ids)
    for model in [full_model, *other_models]:
        if set(candidates.get(model, {})) != set(products):
            raise ValueError(f"{model} must have exactly one candidate for every product")
    if per_other * len(other_models) > len(products):
        raise ValueError("Not enough products for distinct non-census samples")
    rng = random.Random(seed)
    picked = [(full_model, pid) for pid in products]
    drawn = rng.sample(products, per_other * len(other_models))
    for index, model in enumerate(other_models):
        picked += [(model, pid) for pid in drawn[index * per_other:(index + 1) * per_other]]
    rng.shuffle(picked)
    return [{"item_id": f"rubric_item_{n:03d}", "candidate_id": candidates[model][pid]["candidate_id"],
             "product_id": pid, "generator": model}
            for n, (model, pid) in enumerate(picked, start=1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--per-other", type=int, default=10)
    parser.add_argument("--output", type=Path, default=GOLD / "rubric_sample.jsonl")
    args = parser.parse_args()

    products_path = GOLD / "product_items.jsonl"
    products, products_sha = _read_jsonl(products_path)
    inputs, candidates = [], {}
    for model in [FULL_MODEL, *OTHER_MODELS]:
        path = GOLD / f"results_{model}.jsonl"
        records, digest = _read_jsonl(path)
        candidates[model] = {r["product_id"]: r for r in records}
        if len(candidates[model]) != len(records):
            raise ValueError(f"{path} has duplicate product_id values")
        inputs.append({"path": _relative(path), "sha256": digest, "record_count": len(records)})

    rows = build_sample([p["product_id"] for p in products], candidates, FULL_MODEL,
                        list(OTHER_MODELS), args.per_other, args.seed)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    args.output.write_text(text, encoding="utf-8")

    counts = {model: sum(row["generator"] == model for row in rows) for model in [FULL_MODEL, *OTHER_MODELS]}
    manifest = {
        "purpose": "human factual_support labels for rubric judge agreement",
        "inputs": inputs,
        "products": _relative(products_path),
        "products_sha256": products_sha,
        "seed": args.seed,
        "item_count": len(rows),
        "generator_counts": counts,
        "sampling": (f"all {FULL_MODEL} candidates; {args.per_other} per other model from "
                     f"{args.per_other * len(OTHER_MODELS)} distinct products; items shuffled"),
        "rng": "random.Random(seed): sample non-census products, then shuffle all items",
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    manifest_path = args.output.with_name(args.output.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} items to {_relative(args.output)}: {counts}")


if __name__ == "__main__":
    main()
