"""Freeze one randomly selected pair per product from complete candidate sets.

This is an offline transformation: no API requests or environment keys are used.
Every requested model must have exactly one successful sample_index=0 candidate
for every product. Candidate order is canonical storage order, not A/B order.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_MODELS = ("qwen_local", "gemma_31b", "glm53_flash", "longcat")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    records = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records, hashlib.sha256(raw).hexdigest()


def _identifier(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {field}")
    return value


def _product_ids(path: Path) -> tuple[set[str], str]:
    records, digest = _read_jsonl(path)
    identifiers = [_identifier(record, "product_id") for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("ProductItem input contains duplicate product_id values")
    return set(identifiers), digest


def build_pairs(records: list[dict[str, Any]], models: list[str], seed: int,
                expected_products: set[str] | None = None
                ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate complete candidate sets before sampling; input order is irrelevant."""
    if len(models) < 2 or len(set(models)) != len(models):
        raise ValueError("Provide at least two distinct model aliases")
    if any(not isinstance(model, str) or not model.strip() for model in models):
        raise ValueError("Model aliases must be nonempty strings")
    aliases = sorted(models)
    by_product: dict[str, dict[str, dict[str, str]]] = {}
    seen_ids: set[str] = set()
    for record in records:
        candidate_id = _identifier(record, "candidate_id")
        product_id = _identifier(record, "product_id")
        run_id = _identifier(record, "run_id")
        if candidate_id in seen_ids:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)
        matches = [model for model in aliases
                   if run_id.startswith(f"gen_{model}_")
                   and len(run_id) > len(f"gen_{model}_")]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely match run_id to requested model: {run_id}")
        model = matches[0]
        if type(record.get("sample_index")) is not int or record["sample_index"] != 0:
            raise ValueError(f"Expected sample_index=0 for {candidate_id}")
        result = record.get("generation_result")
        if not isinstance(result, dict) or result.get("status") != "success":
            raise ValueError(f"Candidate is not successful: {candidate_id}")
        text = result.get("text")
        if not isinstance(text, str) or not text.strip() or result.get("error") is not None:
            raise ValueError(f"Invalid successful candidate: {candidate_id}")
        candidates = by_product.setdefault(product_id, {})
        if model in candidates:
            raise ValueError(f"Duplicate candidate for {product_id}/{model}")
        candidates[model] = {"candidate_id": candidate_id, "text": text}

    if not by_product:
        raise ValueError("No candidates found")
    if expected_products is not None and set(by_product) != expected_products:
        missing = sorted(expected_products - set(by_product))
        extra = sorted(set(by_product) - expected_products)
        raise ValueError(f"Product coverage mismatch: missing={missing}, extra={extra}")
    for product_id, candidates in by_product.items():
        missing_models = sorted(set(aliases) - set(candidates))
        if missing_models:
            raise ValueError(f"Missing candidates for {product_id}: {missing_models}")

    combinations = {" + ".join(combination): 0
                    for combination in itertools.combinations(aliases, 2)}
    pairs = []
    for product_id in sorted(by_product):
        # Independent per-product RNG avoids changing other products when the
        # input is reordered or a different product is added to a later run.
        random_seed = int.from_bytes(hashlib.sha256(
            _canonical([seed, product_id])).digest(), "big")
        chosen = sorted(random.Random(random_seed).sample(aliases, 2))
        candidates = sorted((by_product[product_id][model] for model in chosen),
                            key=lambda candidate: candidate["candidate_id"])
        pair_key = {"seed": seed, "product_id": product_id, "candidates": candidates}
        pair_id = "pair_" + hashlib.sha256(_canonical(pair_key)).hexdigest()[:24]
        pairs.append({"pair_id": pair_id, "product_id": product_id,
                      "candidates": candidates})
        combinations[" + ".join(chosen)] += 1
    return pairs, combinations


def _write_outputs(output: Path, pairs: list[dict[str, Any]],
                   manifest_output: Path, manifest: dict[str, Any]) -> None:
    """Reserve both targets without overwrite, and clean up partial writes."""
    created: list[Path] = []
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as pair_file:
            created.append(output)
            with manifest_output.open("x", encoding="utf-8") as manifest_file:
                created.append(manifest_output)
                for pair in pairs:
                    pair_file.write(json.dumps(pair, ensure_ascii=False) + "\n")
                manifest_file.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, nargs="+",
                        help="One or more GenerationResult JSONL files covering all requested models")
    parser.add_argument("--output", required=True, type=Path,
                        help="CandidatePair JSONL; must not already exist")
    parser.add_argument("--products", type=Path,
                        help="Optional ProductItem JSONL for exact product coverage validation")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-output", type=Path,
                        help="Defaults to OUTPUT with suffix .manifest.json; no overwrite")
    args = parser.parse_args(argv)
    manifest_output = args.manifest_output or args.output.with_suffix(".manifest.json")
    output_paths = [args.output.resolve(), manifest_output.resolve()]
    input_paths = [path.resolve() for path in args.input]
    if len(set(input_paths)) != len(input_paths):
        parser.error("--input must not contain duplicate paths")
    if args.products:
        input_paths.append(args.products.resolve())
    if len(set(output_paths)) != 2 or set(output_paths) & set(input_paths):
        parser.error("Input, output, and manifest paths must be distinct")
    for path in (args.output, manifest_output):
        if path.exists():
            parser.error(f"Output already exists: {path}")
    try:
        records = []
        inputs = []
        for path in args.input:
            rows, digest = _read_jsonl(path)
            records.extend(rows)
            inputs.append({"path": str(path.resolve()), "sha256": digest,
                           "record_count": len(rows)})
        expected_products, products_digest = (_product_ids(args.products) if args.products
                                              else (None, None))
        pairs, combinations = build_pairs(records, args.models, args.seed, expected_products)
        manifest = {
            "inputs": inputs,
            "products": str(args.products.resolve()) if args.products else None,
            "products_sha256": products_digest,
            "seed": args.seed,
            "models": sorted(args.models),
            "sample_index": 0,
            "product_count": len(pairs),
            "pair_count": len(pairs),
            "source_combination_counts": combinations,
            "sampling": "uniformly sample two distinct model aliases per product",
            "rng": "random.Random seeded by SHA-256 of canonical [seed, product_id]",
            "candidate_order": "candidate_id ascending; not an A/B presentation mapping",
            "output_sha256": hashlib.sha256("".join(
                json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs
            ).encode("utf-8")).hexdigest(),
        }
        if len(inputs) == 1:
            # Preserve metadata keys produced by the original single-file CLI.
            manifest.update(input=inputs[0]["path"], input_sha256=inputs[0]["sha256"])
        _write_outputs(args.output, pairs, manifest_output, manifest)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote {len(pairs)} candidate pairs to {args.output}")
    print(f"Wrote manifest to {manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
