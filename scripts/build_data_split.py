#!/usr/bin/env python3
"""Build a reproducible, leakage-safe split for Office Products metadata.

The split is made at a conservative content-group level. The source dataset
does not expose a reliable product-family/variant relation, so records with
the same generation-relevant content are kept together; all other products
are treated as independent by parent_asin.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SEED = 20261028
GOLD_SIZE = 100
TEST_SIZE = 1_000
SOURCE_FIELDS = (
    "main_category",
    "title",
    "average_rating",
    "rating_number",
    "features",
    "description",
    "price",
    "store",
    "categories",
    "details",
)


@dataclass(frozen=True)
class RecordInfo:
    product_id: str
    group_key: str
    source_line: int


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_group_key(record: dict[str, Any]) -> str:
    """Hash only fields used as model input; parent_asin is deliberately excluded."""
    content = {field: record.get(field) for field in SOURCE_FIELDS}
    return hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()


def read_index(source: Path) -> tuple[list[RecordInfo], dict[str, int], Counter[str], list[str]]:
    records: list[RecordInfo] = []
    group_sizes: dict[str, int] = defaultdict(int)
    missing: Counter[str] = Counter()
    errors: list[str] = []
    seen: set[str] = set()

    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {source_line}: invalid JSON ({exc})")
                continue
            if not isinstance(record, dict):
                errors.append(f"line {source_line}: record is not an object")
                continue

            product_id = record.get("parent_asin")
            if not isinstance(product_id, str) or not product_id.strip():
                errors.append(f"line {source_line}: missing parent_asin")
                continue
            if product_id in seen:
                errors.append(f"line {source_line}: duplicate parent_asin={product_id}")
                continue
            seen.add(product_id)

            for field in ("title", "features", "description", "details", "categories"):
                value = record.get(field)
                if value is None or value == "" or value == [] or value == {}:
                    missing[field] += 1

            group_key = content_group_key(record)
            records.append(RecordInfo(product_id, group_key, source_line))
            group_sizes[group_key] += 1

    return records, dict(group_sizes), missing, errors


def ordered_groups(group_sizes: dict[str, int], seed: int) -> list[str]:
    def sort_key(group_key: str) -> str:
        return hashlib.sha256(f"{seed}:{group_key}".encode("ascii")).hexdigest()

    return sorted(group_sizes, key=sort_key)


def choose_groups(
    groups: Iterable[str], group_sizes: dict[str, int], target: int, excluded: set[str]
) -> set[str]:
    chosen: set[str] = set()
    remaining = target
    for group_key in groups:
        if group_key in excluded:
            continue
        size = group_sizes[group_key]
        if size > remaining:
            continue
        chosen.add(group_key)
        remaining -= size
        if remaining == 0:
            return chosen
    raise RuntimeError(
        f"Could not select exactly {target} records without splitting a content group; "
        f"{remaining} records remain. Use a different seed or inspect group sizes."
    )


def jsonl_write(handle: Any, record: dict[str, Any]) -> None:
    handle.write(canonical_json(record))
    handle.write("\n")


def build_split(source: Path, repo_root: Path, seed: int) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    split_dir = repo_root / "data" / "splits" / "v1"
    processed_dir = repo_root / "data" / "processed" / "v1"
    split_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    source_sha256 = sha256_file(source)
    records, group_sizes, missing, errors = read_index(source)
    if errors:
        raise RuntimeError("Source validation failed:\n" + "\n".join(errors[:20]))

    groups = ordered_groups(group_sizes, seed)
    gold_groups = choose_groups(groups, group_sizes, GOLD_SIZE, set())
    test_groups = choose_groups(groups, group_sizes, TEST_SIZE, gold_groups)

    split_by_group = {
        **{group: "gold" for group in gold_groups},
        **{group: "test" for group in test_groups},
    }
    split_counts: Counter[str] = Counter()
    id_sets: dict[str, set[str]] = {"gold": set(), "test": set(), "train_pool": set()}
    for record in records:
        split = split_by_group.get(record.group_key, "train_pool")
        split_counts[split] += 1
        id_sets[split].add(record.product_id)

    expected_counts = {"gold": GOLD_SIZE, "test": TEST_SIZE, "train_pool": len(records) - GOLD_SIZE - TEST_SIZE}
    if dict(split_counts) != expected_counts:
        raise RuntimeError(f"Unexpected split sizes: got {dict(split_counts)}, expected {expected_counts}")
    if sum(len(values) for values in id_sets.values()) != len(records):
        raise RuntimeError("Product IDs overlap across splits")

    split_by_product = {
        record.product_id: (split_by_group.get(record.group_key, "train_pool"), record)
        for record in records
    }

    split_paths = {name: split_dir / f"{name}.jsonl" for name in ("gold", "test", "train_pool")}
    processed_paths = {
        name: processed_dir / f"{name}_products.jsonl" for name in ("gold", "test")
    }
    split_handles = {name: path.open("w", encoding="utf-8") for name, path in split_paths.items()}
    processed_handles = {name: path.open("w", encoding="utf-8") for name, path in processed_paths.items()}
    raw_written: Counter[str] = Counter()
    try:
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for source_line, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                product_id = record["parent_asin"]
                split, info = split_by_product[product_id]
                manifest_record = {
                    "product_id": product_id,
                    "group_id": f"content_{info.group_key[:16]}",
                    "split": split,
                    "source_line": source_line,
                    "seed": seed,
                    "source_dataset": "Amazon Reviews 2023 metadata / Office Products",
                    "source_sha256": source_sha256,
                }
                jsonl_write(split_handles[split], manifest_record)
                if split in processed_handles:
                    jsonl_write(processed_handles[split], record)
                    raw_written[split] += 1
    finally:
        for handle in [*split_handles.values(), *processed_handles.values()]:
            handle.close()

    if dict(raw_written) != {"gold": GOLD_SIZE, "test": TEST_SIZE}:
        raise RuntimeError(f"Selected raw product output is incomplete: {dict(raw_written)}")

    output_sha256 = {
        path.relative_to(repo_root).as_posix(): sha256_file(path)
        for path in [*split_paths.values(), *processed_paths.values()]
    }
    singleton_groups = sum(size == 1 for size in group_sizes.values())
    manifest = {
        "version": "v1",
        "seed": seed,
        "source": {
            "path": source.as_posix(),
            "dataset": "Amazon Reviews 2023 metadata",
            "category": "Office Products",
            "sha256": source_sha256,
            "records": len(records),
        },
        "split_counts": dict(split_counts),
        "grouping": {
            "method": "sha256 of canonical generation-relevant product fields, excluding parent_asin",
            "groups": len(group_sizes),
            "singleton_groups": singleton_groups,
            "multi_record_groups": len(group_sizes) - singleton_groups,
            "group_id_prefix": "content_",
        },
        "missing_fields": dict(missing),
        "outputs_sha256": output_sha256,
        "notes": [
            "gold and test are frozen evaluation-only products; neither may enter training",
            "train_pool is the complete remainder and is a candidate pool, not a required training size",
            "no reliable variant/family relation exists in this source; exact generation-content duplicates are kept together",
            "processed gold/test files contain original product records only; no generated copy or human labels",
        ],
    }
    (split_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_source = repo_root.parent / "data" / "raw" / "amazon_reviews_2023" / "meta_categories" / "meta_Office_Products.jsonl.gz"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_split(args.source.resolve(), args.repo_root.resolve(), args.seed)
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"split_counts": manifest["split_counts"], "manifest": "data/splits/v1/manifest.json"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
