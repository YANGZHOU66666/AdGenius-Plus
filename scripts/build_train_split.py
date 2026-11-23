#!/usr/bin/env python3
"""Freeze the training and dev ProductItems for the distillation experiments.

Reads the train-pool index (artifacts/cache/office_products/train_pool.jsonl)
and the raw Office Products metadata, and writes data/train/v1/:

- dev_products.jsonl    200 ProductItems for hyperparameter selection
- train_products.jsonl  8,000 ProductItems in a fixed order; the first 1k, 2k,
                        4k and 8k lines are the nested data-scale subsets
- split_index.jsonl     rank, group_id and source line of every selected item
- manifest.json         seed, filters, counts, hashes and per-scale category mix

Candidates are visited in a seeded hash order over content groups, so the
selection does not depend on file order and every prefix of the training list
is a uniform random sample of the eligible pool. Gold and test products are
already excluded from the train pool by data/splits/v1.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import jsonschema

from adgenius.shared.product import render_product_information

SEED = 20261123
DEV_SIZE = 200
TRAIN_SIZE = 8_000
SCALES = (1_000, 2_000, 4_000, 8_000)
MAX_RENDERED_CHARS = 6_000
VERSION = "v1"
PRODUCT_SCHEMA = Path("metadata") / "schemas" / "product_item.schema.json"


def code_version(repo_root: Path) -> dict[str, Any]:
    """Git commit of the code that produced an output, and whether code directories had local changes."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "--no-optional-locks", *args], cwd=repo_root, capture_output=True,
                              text=True, check=False).stdout.strip()
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", *CODE_PATHS)
    return {"git_commit": commit or None, "code_paths_dirty": bool(dirty) if commit else None}


CODE_PATHS = ("scripts", "src", "configs", "metadata")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _nonempty(value: Any) -> bool:
    return value not in (None, "", [], {})


def to_product_item(product_id: str, record: dict[str, Any]) -> dict[str, Any]:
    """ProductItem v1: project ID, source ASIN, and every other raw field."""
    info = {key: value for key, value in record.items() if key != "parent_asin"}
    return {"product_id": product_id, "source_id": record["parent_asin"], "product_info": info}


def rejection_reason(item: dict[str, Any]) -> str | None:
    info = item["product_info"]
    if not _nonempty(info.get("title")):
        return "missing_title"
    if not (_nonempty(info.get("features")) or _nonempty(info.get("description"))):
        return "missing_features_and_description"
    if len(render_product_information(item)) > MAX_RENDERED_CHARS:
        return "rendered_too_long"
    return None


def hash_order(entries: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """One entry per content group, ordered by sha256(seed:group_id)."""
    by_group: dict[str, dict[str, Any]] = {}
    for entry in entries:
        group = entry["group_id"]
        # Within a duplicate-content group keep the smallest project ID.
        if group not in by_group or entry["product_id"] < by_group[group]["product_id"]:
            by_group[group] = entry
    return sorted(by_group.values(),
                  key=lambda e: hashlib.sha256(f"{seed}:{e['group_id']}".encode("utf-8")).hexdigest())


def read_raw_lines(source: Path, wanted_lines: set[int]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number in wanted_lines:
                records[line_number] = json.loads(line)
                if len(records) == len(wanted_lines):
                    break
    missing = wanted_lines - records.keys()
    if missing:
        raise RuntimeError(f"{len(missing)} source lines not found in {source}")
    return records


class NotEnoughCandidates(RuntimeError):
    """The scanned prefix of the hash order held too few eligible products."""


def select(ordered: list[dict[str, Any]], raw: dict[int, dict[str, Any]],
           dev_size: int, train_size: int) -> tuple[list, list, Counter, int]:
    """Walk the hash order: the first dev_size eligible items are dev, the next train_size are train."""
    dev: list[tuple[dict, dict]] = []
    train: list[tuple[dict, dict]] = []
    rejected: Counter[str] = Counter()
    for visited, entry in enumerate(ordered, start=1):
        record = raw.get(entry["source_line"])
        if record is None:
            continue
        if record.get("parent_asin") != entry["source_id"]:
            raise RuntimeError(f"ASIN mismatch at source line {entry['source_line']}")
        item = to_product_item(entry["product_id"], record)
        reason = rejection_reason(item)
        if reason:
            rejected[reason] += 1
            continue
        (dev if len(dev) < dev_size else train).append((entry, item))
        if len(train) == train_size:
            return dev, train, rejected, visited
    raise NotEnoughCandidates(f"Only {len(dev)} dev and {len(train)} train items found; scan more candidates")


def top_category(item: dict[str, Any]) -> str:
    categories = item["product_info"].get("categories") or []
    return categories[1] if len(categories) > 1 else "(none)"


def category_mix(items: list[dict[str, Any]], top_n: int = 8) -> dict[str, float]:
    counts = Counter(top_category(item) for item in items)
    return {name: round(count / len(items), 4) for name, count in counts.most_common(top_n)}


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def build(pool_path: Path, source: Path, out_dir: Path, repo_root: Path, seed: int,
          dev_size: int = DEV_SIZE, train_size: int = TRAIN_SIZE, scan_factor: float = 1.6) -> dict[str, Any]:
    entries = read_jsonl(pool_path)
    if any(entry.get("split") != "train_pool" for entry in entries):
        raise RuntimeError("Train-pool index contains non-train_pool records")
    ordered = hash_order(entries, seed)
    scan = min(len(ordered), int((dev_size + train_size) * scan_factor))
    while True:
        candidates = ordered[:scan]
        raw = read_raw_lines(source, {entry["source_line"] for entry in candidates})
        try:
            dev, train, rejected, visited = select(candidates, raw, dev_size, train_size)
            break
        except NotEnoughCandidates:
            if scan == len(ordered):
                raise
            scan = min(len(ordered), scan * 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    dev_items = [item for _, item in dev]
    train_items = [item for _, item in train]
    paths = {name: out_dir / f"{name}.jsonl" for name in ("dev_products", "train_products", "split_index")}
    write_jsonl(paths["dev_products"], dev_items)
    write_jsonl(paths["train_products"], train_items)
    write_jsonl(paths["split_index"], [
        {"product_id": entry["product_id"], "source_id": entry["source_id"], "split": split,
         "rank": rank, "group_id": entry["group_id"], "source_line": entry["source_line"]}
        for split, chosen in (("dev", dev), ("train", train))
        for rank, (entry, _) in enumerate(chosen)
    ])

    schema_path = repo_root / PRODUCT_SCHEMA
    manifest = {
        "version": VERSION,
        "seed": seed,
        "schemas": {"products": {"path": PRODUCT_SCHEMA.as_posix(), "version": "ProductItem v1",
                                 "sha256": sha256_file(schema_path) if schema_path.exists() else None}},
        "source_id_field": "parent_asin",
        "id_mapping": ("product_id is reused from the train-pool index (train_product_NNNNNN, assigned by "
                       "scripts/build_data_split.py in source order); it never changes across rebuilds"),
        "code": {"script": "scripts/build_train_split.py", **code_version(repo_root)},
        "selection": {
            "order": "one product per content group, sorted by sha256(f'{seed}:{group_id}')",
            "dev": f"first {dev_size} eligible products in that order",
            "train": f"next {train_size} eligible products; line order is rank",
            "scales": {str(n): f"first {n} lines of train_products.jsonl" for n in SCALES if n <= train_size},
            "filters": {
                "title": "non-empty",
                "features_or_description": "at least one non-empty",
                "rendered_product_information_max_chars": MAX_RENDERED_CHARS,
                "duplicates": "one product per group_id",
            },
            "candidates_visited": visited,
            "rejected": dict(rejected),
        },
        "counts": {"dev": len(dev_items), "train": len(train_items)},
        "category_mix_top8": {
            "dev": category_mix(dev_items),
            **{f"train_{n}": category_mix(train_items[:n]) for n in SCALES if n <= train_size},
        },
        "inputs": {
            "train_pool_index": {"path": _rel(pool_path, repo_root), "sha256": sha256_file(pool_path)},
            "raw_source_sha256": sha256_file(source),
            "split_manifest": "data/splits/v1/manifest.json",
        },
        "outputs_sha256": {_rel(path, repo_root): sha256_file(path) for path in paths.values()},
        "notes": [
            "gold and test products are excluded upstream by data/splits/v1; dev and train are disjoint by product and group",
            "dev is for hyperparameter selection only and must never enter training",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    validate(out_dir, repo_root)
    return manifest


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def validate(out_dir: Path, repo_root: Path) -> None:
    """Re-read the outputs and check sizes, disjointness, eval exclusion and hashes."""
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["outputs_sha256"].items():
        path = repo_root / rel if not Path(rel).is_absolute() else Path(rel)
        if sha256_file(path) != expected:
            raise RuntimeError(f"Hash mismatch for {rel}")
    dev = read_jsonl(out_dir / "dev_products.jsonl")
    train = read_jsonl(out_dir / "train_products.jsonl")
    index = read_jsonl(out_dir / "split_index.jsonl")
    if (len(dev), len(train)) != (manifest["counts"]["dev"], manifest["counts"]["train"]):
        raise RuntimeError("Output sizes do not match the manifest")
    schema_path = repo_root / PRODUCT_SCHEMA
    if schema_path.exists():
        validator = jsonschema.Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))
        for item in dev + train:
            validator.validate(item)
    ids = [item["product_id"] for item in dev + train]
    sources = [item["source_id"] for item in dev + train]
    groups = [row["group_id"] for row in index]
    if len(set(ids)) != len(ids) or len(set(sources)) != len(sources) or len(set(groups)) != len(groups):
        raise RuntimeError("Duplicate product, source or group across dev and train")
    eval_sources: set[str] = set()
    for name in ("gold", "test"):
        split_file = repo_root / "data" / "splits" / "v1" / f"{name}.jsonl"
        if split_file.exists():
            eval_sources |= {row["product_id"] for row in read_jsonl(split_file)}
    if eval_sources & set(sources):
        raise RuntimeError("A gold or test product appears in dev or train")
    for split, items in (("dev", dev), ("train", train)):
        ranks = [row["rank"] for row in index if row["split"] == split]
        if ranks != list(range(len(items))):
            raise RuntimeError(f"Ranks for {split} are not 0..N-1 in order")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_source = (repo_root.parent / "data" / "raw" / "amazon_reviews_2023" / "meta_categories"
                      / "meta_Office_Products.jsonl.gz")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-pool", type=Path,
                        default=repo_root / "artifacts" / "cache" / "office_products" / "train_pool.jsonl")
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path, default=repo_root / "data" / "train" / VERSION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory")
    args = parser.parse_args()
    if (args.output_dir / "manifest.json").exists() and not args.force:
        print(f"ERROR: {args.output_dir} already holds a frozen split; pass --force to rebuild", file=sys.stderr)
        return 1
    try:
        manifest = build(args.train_pool, args.source, args.output_dir, repo_root, args.seed)
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"counts": manifest["counts"], "rejected": manifest["selection"]["rejected"],
                      "candidates_visited": manifest["selection"]["candidates_visited"],
                      "output_dir": _rel(args.output_dir, repo_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
