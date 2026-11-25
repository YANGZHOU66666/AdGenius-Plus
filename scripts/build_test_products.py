"""Convert the frozen test products to ProductItem v1 (data/eval/test/), using the same rule as gold.

Input: data/processed/v1/test_products.jsonl (1,000 source-format records) and its split index
data/splits/v1/test.jsonl (same line order). Output:
- data/eval/test/product_items.jsonl: one ProductItem per line, in the source line order;
  product_id = "test_product_" + six-digit line index, source_id = parent_asin,
  product_info = the source record without parent_asin (values kept as they are; nothing is cleaned or filtered);
- data/eval/test/manifest.json: input hashes, ID mapping, schema version, code commit, counts and checks.

Test is an evaluation-only set: no product is dropped (unlike the train/dev filters); descriptive counts of
products the training filters would have rejected are recorded instead. The outputs are re-validated: count,
schema, order against the split index, input hashes against data/splits/v1/manifest.json, and no ASIN or content
group shared with gold, dev or train.

    PYTHONPATH=src .venv/bin/python scripts/build_test_products.py           # refuses to overwrite
    PYTHONPATH=src .venv/bin/python scripts/build_test_products.py --force   # rebuild; byte-identical
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema

from adgenius.shared.product import render_product_information

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("build_train_split", REPO_ROOT / "scripts" / "build_train_split.py")
_split = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_split)
code_version = _split.code_version

SOURCE = Path("data/processed/v1/test_products.jsonl")
INDEX = Path("data/splits/v1/test.jsonl")
SPLIT_MANIFEST = Path("data/splits/v1/manifest.json")
GOLD_INDEX = Path("data/splits/v1/gold.jsonl")
TRAIN_INDEX = Path("data/train/v1/split_index.jsonl")
SCHEMA = Path("metadata/schemas/product_item.schema.json")
OUT_DIR = Path("data/eval/test")
EXPECTED = 1000
MAX_RENDERED_CHARS = 6000  # the train/dev filter, recorded here for description only


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_product_item(index: int, record: dict[str, Any]) -> dict[str, Any]:
    info = dict(record)
    source_id = info.pop("parent_asin")
    return {"product_id": f"test_product_{index:06d}", "source_id": source_id, "product_info": info}


def build(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source, index = read_jsonl(root / SOURCE), read_jsonl(root / INDEX)
    split_manifest = json.loads((root / SPLIT_MANIFEST).read_text(encoding="utf-8"))
    for rel in (SOURCE, INDEX):
        if sha256_file(root / rel) != split_manifest["outputs_sha256"][rel.as_posix()]:
            raise RuntimeError(f"{rel} does not match the hash in {SPLIT_MANIFEST}")
    if len(source) != EXPECTED or len(index) != EXPECTED:
        raise RuntimeError(f"expected {EXPECTED} test records, got {len(source)} and {len(index)}")
    if [r["parent_asin"] for r in source] != [r["product_id"] for r in index]:
        raise RuntimeError("source order differs from the test split index")
    items = [to_product_item(i, r) for i, r in enumerate(source)]

    def renders(item):
        try:
            return render_product_information(item)
        except ValueError:
            return None

    rendered = [renders(item) for item in items]
    info = [item["product_info"] for item in items]
    descriptive = {
        "missing_title": sum(not (i.get("title") or "").strip() for i in info),
        "missing_features_and_description": sum(not i.get("features") and not i.get("description") for i in info),
        "not_renderable": sum(r is None for r in rendered),
        f"rendered_over_{MAX_RENDERED_CHARS}_chars": sum(r is not None and len(r) > MAX_RENDERED_CHARS for r in rendered),
    }
    manifest = {
        "dataset_id": "test_v1",
        "status": "frozen",
        "purpose": "final evaluation only; never used for training, prompt tuning, judge selection or model selection",
        "schemas": {"products": {"path": SCHEMA.as_posix(), "version": "ProductItem v1", "sha256": sha256_file(root / SCHEMA)}},
        "source": {"products": {"path": SOURCE.as_posix(), "sha256": sha256_file(root / SOURCE)},
                   "split_index": {"path": INDEX.as_posix(), "sha256": sha256_file(root / INDEX)},
                   "split_manifest": SPLIT_MANIFEST.as_posix()},
        "source_id_field": "parent_asin",
        "id_mapping": "product_id = 'test_product_' + six-digit zero-padded line index of the source file (same rule as "
                      "gold_product_NNNNNN); it never changes across rebuilds",
        "conversion": "source_id = parent_asin; product_info = the source record without parent_asin, values unchanged",
        "code": {"script": "scripts/build_test_products.py", **code_version(root)},
        "counts": {"products": len(items)},
        "descriptive": descriptive,
        "notes": ["No product is filtered: the test set is fixed. Descriptive counts show how many products the "
                  "train/dev filters would have rejected; generation still runs on all 1,000."],
    }
    return items, manifest


def validate(root: Path, items: list[dict[str, Any]]) -> dict[str, Any]:
    validator = jsonschema.Draft202012Validator(json.loads((root / SCHEMA).read_text(encoding="utf-8")))
    for item in items:
        validator.validate(item)
    ids, sources = [i["product_id"] for i in items], [i["source_id"] for i in items]
    if len(set(ids)) != len(ids) or len(set(sources)) != len(sources):
        raise RuntimeError("duplicate product_id or source_id in test")
    test_groups = {r["group_id"] for r in read_jsonl(root / INDEX)}
    gold = read_jsonl(root / GOLD_INDEX)
    train = read_jsonl(root / TRAIN_INDEX)
    overlaps = {
        "gold_asins": len(set(sources) & {r["product_id"] for r in gold}),
        "gold_groups": len(test_groups & {r["group_id"] for r in gold}),
        "dev_train_asins": len(set(sources) & {r["source_id"] for r in train}),
        "dev_train_groups": len(test_groups & {r["group_id"] for r in train}),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"test overlaps another split: {overlaps}")
    raw = read_jsonl(root / SOURCE)
    for item, record in zip(items, raw):
        expected = dict(record)
        if item["source_id"] != expected.pop("parent_asin") or item["product_info"] != expected:
            raise RuntimeError(f"conversion changed {item['product_id']}")
    return {"schema_valid": True, "unique_ids": True, "unchanged_values": True, "overlaps": overlaps}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out_dir = root / OUT_DIR
    out = out_dir / "product_items.jsonl"
    if out.exists() and not args.force:
        print(f"{OUT_DIR}/product_items.jsonl exists; use --force to rebuild", file=sys.stderr)
        return 2
    items, manifest = build(root)
    manifest["checks"] = validate(root, items)
    out_dir.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest["files"] = {"product_items.jsonl": {"path": (OUT_DIR / "product_items.jsonl").as_posix(),
                                                 "sha256": sha256_file(out), "records": len(items)}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "descriptive", "checks", "code")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
