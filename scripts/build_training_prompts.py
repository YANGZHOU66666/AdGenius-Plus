"""Freeze the student prompts for training, dev and test generation (data/train/v1 and data/eval/test).

Each output line is {"product_id", "rank", "prompt"} where prompt is the same two messages that produced the gold
candidates and the teacher ads: the frozen SYSTEM_PROMPT of scripts/generate_candidates.py and
render_product_information(ProductItem). rank is the product's line index in the split file, which defines the
nested 1k/2k/4k/8k scales. The files are written once and then uploaded to the training VM; prompts_manifest.json
records the source commit and hashes so any run can be checked against them.

    PYTHONPATH=src .venv/bin/python scripts/build_training_prompts.py --out artifacts/train/prompts/v1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

from adgenius.shared.product import render_product_information

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("generate_candidates", REPO_ROOT / "scripts" / "generate_candidates.py")
_candidates = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_candidates)
SYSTEM_PROMPT: str = _candidates.SYSTEM_PROMPT
SPLITS = {"prompts_train_v1.jsonl": "train_products.jsonl", "prompts_dev_v1.jsonl": "dev_products.jsonl"}
TEST_PRODUCTS = REPO_ROOT / "data" / "eval" / "test" / "product_items.jsonl"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_rows(products: list[dict]) -> list[dict]:
    return [{"product_id": item["product_id"], "rank": rank,
             "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": render_product_information(item)}]}
            for rank, item in enumerate(products)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "train" / "v1")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "train" / "prompts" / "v1")
    parser.add_argument("--test-products", type=Path, default=TEST_PRODUCTS,
                        help="also writes prompts_test_v1.jsonl when this file exists")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {"source_commit": commit or None,
                "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                "renderer": "adgenius.shared.product.render_product_information", "files": {}}
    sources = {name: args.data_dir / source for name, source in SPLITS.items()}
    if args.test_products.exists():
        sources["prompts_test_v1.jsonl"] = args.test_products
    for name, src in sources.items():
        products = [json.loads(line) for line in src.open(encoding="utf-8") if line.strip()]
        out = args.out / name
        with out.open("w", encoding="utf-8") as handle:
            for row in prompt_rows(products):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["files"][name] = {"source": str(src.relative_to(REPO_ROOT)) if src.is_relative_to(REPO_ROOT) else str(src),
                                   "source_sha256": sha256(src), "rows": len(products), "sha256": sha256(out)}
    (args.out / "prompts_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
