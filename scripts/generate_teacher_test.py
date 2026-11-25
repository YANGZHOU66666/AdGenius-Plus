"""Generate Qwen3.5-397B teacher ads for the 1,000 test products (a reference system for the final evaluation).

Same setup as the frozen teacher data (scripts/generate_teacher_data.py, data/train/v1/teacher_qwen397b.manifest.json):
model alias qwen397b (Bailian qwen3.5-397b-a17b, thinking off, top_p 1.0), temperature 1.0, max_output_tokens 512,
the frozen gold system prompt and render_product_information(ProductItem). Uses the repo's run_split read-only, so
transport retries and the call journal are identical; it is not re-run to replace outputs based on their content.

    python scripts/generate_teacher_test.py --limit 5     # pilot (separate run directory)
    python scripts/generate_teacher_test.py               # all 1,000; writes teacher_qwen397b_test.jsonl and manifest.json

Writes to --out-dir (default artifacts/generation/teacher_qwen397b_test_v1/, ignored): the call journal under run/,
the GenerationResult file and its manifest. The shared copy is results/training/test_references_v1/ (copied and
validated when the formal runs were landed). The landed outputs were produced by this script at commit 7b9ccde
(then kept outside the repository; only the paths differ).
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
_SPEC = importlib.util.spec_from_file_location("generate_teacher_data", REPO / "scripts" / "generate_teacher_data.py")
gtd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gtd)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out-dir", type=Path, default=REPO / "artifacts/generation/teacher_qwen397b_test_v1")
    args = p.parse_args()
    out_dir = args.out_dir
    gtd.load_dotenv(REPO / ".env")
    gold = json.loads((REPO / "data/eval/gold/manifest.json").read_text(encoding="utf-8"))
    prompt_sha = hashlib.sha256(gtd.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert prompt_sha == gold["generation"]["system_prompt_sha256"], "generation prompt differs from the frozen gold prompt"
    config = gtd.replace(gtd.ModelConfig.from_toml(REPO / "configs/models.toml", "qwen397b"), max_retries=0)
    products_path = REPO / "data/eval/test/product_items.jsonl"
    items = gtd.read_jsonl(products_path)
    assert len(items) == 1000
    if args.limit:
        items = items[:args.limit]
    run_dir = out_dir / ("run" if not args.limit else f"run_limit{args.limit}")
    latest = gtd.run_split("test", items, config, "qwen397b", run_dir, workers=args.workers, temperature=1.0,
                           max_output_tokens=512, max_retries=3, retry_delay=2.0)
    records = [latest[i["product_id"]] for i in items if i["product_id"] in latest]
    status = Counter(r["generation_result"]["status"] for r in records)
    print(json.dumps({"records": len(records), "status": status}), flush=True)
    if args.limit:
        for r in records:
            print(" -", r["generation_result"]["status"], (r["generation_result"]["text"] or "")[:160])
        return
    validator = gtd.jsonschema.Draft202012Validator(json.loads((REPO / "metadata/schemas/generation_result.schema.json").read_text()))
    for r in records:
        validator.validate(r)
    out = out_dir / "teacher_qwen397b_test.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    usage = Counter()
    calls = gtd.read_jsonl(run_dir / "calls_qwen397b.jsonl")
    for c in calls:
        u = (c.get("response") or {}).get("usage") or {}
        usage["input_tokens"] += u.get("input_tokens") or 0
        usage["output_tokens"] += u.get("output_tokens") or 0
    words = [len((r["generation_result"]["text"] or "").split()) for r in records]
    manifest = {
        "dataset": "teacher_qwen397b_test (reference outputs for the final evaluation)",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": "gen_teacher_qwen397b_test_v1",
        "products": {"path": "data/eval/test/product_items.jsonl", "sha256": sha(products_path), "records": len(items)},
        "output": {"path": out.name, "sha256": sha(out), "records": len(records), "status_counts": dict(status)},
        "generation": {"model": config.snapshot(), "system_prompt_sha256": prompt_sha,
                       "user_message": "adgenius.shared.product.render_product_information(ProductItem)",
                       "temperature": 1.0, "max_output_tokens": 512, "samples_per_product": 1,
                       "same_settings_as": "data/train/v1/teacher_qwen397b.manifest.json"},
        "calls": {"attempts": len(calls), "usage": dict(usage), "journal": "run/calls_qwen397b.jsonl"},
        "length_words": {"mean": round(sum(words) / len(words), 1), "over_35": sum(w > 35 for w in words)},
        "code": {"script": "scripts/generate_teacher_test.py",
                 "repo_functions": "scripts/generate_teacher_data.py run_split (read-only)", **gtd.code_version()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("output", "calls", "length_words")}, indent=1))


if __name__ == "__main__":
    main()
