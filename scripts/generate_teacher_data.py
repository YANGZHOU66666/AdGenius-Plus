#!/usr/bin/env python3
"""Generate and freeze teacher ads for the training and dev ProductItems.

For each split in data/train/v1 (dev_products.jsonl, train_products.jsonl) the
teacher writes one ad per product with the frozen gold generation prompt. Runs
are resumable: every attempt is logged under
artifacts/generation/train_v1/<split>/, and products that already have a
successful result are skipped on the next run.

Once every product of a split has a successful result, the split is frozen
into data/train/v1/teacher_<alias>_<split>.jsonl (GenerationResult, in the
split's line order) and teacher_<alias>.manifest.json records the model,
prompt hash, sampling settings, counts, token usage and output hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema
from dotenv import load_dotenv

from adgenius.shared.llm import LLMClient, ModelConfig
from adgenius.shared.product import render_product_information

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("generate_candidates", REPO_ROOT / "scripts" / "generate_candidates.py")
_candidates = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_candidates)
SYSTEM_PROMPT: str = _candidates.SYSTEM_PROMPT
generate_with_retries = _candidates.generate_with_retries

SPLITS = {"dev": "dev_products.jsonl", "train": "train_products.jsonl"}
DATA_VERSION = "v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version() -> dict[str, Any]:
    """Git commit of the generating code, and whether code directories had local changes."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "--no-optional-locks", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", "scripts", "src", "configs", "metadata")
    return {"git_commit": commit or None, "code_paths_dirty": bool(dirty) if commit else None}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def candidate_id(run_id: str, product_id: str) -> str:
    key = json.dumps([run_id, product_id, 0], ensure_ascii=False).encode("utf-8")
    return "cand_" + hashlib.sha256(key).hexdigest()[:24]


def latest_results(path: Path) -> dict[str, dict[str, Any]]:
    """Last recorded result per product (a later success supersedes an earlier failure)."""
    latest: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        latest[record["product_id"]] = record
    return latest


def run_split(split: str, items: list[dict[str, Any]], config: ModelConfig, alias: str, run_dir: Path,
              *, workers: int, temperature: float, max_output_tokens: int, max_retries: int,
              retry_delay: float) -> dict[str, dict[str, Any]]:
    run_id = f"gen_teacher_{alias}_{split}_{DATA_VERSION}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path, calls_path = run_dir / f"results_{alias}.jsonl", run_dir / f"calls_{alias}.jsonl"
    done = {pid for pid, rec in latest_results(results_path).items()
            if rec["generation_result"]["status"] == "success"}
    todo = [item for item in items if item["product_id"] not in done]
    print(f"[{split}] {len(done)} already done, {len(todo)} to generate", flush=True)
    lock = threading.Lock()
    local = threading.local()
    clients: list[LLMClient] = []

    with results_path.open("a", encoding="utf-8") as results_file, calls_path.open("a", encoding="utf-8") as calls_file:
        def log_call(record: dict[str, Any]) -> None:
            with lock:
                calls_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                calls_file.flush()

        def client() -> LLMClient:
            if not hasattr(local, "client"):
                local.client = LLMClient(config)
                with lock:
                    clients.append(local.client)
            return local.client

        def work(item: dict[str, Any]) -> str:
            cid = candidate_id(run_id, item["product_id"])
            outcome, trace_id = generate_with_retries(
                client(), model_alias=alias, candidate_id=cid,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": render_product_information(item)}],
                max_output_tokens=max_output_tokens, temperature=temperature,
                max_retries=max_retries, retry_delay=retry_delay, log_call=log_call)
            record = {"candidate_id": cid, "product_id": item["product_id"], "run_id": run_id,
                      "sample_index": 0, "generation_result": outcome, "trace_id": trace_id}
            with lock:
                results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                results_file.flush()
            return outcome["status"]

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(work, item) for item in todo]
                for count, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    if count % 200 == 0 or count == len(futures):
                        print(f"[{split}] {count}/{len(futures)} processed", flush=True)
        finally:
            for c in clients:
                c.close()
    return latest_results(results_path)


def usage_totals(calls_path: Path) -> dict[str, int]:
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    for record in read_jsonl(calls_path):
        totals["calls"] += 1
        usage = (record.get("response") or {}).get("usage") or {}
        totals["input_tokens"] += usage.get("input_tokens") or 0
        totals["output_tokens"] += usage.get("output_tokens") or 0
    return totals


def freeze(split: str, items: list[dict[str, Any]], latest: dict[str, dict[str, Any]], alias: str,
           data_dir: Path, validator: jsonschema.Draft202012Validator) -> Path | None:
    missing = [item["product_id"] for item in items
               if latest.get(item["product_id"], {}).get("generation_result", {}).get("status") != "success"]
    if missing:
        print(f"[{split}] not frozen: {len(missing)} products lack a successful result "
              f"(rerun to retry), e.g. {missing[:3]}", flush=True)
        return None
    out = data_dir / f"teacher_{alias}_{split}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for item in items:
            record = latest[item["product_id"]]
            validator.validate(record)
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return out


def word_stats(path: Path) -> dict[str, Any]:
    words = [len(r["generation_result"]["text"].split()) for r in read_jsonl(path)]
    words.sort()
    return {"mean_words": round(sum(words) / len(words), 2), "median_words": words[len(words) // 2],
            "over_25_words": sum(w > 25 for w in words), "over_35_words": sum(w > 35 for w in words)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen397b", help="Config alias in configs/models.toml (mock for a dry run)")
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=["dev", "train"])
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "train" / DATA_VERSION)
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "artifacts" / "generation" / f"train_{DATA_VERSION}")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "models.toml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--limit", type=int, help="Only the first N products of each split (smoke test; never frozen)")
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive")

    load_dotenv(REPO_ROOT / ".env")
    gold = json.loads((REPO_ROOT / "data" / "eval" / "gold" / "manifest.json").read_text(encoding="utf-8"))
    prompt_sha = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    if prompt_sha != gold["generation"]["system_prompt_sha256"]:
        parser.error("Generation prompt differs from the frozen gold prompt")
    config = replace(ModelConfig.from_toml(args.config, args.model), max_retries=0)
    if config.api_key_env and not os.environ.get(config.api_key_env, "").strip():
        parser.error(f"Set environment variable {config.api_key_env}")
    schema = json.loads((REPO_ROOT / "metadata" / "schemas" / "generation_result.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)

    manifest_path = args.data_dir / f"teacher_{args.model}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "version": DATA_VERSION, "teacher_alias": args.model, "splits": {}}
    for split in args.splits:
        items = read_jsonl(args.data_dir / SPLITS[split])
        if not items:
            parser.error(f"No products in {args.data_dir / SPLITS[split]}; run scripts/build_train_split.py first")
        if args.limit:
            items = items[:args.limit]
        run_dir = args.run_dir / split if not args.limit else args.run_dir / f"{split}_limit{args.limit}"
        latest = run_split(split, items, config, args.model, run_dir, workers=args.workers,
                           temperature=args.temperature, max_output_tokens=args.max_output_tokens,
                           max_retries=args.max_retries, retry_delay=args.retry_delay)
        successes = sum(latest.get(i["product_id"], {}).get("generation_result", {}).get("status") == "success"
                        for i in items)
        print(f"[{split}] {successes}/{len(items)} successful", flush=True)
        if args.limit:
            continue
        out = freeze(split, items, latest, args.model, args.data_dir, validator)
        if out is None:
            continue
        manifest["splits"][split] = {
            "products": len(items),
            "input": {"path": rel(args.data_dir / SPLITS[split]),
                      "sha256": sha256_file(args.data_dir / SPLITS[split])},
            "output": {"path": rel(out), "sha256": sha256_file(out)},
            "run_id": f"gen_teacher_{args.model}_{split}_{DATA_VERSION}",
            "calls_log": rel(run_dir / f"calls_{args.model}.jsonl"),
            "usage": usage_totals(run_dir / f"calls_{args.model}.jsonl"),
            "length": word_stats(out),
            "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        schema_path = REPO_ROOT / "metadata" / "schemas" / "generation_result.schema.json"
        schema_title = json.loads(schema_path.read_text(encoding="utf-8"))["title"]
        manifest["schemas"] = {"generations": {"path": rel(schema_path), "version": schema_title,
                                               "sha256": sha256_file(schema_path)}}
        manifest["code"] = {"script": "scripts/generate_teacher_data.py", **code_version()}
        manifest["generation"] = {
            "model": config.snapshot(), "system_prompt_sha256": prompt_sha,
            "system_prompt_source": "scripts/generate_candidates.py SYSTEM_PROMPT (identical to the gold manifest)",
            "user_message": "adgenius.shared.product.render_product_information(ProductItem)",
            "temperature": args.temperature, "max_output_tokens": args.max_output_tokens,
            "samples_per_product": 1,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[{split}] frozen to {rel(out)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
