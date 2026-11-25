"""Check single ads against their product information with an LLM rubric judge.

The judge sees the same rendered product information as the generator and the human annotator, plus
one ad, and answers factual_support (does the ad conflict with the information?). One request per
item; transport and parse failures are retried up to ``max_attempts``. Results are RubricResult v1
model records in input order.

Two inputs: the frozen human-labelled rubric sample (default), or ``--candidates`` GenerationResult
files from any generator, judged in full against ``--products``. Each run writes metrics.json with the
judged conflict rate per generator; the generator name is never sent to the judge.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import threading
import time
import tomllib
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema
from dotenv import load_dotenv

from adgenius.eval.rubric_metrics import generator_rates
from adgenius.shared.llm import LLMClient, ModelConfig
from adgenius.shared.product import render_product_information

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLD = REPO_ROOT / "data" / "eval" / "gold"
SCHEMA_DIR = REPO_ROOT / "metadata" / "schemas"
CRITERION = "factual_support"
METRICS_SCHEMA_VERSION = "rubric_generator_metrics_v1"
OUTPUT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["value", "conflicts", "reason"],
    "properties": {
        "value": {"type": "boolean"},
        "reason": {"type": "string"},
        "conflicts": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["ad", "information"],
            "properties": {"ad": {"type": "string"}, "information": {"type": "string"}}}},
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def load_settings(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        judge = tomllib.load(handle).get("judge")
    required = {"model_alias", "protocol_version", "input_mode", "human_annotation_protocol_version",
                "criteria", "system_prompt", "user_prompt", "max_output_tokens", "max_attempts",
                "provider_options"}
    if not isinstance(judge, dict) or not required <= set(judge) or set(judge) - required - {"temperature"}:
        raise ValueError(f"[judge] requires: {', '.join(sorted(required))}; optional: temperature")
    if judge["input_mode"] != "product_and_ad":
        raise ValueError("Rubric judge supports input_mode = 'product_and_ad'")
    if judge["criteria"] != [CRITERION]:
        raise ValueError("Rubric judge currently supports criteria = ['factual_support']")
    for key in ("max_output_tokens", "max_attempts"):
        if type(judge[key]) is not int or judge[key] < 1:
            raise ValueError(f"judge.{key} must be a positive integer")
    temperature = judge.setdefault("temperature", None)
    if temperature is not None and (type(temperature) not in (int, float) or not 0 <= temperature <= 2):
        raise ValueError("judge.temperature must be between 0 and 2 when specified")
    if not isinstance(judge["provider_options"], dict):
        raise ValueError("judge.provider_options must be a table")
    return judge


def load_items(gold_dir: Path = GOLD, limit: int | None = None) -> list[dict[str, Any]]:
    """Sample rows with the ad text and rendered product information; the generator is not sent."""
    products = {p["product_id"]: p for p in _read_jsonl(gold_dir / "product_items.jsonl")}
    sample = _read_jsonl(gold_dir / "rubric_sample.jsonl")
    texts: dict[str, tuple[str, str]] = {}
    for generator in sorted({row["generator"] for row in sample}):
        for result in _read_jsonl(gold_dir / f"results_{generator}.jsonl"):
            texts[result["candidate_id"]] = (result["product_id"], result["generation_result"]["text"])
    items = []
    for row in sample[:limit]:
        product_id, ad = texts[row["candidate_id"]]
        if product_id != row["product_id"] or not isinstance(ad, str) or not ad.strip():
            raise ValueError(f"Rubric sample row does not match the gold data: {row['item_id']}")
        items.append({"item_id": row["item_id"], "candidate_id": row["candidate_id"],
                      "product_id": product_id, "generator": row["generator"], "ad": ad,
                      "information": render_product_information(products[product_id])})
    return items


def generator_name(run_id: str) -> str:
    """gen_<alias>_<UTC stamp> -> alias; other run IDs are kept whole."""
    match = re.fullmatch(r"gen_(.+)_\d{8}T\d{6}Z", run_id)
    return match.group(1) if match else run_id


def load_candidates(candidate_paths: list[Path], products_path: Path,
                    limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every usable GenerationResult in the given files, in file order.

    Success and truncated ads are judged (a truncated ad is what the generator produced). Failed
    generations have no ad to judge; they are counted per generator and skipped.
    """
    products = {p["product_id"]: p for p in _read_jsonl(products_path)}
    validator = _validator("generation_result.schema.json")
    items, seen = [], set()
    skipped: dict[str, int] = {}
    for path in candidate_paths:
        for row in _read_jsonl(path):
            validator.validate(row)
            candidate_id, product_id = row["candidate_id"], row["product_id"]
            if candidate_id in seen:
                raise ValueError(f"Duplicate candidate_id across candidate files: {candidate_id}")
            seen.add(candidate_id)
            if product_id not in products:
                raise ValueError(f"Candidate {candidate_id} refers to a product missing from {products_path}")
            generator = generator_name(row["run_id"])
            result = row["generation_result"]
            if result["status"] not in ("success", "truncated") or not (result["text"] or "").strip():
                skipped[generator] = skipped.get(generator, 0) + 1
                continue
            items.append({"item_id": candidate_id, "candidate_id": candidate_id, "product_id": product_id,
                          "generator": generator, "ad": result["text"],
                          "information": render_product_information(products[product_id])})
    if not items:
        raise ValueError("No successful candidates to judge")
    return items[:limit], {"skipped_failed_generations": skipped}


def messages(system: str, template: str, item: dict[str, Any]) -> list[dict[str, str]]:
    user = template.replace("{product}", item["information"]).replace("{ad}", item["ad"])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def judge_request(config: ModelConfig, settings: dict[str, Any], chat: list[dict[str, str]]) -> dict[str, Any]:
    options = copy.deepcopy(settings["provider_options"])
    if config.provider == "anthropic":
        if settings["temperature"] is not None:
            raise ValueError("Omit judge.temperature for Claude judges")
        merged = copy.deepcopy(config.request_options)
        merged.update(options)
        merged.setdefault("output_config", {})["format"] = {"type": "json_schema", "schema": OUTPUT_SCHEMA}
        return {"messages": chat, "temperature": None, "max_output_tokens": settings["max_output_tokens"],
                "json_mode": False, "provider_options": merged}
    return {"messages": chat, "temperature": settings["temperature"],
            "max_output_tokens": settings["max_output_tokens"], "json_mode": True, "provider_options": options}


def parse_verdict(text: str) -> dict[str, Any]:
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("value"), bool):
        raise ValueError("value is not boolean")
    conflicts = data.get("conflicts")
    if not isinstance(conflicts, list) or not all(isinstance(c, dict) for c in conflicts):
        raise ValueError("conflicts is not a list of objects")
    reason = data.get("reason")
    return {"value": data["value"], "conflicts": conflicts, "reason": reason if isinstance(reason, str) else ""}


def format_reason(verdict: dict[str, Any]) -> str | None:
    """Fold the quoted conflicts into the schema's single reason string."""
    parts = [f'ad "{c.get("ad", "")}" vs information "{c.get("information", "")}"' for c in verdict["conflicts"]]
    text = verdict["reason"].strip()
    if parts:
        text = (text + " | " if text else "") + "conflicts: " + "; ".join(parts)
    return text or None


def result_record(item: dict[str, Any], run_id: str, protocol: str, verdict: dict[str, Any] | None,
                  trace_id: str | None, error: str | None = None) -> dict[str, Any]:
    if verdict is None:
        rubric = {"status": "failed", "criteria": [], "error": error or "judge_failed"}
    else:
        rubric = {"status": "success", "error": None, "criteria": [
            {"criterion_id": CRITERION, "value": verdict["value"], "reason": format_reason(verdict)}]}
    return {"result_id": f"rb_{_hash(json.dumps([run_id, item['candidate_id']]).encode())[:24]}",
            "judge_type": "model", "candidate_id": item["candidate_id"], "product_id": item["product_id"],
            "judge_run_id": run_id, "protocol_version": protocol, "rubric_result": rubric, "trace_id": trace_id}


def _validator(name: str = "rubric_result.schema.json") -> jsonschema.Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def _relative(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(REPO_ROOT)) if resolved.is_relative_to(REPO_ROOT) else str(resolved)


def specification(settings, config, paths, items) -> dict[str, Any]:
    code = [Path(__file__), Path(__file__).with_name("rubric_metrics.py"), REPO_ROOT / "src/adgenius/shared/llm.py",
            REPO_ROOT / "src/adgenius/shared/product.py", SCHEMA_DIR / "rubric_result.schema.json",
            SCHEMA_DIR / "generation_result.schema.json", SCHEMA_DIR / "rubric_generator_metrics.schema.json"]
    return {"settings": settings, "model_config": config.snapshot(),
            "input_files": {name: {"path": _relative(path), "sha256": _hash(path.read_bytes())}
                            for name, path in paths.items()},
            "code_and_schema_sha256": {str(p.relative_to(REPO_ROOT)): _hash(p.read_bytes()) for p in code},
            "candidate_ids": [item["candidate_id"] for item in items]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(Path(args.env_file) if args.env_file else REPO_ROOT / ".env", override=False)
    config_path = Path(args.eval_config).resolve()
    settings = load_settings(config_path)
    config = ModelConfig.from_toml(REPO_ROOT / "configs" / "models.toml", settings["model_alias"])
    paths = {"system_prompt": REPO_ROOT / settings["system_prompt"], "user_prompt": REPO_ROOT / settings["user_prompt"],
             "eval_config": config_path}
    extra: dict[str, Any] = {}
    if args.candidates:
        candidate_paths = [Path(path).resolve() for path in args.candidates]
        products_path = Path(args.products).resolve() if args.products else GOLD / "product_items.jsonl"
        paths["product_items"] = products_path
        paths.update({f"candidates_{i}": path for i, path in enumerate(candidate_paths)})
        items, extra = load_candidates(candidate_paths, products_path, args.limit)
        mode = "candidates"
    else:
        if args.products:
            raise ValueError("--products is used with --candidates; the rubric sample uses --gold-dir")
        gold = Path(args.gold_dir).resolve()
        paths.update(rubric_sample=gold / "rubric_sample.jsonl", product_items=gold / "product_items.jsonl")
        items = load_items(gold, args.limit)
        mode = "rubric_sample"
    system, template = (paths[k].read_text(encoding="utf-8") for k in ("system_prompt", "user_prompt"))
    if "{product}" not in template or "{ad}" not in template:
        raise ValueError("The user prompt must contain {product} and {ad}")
    run_id = args.run_id
    if re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run_id may contain only letters, digits, underscores and hyphens")
    output = Path(args.output_dir).resolve() if args.output_dir else REPO_ROOT / "results/rubric" / run_id
    calls_path = (Path(args.calls_dir).resolve() if args.calls_dir else REPO_ROOT / "artifacts/rubric" / run_id) / "calls.jsonl"
    spec = specification(settings, config, paths, items)
    spec.update(input_mode=mode, **extra)
    fingerprint = _hash(_json_bytes(spec))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config_fingerprint"] != fingerprint:
            raise ValueError("Inputs, config or code changed; use a new --run-id")
    else:
        manifest = {"run_id": run_id, "created_at": _now(), "status": "running",
                    "config_fingerprint": fingerprint, "specification": spec}
    counts: dict[str, int] = {}
    for item in items:
        counts[item["generator"]] = counts.get(item["generator"], 0) + 1
    if args.dry_run:
        return {"dry_run": True, "run_id": run_id, "model": config.model, "input_mode": mode,
                "items": len(items), "items_by_generator": counts, **extra,
                "output_dir": str(output), "calls_file": str(calls_path)}

    results_path = output / "results.jsonl"
    saved = {r["candidate_id"]: r for r in _read_jsonl(results_path)} if results_path.exists() else {}
    calls = _read_jsonl(calls_path) if calls_path.exists() else []
    todo = [it for it in items if saved.get(it["candidate_id"], {}).get("rubric_result", {}).get("status") != "success"]
    if todo and config.api_key_env and not os.environ.get(config.api_key_env, "").strip():
        raise ValueError(f"Set {config.api_key_env} in .env or the environment")
    validator, lock, local = _validator(), threading.Lock(), threading.local()

    def save() -> None:
        _write_rows(results_path, [saved[it["candidate_id"]] for it in items if it["candidate_id"] in saved])
        _write_rows(calls_path, calls)

    def work(item: dict[str, Any]) -> None:
        if not hasattr(local, "client"):
            local.client = LLMClient(config)
        request = judge_request(config, settings, messages(system, template, item))
        error, trace = None, None
        for attempt in range(1, settings["max_attempts"] + 1):
            trace, started, text, usage = f"call_rb_{uuid.uuid4().hex}", time.perf_counter(), None, None
            row = {"trace_id": trace, "judge_run_id": run_id, "candidate_id": item["candidate_id"],
                   "item_id": item["item_id"], "attempt": attempt, "created_at": _now()}
            try:
                response = local.client.generate(**request)
                text, usage = response.text, response.to_dict().get("usage")
                verdict = parse_verdict(text)
            except Exception as exc:  # noqa: BLE001 - transport and parse failures are retried
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
                row.update(status="failed", error=error, text=text, usage=usage,
                           latency_seconds=time.perf_counter() - started)
                with lock:
                    calls.append(row)
                time.sleep(2 ** attempt)
                continue
            record = result_record(item, run_id, settings["protocol_version"], verdict, trace)
            validator.validate(record)
            row.update(status="success", text=text, usage=usage, latency_seconds=time.perf_counter() - started)
            with lock:
                calls.append(row)
                saved[item["candidate_id"]] = record
                save()
            return
        record = result_record(item, run_id, settings["protocol_version"], None, trace, "judge_failed")
        validator.validate(record)
        with lock:
            saved[item["candidate_id"]] = record
            save()
        print(f"FAILED {item['item_id']}: {error}", file=sys.stderr, flush=True)

    manifest.update(status="running", updated_at=_now())
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(_json_bytes(manifest))
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, todo))
    save()
    failed = sum(saved[it["candidate_id"]]["rubric_result"]["status"] != "success" for it in items)
    metrics = {"schema_version": METRICS_SCHEMA_VERSION, "run_id": run_id, "input_mode": mode,
               **generator_rates([saved[it["candidate_id"]] for it in items], items, calls), **extra}
    _validator("rubric_generator_metrics.schema.json").validate(metrics)
    (output / "metrics.json").write_bytes(_json_bytes(metrics))
    manifest.update(status="completed" if not failed else "completed_with_failures", updated_at=_now())
    manifest_path.write_bytes(_json_bytes(manifest))
    return {"run_id": run_id, "status": manifest["status"], "items": len(items), "failed": failed,
            "false_rate_by_generator": {g: m["false_rate"] for g, m in metrics["by_generator"].items()},
            "output_dir": str(output), "calls_file": str(calls_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-config", default=str(REPO_ROOT / "configs/eval/rubric_deepseek.toml"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidates", nargs="+", metavar="RESULTS_JSONL",
                        help="GenerationResult files to judge in full instead of the human-labelled rubric sample")
    parser.add_argument("--products", help="ProductItem file for --candidates; default data/eval/gold/product_items.jsonl")
    parser.add_argument("--gold-dir", default=str(GOLD), help="rubric sample mode only")
    parser.add_argument("--output-dir", help="default: results/rubric/<run_id>")
    parser.add_argument("--calls-dir", help="default: artifacts/rubric/<run_id>")
    parser.add_argument("--env-file")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run(build_parser().parse_args(argv)), ensure_ascii=False, indent=2))
    return 0
