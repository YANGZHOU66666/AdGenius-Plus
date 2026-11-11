"""Repair failed candidates in place, keeping a resumable generation run.

By default, update the gold v1 results/calls in their original directory.
Before the first repair, preserve original JSONL files in sibling ``v1_original/``.
Generation settings are recovered from the original call logs, not CLI defaults.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from adgenius.shared.llm import LLMClient, ModelConfig
from generate_candidates import _load_items, _product_prompt, generate_with_retries


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{number}")
            records.append(record)
    return records


def _atomic_write(path: Path, content: str) -> None:
    """Publish a complete file; interruption cannot leave half a JSONL record."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_results(path: Path, records: list[dict[str, Any]]) -> None:
    _atomic_write(path, "".join(json.dumps(row, ensure_ascii=False) + "\n"
                               for row in records))


def _snapshot(source: Path, backup: Path, names: list[str]) -> None:
    """Publish a complete sibling backup before changing any source JSONL."""
    temporary = Path(tempfile.mkdtemp(prefix=f".{backup.name}.", dir=backup.parent))
    try:
        for name in names:
            shutil.copyfile(source / name, temporary / name)
        temporary.rename(backup)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


@contextmanager
def _output_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".retry.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another retry process is using {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _template(call: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    """Read either a new attempt log or the original provider-native request."""
    response = call.get("response") or {}
    config = call.get("config") or response.get("config")
    generation = call.get("generation_request")
    if not config:
        return None  # Old failed-call logs did not store request settings.
    if generation is None:
        request = response.get("request")
        if not request:
            return None
        provider = config["provider"]
        if provider == "google_generative":
            messages = [{"role": "system", "content": "".join(
                part["text"] for part in request["systemInstruction"]["parts"])}]
            messages.extend({
                "role": "assistant" if content["role"] == "model" else content["role"],
                "content": "".join(part["text"] for part in content["parts"]),
            } for content in request["contents"])
            options = request["generationConfig"]
            budget = options["maxOutputTokens"]
        else:
            messages = copy.deepcopy(request["messages"])
            if provider == "anthropic" and request.get("system"):
                messages.insert(0, {"role": "system", "content": request["system"]})
            options = request.get("options", {}) if provider == "ollama" else request
            budget = options["num_predict"] if provider == "ollama" else options["max_tokens"]
        generation = {"messages": messages, "max_output_tokens": budget,
                      "temperature": options.get("temperature")}
    messages = generation["messages"]
    if ([message["role"] for message in messages] != ["system", "user"]
            or any(not isinstance(message["content"], str) for message in messages)):
        raise ValueError("Expected a candidate-generation system/user prompt in call logs")
    settings = {"config": config, "system_prompt": messages[0]["content"],
                "max_output_tokens": generation["max_output_tokens"],
                "temperature": generation["temperature"]}
    # Validate saved configuration without looking up or exposing any key.
    ModelConfig(**config)
    return settings, messages[1]["content"]


def _signature(template: dict[str, Any]) -> str:
    template = copy.deepcopy(template)
    # Transport policy can change during repair; experiment settings cannot.
    template["config"].pop("timeout_seconds", None)
    template["config"].pop("max_retries", None)
    return json.dumps(template, sort_keys=True, ensure_ascii=False)


def _validate_results(records: list[dict[str, Any]], path: Path) -> None:
    ids = set()
    for row in records:
        candidate_id = row.get("candidate_id")
        result = row.get("generation_result", {})
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in ids:
            raise ValueError(f"Missing or duplicate candidate_id in {path}")
        ids.add(candidate_id)
        if result.get("status") not in {"success", "failed"}:
            raise ValueError(f"Invalid generation status for {candidate_id} in {path}")
        if result["status"] == "success" and (
                not isinstance(result.get("text"), str) or not result["text"].strip()
                or result.get("error") is not None):
            raise ValueError(f"Invalid successful result for {candidate_id} in {path}")
        if result["status"] == "failed" and (
                result.get("text") is not None or not isinstance(result.get("error"), str)):
            raise ValueError(f"Invalid failed result for {candidate_id} in {path}")


def _plan(source: Path, output: Path, input_path: Path, items: dict[str, dict[str, Any]],
          selected: list[str] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = sorted(source.glob("results_*.jsonl"))
    if not paths:
        raise ValueError(f"No results_<alias>.jsonl found in {source}")
    manifest = {"version": 1, "source_dir": str(source), "source_sha256": {},
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()}
    plans = {}
    for results_path in paths:
        alias = results_path.stem.removeprefix("results_")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", alias):
            raise ValueError(f"Invalid model alias: {alias}")
        calls_path = source / f"calls_{alias}.jsonl"
        for path in (results_path, calls_path):
            manifest["source_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        original = _read_jsonl(results_path)
        _validate_results(original, results_path)
        original_calls = _read_jsonl(calls_path)
        target = output / results_path.name
        records = _read_jsonl(target) if target.exists() else copy.deepcopy(original)
        _validate_results(records, target)
        if len(records) != len(original):
            raise ValueError(f"Repaired result count differs from source: {target}")
        for before, after in zip(original, records):
            if any(before[key] != after.get(key) for key in
                   ("candidate_id", "product_id", "run_id", "sample_index")):
                raise ValueError(f"Candidate identities/order differ from source: {target}")
            if before["generation_result"]["status"] == "success" and before != after:
                raise ValueError(f"An original successful result was changed in {target}")
        target_calls = output / calls_path.name
        calls = _read_jsonl(target_calls) if target_calls.exists() else original_calls
        if calls[:len(original_calls)] != original_calls:
            raise ValueError(f"Original call history was changed in {target_calls}")
        by_trace = {}
        candidates = {row["candidate_id"]: row for row in original}
        for call in calls:
            trace = call.get("trace_id")
            if not trace or trace in by_trace:
                raise ValueError(f"Missing/duplicate trace_id in {target_calls}")
            if call.get("candidate_id") not in candidates or call.get("model_alias") != alias:
                raise ValueError(f"Call does not belong to this model/results: {target_calls}")
            by_trace[trace] = call

        # Recover the small interruption window between a flushed successful
        # call and the atomic replacement of the result file.
        for call in calls[len(original_calls):]:
            result = call.get("generation_result")
            if result and result.get("status") == "success":
                for row in records:
                    if (row["candidate_id"] == call["candidate_id"]
                            and row["generation_result"]["status"] == "failed"):
                        row["generation_result"] = copy.deepcopy(result)
                        row["trace_id"] = call["trace_id"]

        templates = {}
        for call in original_calls:
            saved = _template(call)
            if saved is not None:
                template, user_prompt = saved
                product_id = candidates[call["candidate_id"]]["product_id"]
                if product_id not in items or _product_prompt(items[product_id]) != user_prompt:
                    raise ValueError(f"Product input differs from saved prompt: {product_id}")
                templates[_signature(template)] = template
        pending = []
        for index, row in enumerate(records):
            trace = row.get("trace_id")
            if trace not in by_trace or by_trace[trace]["candidate_id"] != row["candidate_id"]:
                raise ValueError(f"Result has no matching call trace: {row['candidate_id']}")
            if row["generation_result"]["status"] != "failed":
                continue
            if selected is not None and alias not in selected:
                continue
            saved = _template(by_trace[trace])
            if saved:
                template, saved_prompt = saved
                if _product_prompt(items[row["product_id"]]) != saved_prompt:
                    raise ValueError(f"Product input differs from saved prompt: {row['product_id']}")
            elif len(templates) == 1:
                template = next(iter(templates.values()))
            else:
                raise ValueError(f"Cannot recover unambiguous generation settings for {alias}; "
                                 "the original logs need a config/request snapshot")
            product = items.get(row["product_id"])
            if product is None:
                raise ValueError(f"Missing input ProductItem: {row['product_id']}")
            pending.append((index, template, [
                {"role": "system", "content": template["system_prompt"]},
                {"role": "user", "content": _product_prompt(product)},
            ]))
        plans[alias] = {"records": records, "pending": pending,
                        "results_path": target, "calls_path": target_calls,
                        "original_calls_path": calls_path}
    if selected is not None and (len(set(selected)) != len(selected)
                                 or set(selected) - plans.keys()):
        raise ValueError("--models must name unique aliases present in --source-dir")
    return plans, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("data/eval/gold/v1/product_items.jsonl"))
    parser.add_argument("--source-dir", type=Path,
                        default=Path("artifacts/eval/gold_generation/v1"))
    parser.add_argument("--output-dir", type=Path,
                        help="Default: update --source-dir in place; optional separate directory")
    parser.add_argument("--models", nargs="+", help="Only retry these aliases (default: all)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Extra attempts per failed candidate this run (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=2.0,
                        help="Initial exponential-backoff delay in seconds (default: 2)")
    parser.add_argument("--timeout-seconds", type=float, default=180.0,
                        help="Per-request timeout for repair calls (default: 180)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect planned retries without keys, API calls or file writes")
    args = parser.parse_args(argv)
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        parser.error("--retry-delay must be finite and >= 0")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    root = Path(__file__).resolve().parents[1]
    source = (root / args.source_dir).resolve()
    output = (root / (args.output_dir or source)).resolve()
    in_place = output == source
    backup = source.with_name(f"{source.name}_original")
    input_path = (root / args.input).resolve()
    if output in source.parents or output == backup:
        parser.error("--output-dir must not be a source parent or its original-backup directory")
    try:
        items_list = _load_items(input_path, None)
        items = {item["product_id"]: item for item in items_list}
        if len(items) != len(items_list):
            raise ValueError("Input must not contain duplicate product_id values")
        with nullcontext() if args.dry_run else _output_lock(output):
            original_source = backup if in_place and backup.exists() else source
            plans, manifest = _plan(original_source, output, input_path, items, args.models)
            if in_place:
                manifest["source_dir"] = str(backup)
            manifest_path = output / "retry_manifest.json"
            if manifest_path.exists():
                if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                    raise ValueError("Source/input changed since repair began; use a new --output-dir")
            elif not in_place and output.exists() and any(
                    path.name != ".retry.lock" for path in output.iterdir()):
                raise ValueError("Output directory is not an initialized repair run; use an empty directory")

            remaining = sum(len(plan["pending"]) for plan in plans.values())
            for alias, plan in plans.items():
                print(f"{alias}: {len(plan['pending'])} failed candidates to retry; "
                      f"{len(plan['records'])} total records", flush=True)
                for _, template, _ in plan["pending"][:1]:
                    print(f"  model={template['config']['model']}; "
                          f"max_output_tokens={template['max_output_tokens']}; "
                          f"temperature={template['temperature']}; timeout={args.timeout_seconds}s",
                          flush=True)
            print(f"Total to retry: {remaining}; output={output}", flush=True)
            if args.dry_run:
                return 0
            if in_place and not backup.exists() and not remaining:
                print("All candidates already succeeded; no repair or backup needed.", flush=True)
                return 0

            load_dotenv(root / ".env")
            # Validate all pending credentials before the first API call. Models
            # with no failures do not need a live server or credentials.
            for alias, plan in plans.items():
                for _, template, _ in plan["pending"]:
                    key_env = template["config"].get("api_key_env")
                    if key_env and not os.environ.get(key_env, "").strip():
                        raise ValueError(f"Set environment variable {key_env} for {alias}")
            if in_place and not backup.exists():
                _snapshot(source, backup, list(manifest["source_sha256"]))
            if not manifest_path.exists():
                _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            for plan in plans.values():
                if not plan["calls_path"].exists():
                    _atomic_write(plan["calls_path"],
                                  plan["original_calls_path"].read_text(encoding="utf-8"))
                _write_results(plan["results_path"], plan["records"])

            repaired = 0
            for alias, plan in plans.items():
                for position, (index, template, messages) in enumerate(plan["pending"], 1):
                    row = plan["records"][index]
                    print(f"{alias}: retrying {position}/{len(plan['pending'])} "
                          f"{row['product_id']}", flush=True)
                    config = replace(ModelConfig(**template["config"]),
                                     timeout_seconds=args.timeout_seconds, max_retries=0)
                    with (plan["calls_path"].open("a", encoding="utf-8") as calls_file,
                          LLMClient(config) as client):
                        def log_call(call: dict[str, Any]) -> None:
                            calls_file.write(json.dumps(call, ensure_ascii=False) + "\n")
                            calls_file.flush()
                            os.fsync(calls_file.fileno())

                        result, trace = generate_with_retries(
                            client, model_alias=alias, candidate_id=row["candidate_id"],
                            messages=messages, max_output_tokens=template["max_output_tokens"],
                            temperature=template["temperature"], max_retries=args.max_retries,
                            retry_delay=args.retry_delay, log_call=log_call,
                            retry_of_trace_id=row["trace_id"],
                        )
                    row["generation_result"] = result
                    row["trace_id"] = trace
                    _write_results(plan["results_path"], plan["records"])
                    repaired += result["status"] == "success"
                    print(f"{alias}: {row['product_id']} {result['status']}", flush=True)
            still_failed = sum(row["generation_result"]["status"] == "failed"
                               for plan in plans.values() for row in plan["records"])
            print(f"Repaired {repaired}/{remaining}; failures remaining across models: "
                  f"{still_failed}. Results/calls: {output}", flush=True)
            return 1 if still_failed else 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
