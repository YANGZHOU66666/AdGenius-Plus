"""Compare ad text quality with two independent, blind AB/BA LLM requests."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
import tomllib
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import jsonschema
from dotenv import load_dotenv

from adgenius.eval.pairwise_metrics import decision, summarize
from adgenius.shared.llm import LLMClient, LLMError, ModelConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = REPO_ROOT / "metadata" / "schemas"
DEFAULT_PAIRS = REPO_ROOT / "data/eval/gold/candidate_pairs.jsonl"
DEFAULT_HUMAN = REPO_ROOT / "data/eval/gold/pairwise_annotations_zhou.jsonl"
ORDERS = ("AB", "BA")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _no_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(text: str | bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON constant: {value}")
    return json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=reject_constant)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = _strict_json(line)
            if not isinstance(value, dict):
                raise ValueError("Expected an object")
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        rows.append(value)
    return rows


def _atomic_write(path: Path, raw: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                         + "\n").encode("utf-8"))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write(path, b"".join(_json_bytes(row) + b"\n" for row in rows))


@contextmanager
def _run_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another evaluator is using {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _validator(name: str) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(_strict_json((SCHEMA_DIR / name).read_bytes()))


def _mapping(pair: dict[str, Any]) -> dict[str, str]:
    return dict(zip("AB", (candidate["candidate_id"] for candidate in pair["candidates"])))


def _load_inputs(pairs_path: Path, human_path: Path | None,
                 limit: int | None, human_protocol: str):
    pairs = _read_jsonl(pairs_path)
    pair_validator = _validator("candidate_pair.schema.json")
    pair_map, candidate_map, seen_pairs = {}, {}, set()
    for pair in pairs:
        pair_validator.validate(pair)
        pair_id, product_id = pair["pair_id"], pair["product_id"]
        ids = list(_mapping(pair).values())
        if pair_id in pair_map or len(set(ids)) != 2:
            raise ValueError(f"Duplicate pair ID or identical candidates: {pair_id}")
        identity = frozenset(ids)
        if identity in seen_pairs:
            raise ValueError(f"Duplicate candidate pair (including reversed order): {pair_id}")
        seen_pairs.add(identity)
        for candidate in pair["candidates"]:
            content = (product_id, candidate["text"])
            previous = candidate_map.setdefault(candidate["candidate_id"], content)
            if previous != content:
                raise ValueError(f"Candidate ID reused with different product/text: {candidate['candidate_id']}")
        pair_map[pair_id] = pair
    selected = pairs[:limit] if limit else pairs
    if not selected:
        raise ValueError("No CandidatePair records to evaluate")
    humans = {}
    if human_path is not None:
        validator = _validator("pairwise_result.schema.json")
        for row in _read_jsonl(human_path):
            validator.validate(row)
            if row["judge_type"] != "human" or row["protocol_version"] != human_protocol:
                raise ValueError("Human file must contain human labels matching "
                                 "judge.human_annotation_protocol_version "
                                 "(defaults to judge.protocol_version)")
            pair_id = row["pair_id"]
            if pair_id not in pair_map or pair_id in humans:
                raise ValueError(f"Unknown/duplicate human pair ID: {pair_id}; use one annotator file")
            result, presentation = row["pairwise_result"], row["presentation"]
            if set(presentation.values()) != set(_mapping(pair_map[pair_id]).values()):
                raise ValueError(f"Human presentation does not match pair {pair_id}")
            if result["status"] == "success":
                if (result["choice"] not in ("A", "B") or
                        result["winner_candidate_id"] != presentation[result["choice"]]):
                    raise ValueError(f"Expected a forced A/B human winner consistent with presentation: {pair_id}")
            humans[pair_id] = row
    selected_ids = {pair["pair_id"] for pair in selected}
    return selected, {key: row for key, row in humans.items() if key in selected_ids}


def _load_settings(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        judge = tomllib.load(handle).get("judge")
    required = {"model_alias", "protocol_version", "input_mode", "system_prompt", "user_prompt",
                "max_output_tokens", "timeout_seconds", "max_retries",
                "retry_delay", "provider_options"}
    optional = {"human_annotation_protocol_version", "temperature"}
    if (not isinstance(judge, dict) or not required.issubset(judge)
            or set(judge) - required - optional):
        raise ValueError(f"[judge] requires: {', '.join(sorted(required))}; "
                         "optional: human_annotation_protocol_version, temperature")
    judge.setdefault("human_annotation_protocol_version", judge["protocol_version"])
    judge.setdefault("temperature", None)
    for key in ("model_alias", "protocol_version", "human_annotation_protocol_version",
                "system_prompt", "user_prompt"):
        if not isinstance(judge[key], str) or not judge[key].strip():
            raise ValueError(f"judge.{key} must be a nonblank string")
    if judge["input_mode"] != "ad_text_only":
        raise ValueError("Pairwise judge supports input_mode = 'ad_text_only'; "
                         "product-grounded factual checks belong to the rubric judge")
    for key, minimum in (("max_output_tokens", 1), ("max_retries", 0)):
        if type(judge[key]) is not int or judge[key] < minimum:
            raise ValueError(f"judge.{key} must be an integer >= {minimum}")
    for key in ("timeout_seconds", "retry_delay"):
        value = judge[key]
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"judge.{key} must be finite")
    temperature = judge["temperature"]
    if temperature is not None and (type(temperature) not in (int, float)
                                    or not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise ValueError("judge.temperature must be finite and between 0 and 2 when specified")
    if judge["timeout_seconds"] <= 0 or judge["retry_delay"] < 0:
        raise ValueError("Invalid temperature, timeout_seconds, or retry_delay")
    if not isinstance(judge["provider_options"], dict):
        raise ValueError("judge.provider_options must be a table")
    # The compatible wrapper rejects these before sending; validate before creating a run.
    blocked = {"model", "messages", "system", "max_tokens", "max_completion_tokens",
               "temperature", "stream", "n", "response_format"}
    if blocked.intersection(judge["provider_options"]):
        raise ValueError("judge.provider_options overrides core request fields")
    _json_bytes(judge)
    return judge


def _messages(system: str, template: str, ad_1: str, ad_2: str):
    try:
        user = template.format(ad_1=ad_1, ad_2=ad_2)
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError(f"Invalid pairwise prompt template: {error}") from error
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _judge_request(config: ModelConfig, settings: dict[str, Any], messages):
    options = copy.deepcopy(settings["provider_options"])
    json_mode = True
    if config.provider == "anthropic":
        if settings["temperature"] is not None:
            raise ValueError("Omit judge.temperature for Claude judges; Haiku 5.5 rejects sampling parameters")
        combined = copy.deepcopy(config.request_options)
        combined.update(options)
        options = combined
        if {"temperature", "top_p", "top_k", "enable_thinking", "response_format"}.intersection(options):
            raise ValueError("Claude judge options contain unsupported sampling or provider fields")
        output_config = options.setdefault("output_config", {})
        if not isinstance(output_config, dict) or "format" in output_config:
            raise ValueError("Claude output_config may set effort; its format is supplied by the pairwise judge")
        output_config["format"] = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"choice": {"type": "string", "enum": ["1", "2"]},
                               "reason": {"type": "string"}},
                "required": ["choice", "reason"],
                "additionalProperties": False,
            },
        }
        json_mode = False
    return {"messages": messages, "temperature": settings["temperature"],
            "max_output_tokens": settings["max_output_tokens"],
            "json_mode": json_mode, "provider_options": options}


def _parse_choice(text: str) -> tuple[str, str]:
    value = _strict_json(text)
    if not isinstance(value, dict) or set(value) != {"choice", "reason"}:
        raise ValueError("Expected exactly choice and reason")
    if value["choice"] not in ("1", "2") or not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("Expected choice string 1/2 and a nonblank reason")
    return value["choice"], value["reason"].strip()


def _failure(code: str, trace: str | None = None) -> dict[str, Any]:
    return {"status": "failed", "choice": None, "winner_candidate_id": None,
            "reason": None, "error": code, "trace_id": trace}


def _normalize(response, slots: dict[str, str], mapping: dict[str, str], trace: str):
    reason = (response.finish_reason or "").lower()
    if reason in {"length", "max_tokens", "max_output_tokens"}:
        return _failure("output_too_long", trace)
    raw_choices = response.raw_response.get("choices") or []
    refused = any((item.get("message") or {}).get("refusal") for item in raw_choices)
    if refused or reason in {"content_filter", "refusal", "safety", "blocked"}:
        return _failure("refusal", trace)
    if not response.text.strip():
        return _failure("empty_output", trace)
    try:
        slot, explanation = _parse_choice(response.text)
    except ValueError:
        return _failure("invalid_output", trace)
    winner = slots[slot]
    return {"status": "success", "choice": "A" if winner == mapping["A"] else "B",
            "winner_candidate_id": winner, "reason": explanation,
            "error": None, "trace_id": trace}


def _error_code(error: LLMError) -> str:
    if error.status_code in {401, 403}:
        return "auth_error"
    if error.status_code == 429:
        return "rate_limit"
    if error.status_code == 408 or "timeout" in str(error).lower():
        return "timeout"
    return "network_error" if error.status_code is None else "provider_error"


def _retry_delay(error: LLMError, attempt: int, initial: float) -> float:
    delay = min(initial * 2 ** min(attempt - 1, 20) + random.uniform(0, 0.25), 60.0)
    retry_after = error.retry_after
    if retry_after:
        try:
            requested = float(retry_after)
        except ValueError:
            try:
                requested = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                return delay
        if math.isfinite(requested):
            delay = min(max(delay, requested), 60.0)
    return delay


def _call_order(client: LLMClient, *, pair: dict[str, Any], order: str, messages,
                slots: dict[str, str], run_id: str, settings: dict[str, Any],
                calls: list[dict[str, Any]], calls_path: Path) -> dict[str, Any]:
    previous = next((row["trace_id"] for row in reversed(calls)
                     if row["pair_id"] == pair["pair_id"] and row["prompt_order"] == order), None)
    for attempt in range(1, settings["max_retries"] + 2):
        trace = f"call_pw_{uuid.uuid4().hex}"
        started = time.perf_counter()
        row = {"trace_id": trace, "judge_run_id": run_id, "pair_id": pair["pair_id"],
               "prompt_order": order, "created_at": _now(), "attempt_in_invocation": attempt,
               "retry_of_trace_id": previous, "displayed_candidates": slots,
               "request": _judge_request(client.config, settings, messages)}
        retry_error = None
        print(f"{pair['pair_id']} {order}: attempt {attempt}/{settings['max_retries'] + 1}", flush=True)
        try:
            response = client.generate(**row["request"])
            row["response"] = response.to_dict()
            result = _normalize(response, slots, _mapping(pair), trace)
        except LLMError as error:
            result = _failure(_error_code(error), trace)
            row["error_detail"] = {"message": str(error), "status_code": error.status_code,
                                   "raw_response": error.raw_response, "request_id": error.request_id,
                                   "may_have_completed": error.may_have_completed,
                                   "retry_after": error.retry_after}
            if error.status_code is None or error.status_code in {408, 429} or (
                    error.status_code is not None and 500 <= error.status_code <= 599):
                retry_error = error
        row.update(status=result["status"], normalized_result=result,
                   latency_seconds=time.perf_counter() - started)
        calls.append(row)
        # Atomic journal snapshots make even the final line safe to recover.
        _write_rows(calls_path, calls)
        print(f"{pair['pair_id']} {order}: {result['status']}"
              + (f" ({result['error']})" if result["error"] else ""), flush=True)
        if retry_error is None or attempt > settings["max_retries"]:
            return result
        delay = _retry_delay(retry_error, attempt, settings["retry_delay"])
        print(f"Retrying in {delay:.1f}s", flush=True)
        time.sleep(delay)
        previous = trace
    raise AssertionError("Unreachable retry state")


def _result_record(pair, run_id, protocol, by_order):
    return {"result_id": f"pw_{_hash(_json_bytes([run_id, pair['pair_id']]))[:24]}",
            "judge_type": "model", "pair_id": pair["pair_id"], "judge_run_id": run_id,
            "protocol_version": protocol, "candidate_mapping": _mapping(pair),
            "pairwise_result": [{"prompt_order": order, **by_order[order]} for order in ORDERS]}


def _validate_order(result, mapping):
    if result["status"] == "success":
        choice = result["choice"]
        if choice not in ("A", "B") or result["winner_candidate_id"] != mapping[choice]:
            raise ValueError("Saved judge choice/winner is inconsistent with candidate mapping")
        if not isinstance(result["reason"], str) or not result["reason"].strip():
            raise ValueError("Successful judge call is missing its reason")
    if not result["trace_id"]:
        raise ValueError("Saved judge result must reference a call trace")


def _load_saved(output: Path, calls_path: Path, pairs, run_id, protocol):
    calls = _read_jsonl(calls_path) if calls_path.exists() else []
    results = _read_jsonl(output / "results.jsonl") if (output / "results.jsonl").exists() else []
    pair_map = {pair["pair_id"]: pair for pair in pairs}
    validator = _validator("pairwise_result.schema.json")
    traces, cached, result_map = {}, {}, {}
    for call in calls:
        pair_id, order, trace = call["pair_id"], call["prompt_order"], call["trace_id"]
        if pair_id not in pair_map or order not in ORDERS or trace in traces or call["judge_run_id"] != run_id:
            raise ValueError("Call journal contains unknown/duplicate identities or another run")
        pair = pair_map[pair_id]
        mapping = _mapping(pair)
        expected_slots = dict(zip(("1", "2"), (mapping[key] for key in order)))
        result = call["normalized_result"]
        validator.evolve(schema=validator.schema["$defs"]["model_call_result"]).validate(
            {"prompt_order": order, **result})
        if call["displayed_candidates"] != expected_slots or call["status"] != result["status"] or result["trace_id"] != trace:
            raise ValueError("Call journal has an invalid presentation or normalized result")
        _validate_order(result, mapping)
        if result["status"] == "success":
            key = (pair_id, order)
            if key in cached:
                raise ValueError("More than one successful call for an order; refusing ambiguous resume")
            cached[key] = result
        traces[trace] = call
    for row in results:
        validator.validate(row)
        pair_id = row["pair_id"]
        if (row["judge_type"] != "model" or pair_id not in pair_map or pair_id in result_map
                or row["judge_run_id"] != run_id or row["protocol_version"] != protocol
                or row["candidate_mapping"] != _mapping(pair_map[pair_id])):
            raise ValueError("Saved results do not match this run and its pairs")
        for item in row["pairwise_result"]:
            if item["error"] == "not_run" and item["trace_id"] is None:
                continue
            _validate_order(item, row["candidate_mapping"])
            trace = traces.get(item["trace_id"])
            normalized = {key: value for key, value in item.items() if key != "prompt_order"}
            if (trace is None or trace["pair_id"] != pair_id or
                    trace["prompt_order"] != item["prompt_order"] or trace["normalized_result"] != normalized):
                raise ValueError("Saved result does not match its call journal entry")
        result_map[pair_id] = row
    return calls, result_map, cached


def _fingerprint(settings, config, paths, pairs, human_path):
    files = {"pairs": paths["pairs"],
             "system_prompt": paths["system"], "user_prompt": paths["user"],
             "eval_config": paths["eval_config"]}
    if human_path:
        files["human_annotations"] = human_path
    provenance_files = [Path(__file__), Path(__file__).with_name("pairwise_metrics.py"),
                        REPO_ROOT / "src/adgenius/shared/llm.py",
                        *[SCHEMA_DIR / name for name in ("candidate_pair.schema.json",
                          "pairwise_result.schema.json")]]
    return {"settings": settings, "model_config": config.snapshot(),
            "input_files": {name: {"path": str(path), "sha256": _hash(path.read_bytes())}
                            for name, path in files.items()},
            "code_and_schema_sha256": {str(path.relative_to(REPO_ROOT)): _hash(path.read_bytes())
                                       for path in provenance_files},
            "selected_pair_ids": [pair["pair_id"] for pair in pairs],
            "orders": list(ORDERS), "display_labels": ["1", "2"]}


def _execute(output, calls_path, manifest, pairs, humans, config, system, template):
    run_id, settings = manifest["run_id"], manifest["specification"]["settings"]
    protocol = settings["protocol_version"]
    calls, results, cached = _load_saved(output, calls_path, pairs, run_id, protocol)
    validator = _validator("pairwise_result.schema.json")

    def save_derived():
        rows = [results[pair["pair_id"]] for pair in pairs if pair["pair_id"] in results]
        _write_rows(output / "results.jsonl", rows)
        _write_rows(output / "decisions.jsonl", [decision(row) for row in rows])
        metrics = summarize(rows, humans, calls, len(pairs))
        human_protocols = sorted({row["protocol_version"] for row in humans.values()})
        agreement = metrics["human_agreement"]
        agreement.update(judge_protocol_version=protocol,
                         annotation_protocol_versions=human_protocols)
        if any(version != protocol for version in human_protocols):
            agreement["comparison_note"] = (
                "Calibration comparison against existing labels from an earlier protocol. "
                "Those labels have not been relabeled as ad-text-only judgments."
            )
        _write_json(output / "metrics.json", metrics)
        return metrics

    manifest.update(status="running", updated_at=_now())
    _write_json(output / "manifest.json", manifest)
    try:
        with LLMClient(config) as client:
            for index, pair in enumerate(pairs, 1):
                pair_id, mapping = pair["pair_id"], _mapping(pair)
                by_order = {order: cached.get((pair_id, order), _failure("not_run")) for order in ORDERS}
                for order in ORDERS:
                    if by_order[order]["status"] == "success":
                        continue
                    print(f"[{index}/{len(pairs)}] {config.model}", flush=True)
                    slots = dict(zip(("1", "2"), (mapping[key] for key in order)))
                    texts = {item["candidate_id"]: item["text"] for item in pair["candidates"]}
                    messages = _messages(system, template, texts[slots["1"]], texts[slots["2"]])
                    by_order[order] = _call_order(client, pair=pair, order=order, messages=messages,
                                                 slots=slots, run_id=run_id, settings=settings,
                                                 calls=calls, calls_path=calls_path)
                    record = _result_record(pair, run_id, protocol, by_order)
                    validator.validate(record)
                    results[pair_id] = record
                    _write_rows(output / "results.jsonl", list(results.values()))
                # This also restores a result missing after a logged success and a crash.
                results[pair_id] = _result_record(pair, run_id, protocol, by_order)
                metrics = save_derived()
                print(f"[{index}/{len(pairs)}] {pair_id}: {decision(results[pair_id])['decision_status']}", flush=True)
    except BaseException:
        save_derived()
        manifest.update(status="interrupted", updated_at=_now())
        _write_json(output / "manifest.json", manifest)
        raise
    metrics = save_derived()
    manifest.update(status="completed" if not metrics["incomplete_pair_count"] else "completed_with_failures",
                    updated_at=_now())
    _write_json(output / "manifest.json", manifest)
    return {"run_id": run_id, "output_dir": str(output), "calls_file": str(calls_path),
            "status": manifest["status"], "metrics": metrics}


def run(args: argparse.Namespace) -> dict[str, Any]:
    env_file = Path(args.env_file).resolve() if args.env_file else REPO_ROOT / ".env"
    if args.env_file and not env_file.is_file():
        raise ValueError(f"Environment file not found: {env_file}")
    load_dotenv(env_file, override=False)
    paths = {name: Path(getattr(args, name)).resolve() for name in ("pairs", "eval_config")}
    settings = _load_settings(paths["eval_config"])
    if args.model_alias:
        settings["model_alias"] = args.model_alias
    for name in ("system", "user"):
        paths[name] = (REPO_ROOT / settings[f"{name}_prompt"]).resolve()
    system, template = paths["system"].read_text(encoding="utf-8"), paths["user"].read_text(encoding="utf-8")
    if not system.strip() or not all("{" + key + "}" in template for key in ("ad_1", "ad_2")):
        raise ValueError("Prompts must be nonempty and include ad_1 and ad_2")
    _messages(system, template, "first", "second")
    human_path = Path(args.human_annotations).resolve() if args.human_annotations else None
    if human_path is None and not args.no_human_annotations and paths["pairs"] == DEFAULT_PAIRS:
        human_path = DEFAULT_HUMAN
    if args.no_human_annotations:
        human_path = None
    pairs, humans = _load_inputs(paths["pairs"], human_path, args.limit,
                                  settings["human_annotation_protocol_version"])
    config = ModelConfig.from_toml(Path(args.model_config).resolve(), settings["model_alias"])
    config = replace(config, max_retries=0, timeout_seconds=settings["timeout_seconds"])
    if config.provider not in {"openai_compatible", "openai_compatible_local", "anthropic"}:
        raise ValueError("Pairwise judge supports OpenAI-compatible and Anthropic models")
    _judge_request(config, settings, _messages(system, template, "first", "second"))
    specification = _fingerprint(settings, config, paths, pairs, human_path)
    fingerprint = _hash(_json_bytes(specification))
    run_id = args.run_id
    if not run_id and args.output_dir and (Path(args.output_dir) / "manifest.json").is_file():
        run_id = _strict_json((Path(args.output_dir) / "manifest.json").read_bytes())["run_id"]
    run_id = run_id or f"pairwise_{settings['model_alias']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if not isinstance(run_id, str) or re.fullmatch(r"[a-zA-Z0-9_-]+", run_id) is None:
        raise ValueError("run_id may contain only letters, digits, underscores, and hyphens")
    output = Path(args.output_dir).resolve() if args.output_dir else REPO_ROOT / "results/pairwise" / run_id
    calls_dir = Path(args.calls_dir).resolve() if args.calls_dir else REPO_ROOT / "artifacts/pairwise" / run_id
    if calls_dir != output and (output in calls_dir.parents or calls_dir in output.parents):
        raise ValueError("Result and call directories must be separate non-nested directories")
    calls_path = calls_dir / "calls.jsonl"
    if args.dry_run:
        return {"dry_run": True, "run_id": run_id, "model": config.model, "pair_count": len(pairs),
                "protocol_version": settings["protocol_version"], "input_mode": settings["input_mode"],
                "human_annotation_protocol_version": settings["human_annotation_protocol_version"],
                "human_label_count": len(humans), "initial_judge_calls": 2 * len(pairs),
                "maximum_attempts_per_order": settings["max_retries"] + 1,
                "output_dir": str(output), "calls_file": str(calls_path),
                "config_fingerprint": fingerprint}
    if config.api_key_env and not os.environ.get(config.api_key_env, "").strip():
        raise ValueError(f"Set {config.api_key_env}, or pass --env-file /path/to/.env")
    with _run_lock(output), (_run_lock(calls_dir) if calls_dir != output else nullcontext()):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = _strict_json(manifest_path.read_bytes())
            if (manifest["run_id"] != run_id or manifest["config_fingerprint"] != fingerprint or
                    _hash(_json_bytes(manifest["specification"])) != fingerprint):
                raise ValueError("Run inputs/config/code changed; use another --run-id/output directory")
            if manifest.get("calls_file") != str(calls_path):
                raise ValueError("Call journal location changed; resume with the original --calls-dir")
        else:
            existing = [path for path in output.iterdir() if path.name != ".run.lock"]
            if existing:
                raise ValueError("Output directory has files without a run manifest; choose an empty directory")
            if calls_path.exists():
                raise ValueError("Call journal already exists without a matching result manifest; choose a new run")
            manifest = {"run_id": run_id, "created_at": _now(), "status": "created",
                        "config_fingerprint": fingerprint, "specification": specification,
                        "calls_file": str(calls_path)}
            _write_json(manifest_path, manifest)
        return _execute(output, calls_path, manifest, pairs, humans, config, system, template)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-config", type=Path, default=REPO_ROOT / "configs/eval/pairwise.toml")
    parser.add_argument("--model-config", type=Path, default=REPO_ROOT / "configs/models.toml")
    parser.add_argument("--model-alias", help="Alias in models.toml; defaults to qwen")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--human-annotations", type=Path, help="One annotator's file; default gold uses Zhou's labels")
    group.add_argument("--no-human-annotations", action="store_true")
    parser.add_argument("--env-file", type=Path, help="Defaults to the repository .env; does not override exported variables")
    parser.add_argument("--output-dir", type=Path,
                        help="Standard results directory; default results/pairwise/<run_id>")
    parser.add_argument("--calls-dir", type=Path,
                        help="Local HTTP journal directory; default artifacts/pairwise/<run_id>")
    parser.add_argument("--run-id", help="Reuse the same ID to resume failed/missing orders")
    parser.add_argument("--limit", type=int, help="First N pairs for a separate pilot run")
    parser.add_argument("--dry-run", action="store_true", help="Validate without network calls, API key, or output writes")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        outcome = run(args)
    except (OSError, ValueError, KeyError, jsonschema.ValidationError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("Interrupted. Reuse the same run ID and arguments to resume.", flush=True)
        return 130
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    return 1 if outcome.get("status") == "completed_with_failures" else 0


if __name__ == "__main__":
    raise SystemExit(main())
