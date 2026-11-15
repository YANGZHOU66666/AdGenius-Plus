"""Generate one candidate advertisement per model for ProductItem JSONL.

The script uses the shared LLM wrapper and writes one GenerationResult JSONL
and one calls JSONL per model. It does not create pairs or call a judge.
Defaults read the 100 frozen gold products and write four model outputs to
artifacts/generation/gold/ using the current short-copy prompt.
Defaults select four live models;
explicitly select ``--models mock`` for a dry run without API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from adgenius.shared.llm import LLMClient, LLMError, ModelConfig
from adgenius.shared.product import _as_text, render_product_information as _product_prompt


DEFAULT_MODELS = ("qwen_local", "gemma_31b", "glm53_flash", "longcat")
DEFAULT_INPUT = Path("data/eval/gold/product_items.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/generation/gold")
SYSTEM_PROMPT = """You write concise English promotional copy for e-commerce product cards.

Write a clear, engaging blurb that helps shoppers quickly understand
the product and what makes it relevant to their needs.

Content and composition:
- You do not need to cover all supplied features or details.
  Select the information you consider most important to shoppers
  and most suitable for a short product ad. Leave out less relevant
  details rather than trying to fit everything into the copy.
- Organize the details into a coherent, natural message. Combine related
  information smoothly and avoid a list-like sequence of specifications.
- Give each detail a clear purpose in the copy. Remove repetition,
  filler, and generic praise.
- Use a short, natural product name or product type where appropriate.

Style and length:
- Use clear, natural, consumer-facing language with a modest promotional tone.
- Keep the copy concise and informative. Use as many words as needed
  to communicate the product clearly, without unnecessary wording.
- Prefer fewer than 25 English words. Do not exceed 35 words.
- Use one or two short sentences.

Factual accuracy:
- Use only information supported by the supplied product information.
- Preserve quantities, units, compatibility, and the relationship
  between each attribute and the item it describes.
- Do not let simplification or compression change the original meaning.
- Do not invent benefits, specifications, guarantees, comparisons,
  rankings, ratings, or customer outcomes.

Before returning the copy, check that it reads smoothly, contains useful
product information, and has no unnecessary wording or unsupported claims.

Return only the promotional copy, without a heading, bullets, quotation
marks, word count, or explanation."""

def _error_code(error: BaseException) -> str:
    if isinstance(error, LLMError):
        if error.status_code == 401 or error.status_code == 403:
            return "auth_error"
        if error.status_code == 408:
            return "timeout"
        if error.status_code == 429:
            return "rate_limit"
        if error.status_code is not None:
            return "provider_error"
        if error.may_have_completed or "timeout" in str(error).lower():
            return "timeout"
        return "network_error"
    if isinstance(error, (TimeoutError,)):
        return "timeout"
    return "provider_error"


def generate_with_retries(
    client: LLMClient, *, model_alias: str, candidate_id: str,
    messages: list[dict[str, str]], max_output_tokens: int,
    temperature: float | None, log_call: Callable[[dict[str, Any]], None],
    max_retries: int = 3, retry_delay: float = 2.0,
    retry_of_trace_id: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Generate one candidate, logging each attempt before retrying.

    The caller disables the client's internal retries so this one bounded loop
    owns both retries and their logs. Retrying a read/write timeout can repeat
    work that the server completed; those attempts retain may_have_completed.
    The GenerationResult itself still contains just the final outcome.
    """
    if client.config.max_retries != 0:
        raise ValueError("generate_with_retries requires client max_retries=0")
    if max_retries < 0 or not math.isfinite(retry_delay) or retry_delay < 0:
        raise ValueError("max_retries and retry_delay must be nonnegative and finite")

    previous_trace = retry_of_trace_id
    for attempt in range(1, max_retries + 2):
        trace_id = f"call_{uuid.uuid4().hex}"
        print(f"{model_alias}: {candidate_id} attempt {attempt}/{max_retries + 1}",
              flush=True)
        call_record: dict[str, Any] = {
            "trace_id": trace_id,
            "candidate_id": candidate_id,
            "model_alias": model_alias,
            "attempt": attempt,
            "retry_of_trace_id": previous_trace,
            "retry_policy": {"max_retries": max_retries,
                             "initial_delay_seconds": retry_delay},
            "config": client.config.snapshot(),
            "generation_request": {
                "messages": [dict(message) for message in messages],
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            },
        }
        retryable = False
        retry_after = None
        try:
            response = client.generate(
                messages, max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            text = response.text.strip()
            if response.finish_reason in {"length", "max_tokens", "model_length", "MAX_TOKENS"}:
                error_code = "output_too_long"
            elif (response.provider == "google_generative"
                  and response.finish_reason != "STOP"):
                error_code = "invalid_output"
            elif not text:
                error_code = "empty_output"
                retryable = True
            else:
                error_code = None
            outcome = {
                "status": "failed" if error_code else "success",
                "text": None if error_code else text,
                "error": error_code,
            }
            call_record.update(status=outcome["status"], error_code=error_code,
                               response=response.to_dict())
        except (LLMError, ValueError, TimeoutError) as error:
            error_code = _error_code(error)
            outcome = {"status": "failed", "text": None, "error": error_code}
            status_code = getattr(error, "status_code", None)
            retry_after = getattr(error, "retry_after", None)
            retryable = isinstance(error, TimeoutError) or (
                isinstance(error, LLMError) and (
                    status_code is None or status_code in {408, 429}
                    or 500 <= status_code < 600
                )
            )
            call_record.update(
                status="failed", error=str(error), error_code=error_code,
                attempts=getattr(error, "attempts", 0), status_code=status_code,
                request_id=getattr(error, "request_id", None),
                raw_response=getattr(error, "raw_response", None),
                may_have_completed=getattr(error, "may_have_completed", False),
                retry_after=retry_after,
            )
        # Persist a completed attempt before any delay or next API call.
        call_record["generation_result"] = dict(outcome)
        log_call(call_record)
        if outcome["status"] == "success" or not retryable or attempt > max_retries:
            return outcome, trace_id
        delay = min(retry_delay * 2 ** min(attempt - 1, 30)
                    + random.uniform(0, 0.25), 60.0)
        if retry_after:
            try:
                requested_delay = float(retry_after)
            except ValueError:
                try:
                    requested_delay = (parsedate_to_datetime(retry_after)
                                       - datetime.now(timezone.utc)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    requested_delay = 0.0
            if math.isfinite(requested_delay):
                delay = min(max(delay, requested_delay), 60.0)
        print(f"{model_alias}: {candidate_id} failed ({outcome['error']}); "
              f"retrying in {delay:.1f}s", flush=True)
        time.sleep(delay)
        previous_trace = trace_id
    raise AssertionError("unreachable")


def _load_items(path: Path, limit: int | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            if not isinstance(item.get("product_id"), str) or not item["product_id"].strip():
                raise ValueError(f"Missing product_id at {path}:{line_number}")
            if not isinstance(item.get("product_info"), dict):
                raise ValueError(f"Missing product_info at {path}:{line_number}")
            items.append(item)
            if limit is not None and len(items) >= limit:
                break
    if not items:
        raise ValueError(f"No ProductItem records found in {path}")
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"ProductItem JSONL input (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=("Directory for results_<alias>.jsonl and calls_<alias>.jsonl; "
              f"no overwrite (default: {DEFAULT_OUTPUT_DIR})"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Config aliases (default: qwen_local gemma_31b glm53_flash longcat); use mock for a dry run",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/models.toml"))
    parser.add_argument("--limit", type=int, help="Only process the first N products")
    parser.add_argument("--max-output-tokens", type=int, default=2048,
                        help="Output token budget, including reasoning where applicable (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Additional attempts per transient failure (default: 3)")
    parser.add_argument("--retry-delay", type=float, default=2.0,
                        help="Initial exponential-backoff delay in seconds (default: 2)")
    parser.add_argument("--timeout-seconds", type=float,
                        help="Override the per-model request timeout")
    parser.add_argument("--run-stamp", help="Stable suffix for run IDs; defaults to UTC time")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    if not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if args.max_retries < 0:
        parser.error("--max-retries must be nonnegative")
    if not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        parser.error("--retry-delay must be nonnegative and finite")
    if args.timeout_seconds is not None and (
            not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0):
        parser.error("--timeout-seconds must be positive and finite")

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    input_path = args.input if args.input.is_absolute() else repo_root / args.input
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    items = _load_items(input_path, args.limit)
    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicate aliases")
    if len({item["product_id"] for item in items}) != len(items):
        parser.error("Input must not contain duplicate product_id values")
    output_paths = {}
    for alias in args.models:
        if not alias or any(character not in
                            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                            for character in alias):
            parser.error("Model aliases must use only letters, digits, underscores, or hyphens")
        output_paths[alias] = (output_dir / f"results_{alias}.jsonl",
                               output_dir / f"calls_{alias}.jsonl")
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"Output directory is not a directory: {output_dir}")
    for path in (path for paths in output_paths.values() for path in paths):
        if path.exists():
            parser.error(f"Output already exists: {path}")
    # Check every config and required secret before any model incurs work.
    configs = {}
    for model_alias in args.models:
        config = ModelConfig.from_toml(config_path, model_alias)
        config = replace(config, max_retries=0,
                         timeout_seconds=args.timeout_seconds or config.timeout_seconds)
        if config.api_key_env and not os.environ.get(config.api_key_env, "").strip():
            parser.error(f"Set environment variable {config.api_key_env} for {model_alias}")
        configs[model_alias] = config

    stamp = args.run_stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_alias in args.models:
        config = configs[model_alias]
        run_id = f"gen_{model_alias}_{stamp}"
        results_path, calls_path = output_paths[model_alias]
        successful = 0
        with (results_path.open("x", encoding="utf-8") as results_file,
              calls_path.open("x", encoding="utf-8") as calls_file,
              LLMClient(config) as client):
            def log_call(record: dict[str, Any]) -> None:
                calls_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                calls_file.flush()

            for item in items:
                candidate_key = json.dumps([run_id, item["product_id"], 0],
                                           ensure_ascii=False).encode("utf-8")
                candidate_id = "cand_" + hashlib.sha256(candidate_key).hexdigest()[:24]
                outcome, trace_id = generate_with_retries(
                    client, model_alias=model_alias, candidate_id=candidate_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _product_prompt(item)},
                    ],
                    max_output_tokens=args.max_output_tokens,
                    temperature=args.temperature, max_retries=args.max_retries,
                    retry_delay=args.retry_delay, log_call=log_call,
                )
                result_record = {
                    "candidate_id": candidate_id,
                    "product_id": item["product_id"],
                    "run_id": run_id,
                    "sample_index": 0,
                    "generation_result": outcome,
                    "trace_id": trace_id,
                }
                results_file.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                results_file.flush()
                successful += outcome["status"] == "success"
                print(f"{model_alias}: {item['product_id']} {outcome['status']}", flush=True)
        print(f"{model_alias}: {successful}/{len(items)} successful; "
              f"results={results_path}; calls={calls_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
