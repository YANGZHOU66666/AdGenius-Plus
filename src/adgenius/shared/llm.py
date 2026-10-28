"""Small, synchronous, text-only client for OpenAI-compatible and Claude APIs.

No evaluation logic, automatic JSON repair, streaming, or implicit paid calls.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal

import httpx

Provider = Literal["openai_compatible", "anthropic", "mock"]
_CORE_FIELDS = {"model", "messages", "system", "max_tokens", "max_completion_tokens",
                "temperature", "stream", "n"}
_RETRYABLE = {408, 429, 500, 502, 503, 504, 529}


class LLMError(RuntimeError):
    """API/transport error with metadata suitable for a failed-run record."""

    def __init__(self, message: str, *, attempts: int = 0,
                 status_code: int | None = None, request_id: str | None = None,
                 raw_response: Any = None, may_have_completed: bool = False):
        super().__init__(message)
        self.attempts = attempts
        self.status_code = status_code
        self.request_id = request_id
        self.raw_response = raw_response
        self.may_have_completed = may_have_completed


@dataclass(frozen=True)
class ModelConfig:
    provider: Provider
    model: str
    base_url: str = ""
    base_url_env: str = ""
    api_key_env: str = ""
    timeout_seconds: float = 60.0
    max_retries: int = 2
    request_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider not in {"openai_compatible", "anthropic", "mock"}:
            raise ValueError(f"Unknown provider: {self.provider}")
        if not self.model:
            raise ValueError("model must not be empty")
        if self.timeout_seconds <= 0 or self.max_retries < 0:
            raise ValueError("timeout_seconds must be positive; max_retries must be >= 0")
        if self.provider != "mock":
            url = httpx.URL(self.base_url)
            if url.scheme != "https" or not url.host or url.userinfo or url.query or url.fragment:
                raise ValueError("base_url must be an HTTPS API root without credentials or query")
            if not self.api_key_env:
                raise ValueError("api_key_env is required")

    @classmethod
    def from_toml(cls, path: str | Path, name: str) -> ModelConfig:
        with Path(path).open("rb") as handle:
            models = tomllib.load(handle).get("models", {})
        if name not in models:
            raise ValueError(f"Unknown model config: {name!r}; choices: {', '.join(models)}")
        values = copy.deepcopy(models[name])
        base_url_env = values.get("base_url_env")
        if base_url_env:
            override = os.environ.get(base_url_env, "").strip()
            if override:
                values["base_url"] = override
        return cls(**values)

    def snapshot(self) -> dict[str, Any]:
        """Configuration safe to persist; never includes an API key value."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "base_url_env": self.base_url_env,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "request_options": copy.deepcopy(self.request_options),
        }


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    requested_model: str
    model: str
    usage: TokenUsage
    finish_reason: str | None
    request_id: str | None
    latency_seconds: float
    attempts: int
    content_missing: bool
    endpoint: str
    config: dict[str, Any]
    request: dict[str, Any]
    raw_response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable experiment record; no authorization headers."""
        return asdict(self)


class LLMClient:
    """Reuse as a context manager so HTTP connections are closed explicitly."""

    def __init__(self, config: ModelConfig, *, transport: httpx.BaseTransport | None = None):
        self.config = config
        self._http = httpx.Client(timeout=config.timeout_seconds, transport=transport,
                                  follow_redirects=False)

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def generate(self, messages: Sequence[Mapping[str, str]], *,
                 max_output_tokens: int = 512, temperature: float | None = None,
                 json_mode: bool = False,
                 provider_options: Mapping[str, Any] | None = None) -> LLMResponse:
        """Return one response. Unsupported provider options surface as API errors.

        json_mode requests JSON-object output on compatible APIs; it does not
        validate the text, insert a prompt, or retry invalid JSON. For Claude,
        use its native structured-output options through provider_options.
        """
        normalized = _validate_messages(messages)
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer")
        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        options = copy.deepcopy(self.config.request_options)
        options.update(copy.deepcopy(dict(provider_options or {})))
        if _CORE_FIELDS & options.keys():
            raise ValueError("provider_options must not override core request fields")
        if json_mode and "response_format" in options:
            raise ValueError("Use either json_mode or a native response_format")

        body: dict[str, Any] = {"model": self.config.model, "messages": normalized,
                                "max_tokens": max_output_tokens, "stream": False}
        if temperature is not None:
            body["temperature"] = temperature
        if self.config.provider == "anthropic":
            if json_mode:
                raise ValueError("Claude json_mode is not mapped; use native output_config")
            systems = [m["content"] for m in normalized if m["role"] == "system"]
            body["messages"] = [m for m in normalized if m["role"] != "system"]
            if systems:
                body["system"] = "\n\n".join(systems)
        elif json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(options)
        # Fail on non-JSON options (including NaN) before attempting the network.
        json.dumps(body, allow_nan=False)
        started = time.perf_counter()
        if self.config.provider == "mock":
            text = '{"mock": true}' if json_mode else 'Mock response: no API call was made.'
            return LLMResponse(text, "mock", self.config.model, self.config.model,
                               TokenUsage(), "stop", None, time.perf_counter() - started,
                               0, False, "", self.config.snapshot(), body,
                               {"mock": True, "text": text})

        key = os.environ.get(self.config.api_key_env, "").strip()
        if not key:
            raise ValueError(f"Set environment variable {self.config.api_key_env}")
        headers = {"content-type": "application/json"}
        if self.config.provider == "anthropic":
            headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
            endpoint = "messages"
        else:
            headers["authorization"] = f"Bearer {key}"
            endpoint = "chat/completions"
        url = f"{self.config.base_url.rstrip('/')}/{endpoint}"
        response, attempts = self._post(url, headers, body)
        request_id = response.headers.get("request-id") or response.headers.get("x-request-id")
        try:
            raw = response.json()
            text, model, usage, reason, response_id, content_missing = self._parse(raw)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise LLMError("API returned a malformed response envelope", attempts=attempts,
                           status_code=response.status_code, request_id=request_id,
                           raw_response=_safe_error_body(response, key)) from None
        return LLMResponse(text, self.config.provider, self.config.model, model, usage,
                           reason, request_id or response_id, time.perf_counter() - started,
                           attempts, content_missing, url, self.config.snapshot(), body, raw)

    def _post(self, url: str, headers: dict[str, str],
              body: dict[str, Any]) -> tuple[httpx.Response, int]:
        for attempt in range(1, self.config.max_retries + 2):
            try:
                response = self._http.post(url, headers=headers, json=body)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                ambiguous = isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout,
                                              httpx.ReadError, httpx.WriteError,
                                              httpx.RemoteProtocolError))
                if ambiguous or attempt > self.config.max_retries:
                    raise LLMError(f"LLM transport failed ({type(exc).__name__})",
                                   attempts=attempt, may_have_completed=ambiguous) from None
                _backoff(attempt)
                continue
            if response.is_success:
                return response, attempt
            if response.status_code in _RETRYABLE and attempt <= self.config.max_retries:
                _backoff(attempt, response.headers.get("retry-after"))
                continue
            # Keep a short, redacted provider error for diagnosing wrong regions/models.
            secret = headers.get("x-api-key") or headers["authorization"].removeprefix("Bearer ")
            error_body = _safe_error_body(response, secret)
            raise LLMError(f"LLM API returned HTTP {response.status_code}", attempts=attempt,
                           status_code=response.status_code,
                           request_id=response.headers.get("request-id") or
                           response.headers.get("x-request-id"), raw_response=error_body)
        raise AssertionError("unreachable")

    def _parse(self, raw: dict[str, Any]) -> tuple[str, str, TokenUsage, str | None, str | None, bool]:
        usage = raw.get("usage") or {}
        if self.config.provider == "anthropic":
            blocks = raw["content"]
            text = "".join(b["text"] for b in blocks if b.get("type") == "text")
            content_missing = not bool(text)
            # Claude input_tokens excludes cache reads/writes; normalize to total input.
            uncached = usage.get("input_tokens")
            cache_read = usage.get("cache_read_input_tokens")
            cache_write = usage.get("cache_creation_input_tokens")
            inputs = (uncached + (cache_read or 0) + (cache_write or 0)
                      if uncached is not None else None)
            outputs = usage.get("output_tokens")
            tokens = TokenUsage(inputs, outputs, _total(inputs, outputs), None,
                                cache_read, cache_write)
            reason = raw.get("stop_reason")
        else:
            choice = raw["choices"][0]
            content = choice["message"].get("content")
            if content is None:  # e.g. refusal, or output budget exhausted by thinking
                text = ""
                content_missing = True
            elif isinstance(content, str):
                text = content
                content_missing = not bool(text)
            else:
                raise ValueError("Only text responses are supported")
            inputs = usage.get("prompt_tokens")
            outputs = usage.get("completion_tokens")
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            if cached is None:
                cached = usage.get("prompt_cache_hit_tokens")
            tokens = TokenUsage(inputs, outputs, usage.get("total_tokens", _total(inputs, outputs)),
                                (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                                cached)
            reason = choice.get("finish_reason")
        return text, raw.get("model", self.config.model), tokens, reason, raw.get("id"), content_missing


def _validate_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen_conversation = False
    for message in messages:
        if set(message) != {"role", "content"}:
            raise ValueError("Each message must contain only role and content")
        role, content = message["role"], message["content"]
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("Only text system/user/assistant messages are supported")
        if role == "system" and seen_conversation:
            raise ValueError("System messages must come before the conversation")
        seen_conversation |= role != "system"
        result.append({"role": role, "content": content})
    if not result or not any(m["role"] == "user" for m in result):
        raise ValueError("At least one user message is required")
    return result


def _total(inputs: int | None, outputs: int | None) -> int | None:
    return inputs + outputs if inputs is not None and outputs is not None else None


def _safe_error_body(response: httpx.Response, secret: str) -> Any:
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text[:4000]
    serialized = json.dumps(body, ensure_ascii=False) if not isinstance(body, str) else body
    if secret:
        serialized = serialized.replace(secret, "[REDACTED]")
    try:
        return json.loads(serialized)
    except json.JSONDecodeError:
        return serialized[:4000]


def _backoff(attempt: int, retry_after: str | None = None) -> None:
    delay = min(2 ** (attempt - 1), 30) + random.uniform(0, 0.25)
    if retry_after:
        try:
            delay = max(delay, float(retry_after))
        except ValueError:
            try:
                delay = max(delay, (parsedate_to_datetime(retry_after) -
                                    datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError):
                pass
    time.sleep(min(delay, 60))
