"""vLLM-compatible teacher endpoint for TRL that forwards scoring to Tinker's Qwen3.5-397B-A17B.

TRL's ServerDistillationTrainer talks to the teacher with vLLM's API:
  GET /health, GET /v1/models,
  POST /v1/completions {prompt: [[token ids]], max_tokens: 1, temperature, prompt_logprobs: k}
This proxy answers those requests with `SamplingClient.compute_logprobs` on Tinker.

Limits, by design:
- Tinker returns only the logprob of the actual token at each position, not a top-k list. That is all
  TRL's reverse-KL path (beta=1, loss_top_k=1) reads. Do not use this proxy with beta < 1.
- Tinker scores at temperature 1. Requests with another temperature are rejected.
- Only the completion is scored (`prompt_logprobs_last_n`). It is located after the last assistant
  header `<|im_start|>assistant\n<think>\n\n</think>\n\n`, so Tinker can reuse the cached prompt prefix.
- A hard cap on tokens sent (--max-tokens) protects the budget. Usage is written to tinker_usage.json.

The API key is read from the environment, a one-time .tinker_key file next to this script, or the repository .env
(TINKER_API_KEY), and is never printed.
"""

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HERE = Path(__file__).resolve().parent
ENV_FILE = HERE.parents[1] / ".env"  # repository .env when run from scripts/opd; absent on the VM
USAGE_FILE = HERE / "tinker_usage.json"
TEACHER = "Qwen/Qwen3.5-397B-A17B"
ASSISTANT_HEADER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


KEY_FILE = HERE / ".tinker_key"


def api_key() -> str:
    """On Colab the key arrives as a one-line, owner-only file that is deleted as soon as it is read."""
    if os.environ.get("TINKER_API_KEY"):
        return os.environ["TINKER_API_KEY"]
    if KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
        KEY_FILE.unlink()
        if key:
            return key
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("TINKER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("TINKER_API_KEY not found")


class Usage:
    def __init__(self, cap: int, path: Path = USAGE_FILE):
        self.cap = cap
        self.path = path
        self.lock = threading.Lock()
        self.state = json.loads(path.read_text()) if path.exists() else \
            {"requests": 0, "sequences": 0, "tokens_sent": 0, "tokens_scored": 0}

    def reserve(self, tokens: int) -> bool:
        with self.lock:
            if self.state["tokens_sent"] + tokens > self.cap:
                return False
            self.state["tokens_sent"] += tokens
            return True

    def record(self, sequences: int, scored: int):
        with self.lock:
            self.state["requests"] += 1
            self.state["sequences"] += sequences
            self.state["tokens_scored"] += scored
            self.path.write_text(json.dumps(self.state, indent=1))


def completion_start(ids: list[int], header: list[int]) -> int | None:
    """Index of the first completion token: right after the FIRST assistant header.

    Prompts are single-turn (system, user, assistant), so the first header ends the prompt. Searching for the last
    one was wrong when the student's own completion contained a header: positions before it were not scored and
    TRL failed with "missing actual-token logprobs".
    """
    for start in range(0, len(ids) - len(header) + 1):
        if ids[start:start + len(header)] == header:
            return start + len(header)
    return None


def request_one(client, ids, last_n, mode):
    from tinker import types

    model_input = types.ModelInput.from_ints(ids)
    if mode == "actual":
        return client.compute_logprobs(model_input, prompt_logprobs_last_n=last_n)
    # top1: actual-token logprobs plus the teacher's own top-1 token at every position
    return client.sample(model_input, num_samples=1,
                         sampling_params=types.SamplingParams(max_tokens=1, temperature=1.0),
                         include_prompt_logprobs=True, topk_prompt_logprobs=1, prompt_logprobs_last_n=last_n)


def unpack(result, mode):
    return (result, None) if mode == "actual" else (result.prompt_logprobs, result.topk_prompt_logprobs)


def make_handler(client, header, usage, mode, tokenizer):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {})
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": TEACHER, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/completions":
                self._send(404, {"error": "not found"})
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if abs(float(request.get("temperature", 1.0)) - 1.0) > 1e-6:
                self._send(400, {"error": "Tinker scores at temperature 1 only"})
                return
            prompts = request["prompt"]
            if prompts and isinstance(prompts[0], int):
                prompts = [prompts]
            if not usage.reserve(sum(len(p) for p in prompts)):
                self._send(429, {"error": f"token cap reached ({usage.cap} tokens sent)"})
                return
            jobs = []
            for ids in prompts:
                start = completion_start(ids, header)
                last_n = len(ids) - start if start is not None and start < len(ids) else None
                jobs.append((ids, start, request_one(client, ids, last_n, mode)))
            choices, scored = [], 0
            try:
                for i, (ids, start, future) in enumerate(jobs):
                    logprobs, topk = unpack(future.result(), mode)
                    begin = start if start is not None else 1
                    missing = [pos for pos in range(max(begin, 1), len(ids))
                               if pos >= len(logprobs) or logprobs[pos] is None]
                    if missing:
                        # Never return holes in the completion: rescore the whole sequence and log what happened.
                        print(f"[tinker-proxy] FALLBACK full scoring: len={len(ids)} start={start} "
                              f"missing={len(missing)} first_missing={missing[0]} completion="
                              f"{tokenizer.decode(ids[begin:])!r}", flush=True)
                        if not usage.reserve(len(ids)):
                            self._send(429, {"error": f"token cap reached ({usage.cap} tokens sent)"})
                            return
                        logprobs, topk = unpack(request_one(client, ids, None, mode).result(), mode)
                    entries = [None]
                    for pos in range(1, len(ids)):
                        lp = logprobs[pos]
                        if lp is None:
                            entries.append(None)
                            continue
                        entry = {str(ids[pos]): {"logprob": lp, "rank": 1, "decoded_token": ""}}
                        if topk is not None and topk[pos]:
                            top_token, top_lp = topk[pos][0]
                            if int(top_token) != ids[pos]:
                                entry[str(ids[pos])]["rank"] = 2  # actual token is not the teacher's first choice
                                entry[str(int(top_token))] = {"logprob": float(top_lp), "rank": 1, "decoded_token": ""}
                        entries.append(entry)
                    scored += sum(lp is not None for lp in logprobs)
                    choices.append({"index": i, "text": "", "finish_reason": "length", "prompt_logprobs": entries})
            except Exception as error:  # noqa: BLE001 - surface Tinker errors to TRL as a 500
                print(f"[tinker-proxy] Tinker error: {type(error).__name__}: {error}", flush=True)
                self._send(500, {"error": f"{type(error).__name__}"})
                return
            usage.record(len(prompts), scored)
            print(f"[tinker-proxy] scored {len(prompts)} seqs, {scored} tokens; totals {usage.state}", flush=True)
            self._send(200, {"object": "text_completion", "model": TEACHER, "choices": choices})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--max-tokens", type=int, default=300_000, help="hard cap on tokens sent to Tinker")
    parser.add_argument("--usage-file", type=Path, default=USAGE_FILE)
    parser.add_argument("--mode", choices=["actual", "top1"], default="actual",
                        help="actual: compute_logprobs (beta=1); top1: sample() with the teacher top-1 token (beta<1)")
    args = parser.parse_args()
    import tinker
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")  # same tokenizer as the 397B teacher
    header = tokenizer(ASSISTANT_HEADER, add_special_tokens=False)["input_ids"]
    client = tinker.ServiceClient(api_key=api_key()).create_sampling_client(base_model=TEACHER).result()
    usage = Usage(args.max_tokens, args.usage_file)
    print(f"[tinker-proxy] {TEACHER} mode={args.mode} ready on 127.0.0.1:{args.port}, cap {args.max_tokens} tokens, "
          f"usage so far {usage.state}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(client, header, usage, args.mode, tokenizer)).serve_forever()


if __name__ == "__main__":
    main()
