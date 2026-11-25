"""Serve a local single-ad rubric annotation UI backed directly by JSONL files.

Items come from the frozen rubric sample (data/eval/gold/rubric_sample.jsonl). The page shows one ad
with its product information and hides the generator. Labels are RubricResult v1 human records written
to rubric_annotations_<annotator>.jsonl beside the sample.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from generate_candidates import _as_text, _product_prompt  # noqa: E402

GOLD = ROOT / "data" / "eval" / "gold"
CRITERIA = ("factual_support", "concrete_product_info")


class APIError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_jsonl(path: Path, *, optional: bool = False) -> tuple[list[dict], bytes]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        if optional:
            return [], b""
        raise APIError(422, f"Input file not found: {path}") from None
    rows = []
    try:
        for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {number} is not a JSON object")
            rows.append(row)
    except (UnicodeDecodeError, ValueError) as error:
        raise APIError(422, f"Invalid file format; file left unchanged: {path.name} ({error})") from None
    return rows, raw


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


class RubricStore:
    def __init__(self, gold_dir: Path, annotations_dir: Path | None = None,
                 criteria: tuple[str, ...] = ("factual_support",),
                 annotation_set_id: str = "gold_rubric_v1", protocol_version: str = "rubric-v0.2"):
        self.gold_dir = Path(gold_dir).resolve()
        self.annotations_dir = Path(annotations_dir or gold_dir).resolve()
        if not criteria or len(set(criteria)) != len(criteria) or not set(criteria) <= set(CRITERIA):
            raise ValueError(f"criteria must be distinct values from {CRITERIA}")
        self.criteria = tuple(criteria)
        self.annotation_set_id = annotation_set_id
        self.protocol_version = protocol_version

    def _annotator(self, value: Any) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
            raise APIError(400, "Annotator ID must be 1–64 letters, digits, underscores, or hyphens")
        return value

    def _path(self, annotator: str) -> Path:
        return self.annotations_dir / f"rubric_annotations_{annotator}.jsonl"

    @contextmanager
    def _lock(self, annotator: str):
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        with (self.annotations_dir / f".rubric_{annotator}.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _dataset(self) -> tuple[list[dict], dict[str, dict], str]:
        sample, sample_raw = _read_jsonl(self.gold_dir / "rubric_sample.jsonl")
        products, product_raw = _read_jsonl(self.gold_dir / "product_items.jsonl")
        product_map = {p.get("product_id"): p for p in products}
        raws, texts = [sample_raw, product_raw], {}
        for generator in sorted({row.get("generator") for row in sample}):
            if not isinstance(generator, str) or not re.fullmatch(r"[A-Za-z0-9_]+", generator):
                raise APIError(422, "Invalid generator in the rubric sample")
            results, raw = _read_jsonl(self.gold_dir / f"results_{generator}.jsonl")
            raws.append(raw)
            for result in results:
                texts[result.get("candidate_id")] = (result.get("product_id"),
                                                     (result.get("generation_result") or {}).get("text"))
        items, by_candidate = [], {}
        for row in sample:
            candidate_id, product_id = row.get("candidate_id"), row.get("product_id")
            known = texts.get(candidate_id)
            if (candidate_id in by_candidate or known is None or known[0] != product_id
                    or product_id not in product_map or not isinstance(known[1], str) or not known[1].strip()):
                raise APIError(422, f"Rubric sample row does not match the gold data: {row.get('item_id')}")
            product = product_map[product_id]
            # The generator is deliberately not sent to the page.
            item = {"item_id": row["item_id"], "candidate_id": candidate_id, "product_id": product_id,
                    "ad": known[1], "product_title": _as_text(product["product_info"].get("title")),
                    "product_information": _product_prompt(product)}
            items.append(item)
            by_candidate[candidate_id] = item
        if not items:
            raise APIError(422, "The rubric sample is empty")
        return items, by_candidate, _digest(b"\0".join(raws))

    def _annotations(self, annotator: str, items: dict) -> tuple[dict, bytes]:
        path = self._path(annotator)
        rows, raw = _read_jsonl(path, optional=True)
        fields = {"annotation_id", "judge_type", "annotation_set_id", "candidate_id", "product_id",
                  "annotator_id", "protocol_version", "rubric_result"}
        annotations = {}
        for row in rows:
            candidate_id = row.get("candidate_id")
            result = row.get("rubric_result")
            if (set(row) != fields or candidate_id not in items or candidate_id in annotations
                    or row.get("product_id") != items[candidate_id]["product_id"]
                    or row.get("judge_type") != "human" or row.get("annotator_id") != annotator
                    or row.get("annotation_set_id") != self.annotation_set_id
                    or row.get("protocol_version") != self.protocol_version
                    or not isinstance(result, dict) or result.get("status") != "success"
                    or result.get("error") is not None
                    or [c.get("criterion_id") for c in result.get("criteria", [])] != list(self.criteria)
                    or not all(isinstance(c.get("value"), bool) for c in result["criteria"])):
                raise APIError(422, f"Annotation record does not match this sample or protocol: {path.name}")
            annotations[candidate_id] = row
        return annotations, raw

    def state(self, annotator_id: str) -> dict:
        annotator = self._annotator(annotator_id)
        items, by_candidate, revision = self._dataset()
        annotations, raw = self._annotations(annotator, by_candidate)
        return {"annotator_id": annotator, "annotation_set_id": self.annotation_set_id,
                "protocol_version": self.protocol_version, "criteria": list(self.criteria),
                "dataset_revision": revision, "annotations_revision": _digest(raw),
                "items": items, "annotations": annotations,
                "paths": {"sample": str(self.gold_dir / "rubric_sample.jsonl"),
                          "annotations": str(self._path(annotator))}}

    def _context(self, payload: dict) -> tuple[str, list[dict], dict, dict]:
        annotator = self._annotator(payload.get("annotator_id"))
        items, by_candidate, revision = self._dataset()
        if payload.get("dataset_revision") != revision:
            raise APIError(409, "The sample or gold files changed on disk; reload the page")
        candidate_id = payload.get("candidate_id")
        if candidate_id not in by_candidate:
            raise APIError(400, "Item not found")
        annotations, _ = self._annotations(annotator, by_candidate)
        if "expected_annotation" not in payload or payload["expected_annotation"] != annotations.get(candidate_id):
            raise APIError(409, "This label changed on disk or in another tab; reload the page first")
        return annotator, items, by_candidate[candidate_id], annotations

    def _write(self, annotator: str, items: list[dict], annotations: dict) -> bytes:
        raw = "".join(json.dumps(annotations[item["candidate_id"]], ensure_ascii=False) + "\n"
                      for item in items if item["candidate_id"] in annotations).encode("utf-8")
        _atomic_write(self._path(annotator), raw)
        return raw

    def save(self, payload: dict) -> dict:
        annotator = self._annotator(payload.get("annotator_id"))
        with self._lock(annotator):
            _, items, item, annotations = self._context(payload)
            values, reasons = payload.get("values"), payload.get("reasons") or {}
            if (not isinstance(values, dict) or set(values) != set(self.criteria)
                    or not all(isinstance(v, bool) for v in values.values())
                    or not isinstance(reasons, dict) or not set(reasons) <= set(self.criteria)
                    or not all(r is None or isinstance(r, str) for r in reasons.values())):
                raise APIError(400, "Give true or false for every criterion; reasons must be text")
            previous = annotations.get(item["candidate_id"])
            criteria = [{"criterion_id": c, "value": values[c],
                         "reason": (reasons.get(c) or "").strip() or None} for c in self.criteria]
            record = {"annotation_id": previous["annotation_id"] if previous else f"human_rb_{uuid.uuid4().hex}",
                      "judge_type": "human", "annotation_set_id": self.annotation_set_id,
                      "candidate_id": item["candidate_id"], "product_id": item["product_id"],
                      "annotator_id": annotator, "protocol_version": self.protocol_version,
                      "rubric_result": {"status": "success", "criteria": criteria, "error": None}}
            annotations[item["candidate_id"]] = record
            raw = self._write(annotator, items, annotations)
            return {"annotation": record, "annotations_revision": _digest(raw)}

    def delete(self, payload: dict) -> dict:
        annotator = self._annotator(payload.get("annotator_id"))
        with self._lock(annotator):
            _, items, item, annotations = self._context(payload)
            annotations.pop(item["candidate_id"], None)
            raw = self._write(annotator, items, annotations)
            return {"annotation": None, "annotations_revision": _digest(raw)}


def make_server(store: RubricStore, host: str = "127.0.0.1", port: int = 0,
                default_annotator: str = "zhou") -> ThreadingHTTPServer:
    assets = Path(__file__).with_suffix("")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            if len(args) > 1 and str(args[1]).startswith(("4", "5")):
                super().log_message(format, *args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, value: Any) -> None:
            self._send(code, _json_bytes(value), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            try:
                url = urlsplit(self.path)
                if url.path == "/api/state":
                    annotator = parse_qs(url.query).get("annotator_id", [default_annotator])[0]
                    return self._json(200, store.state(annotator))
                if url.path == "/favicon.ico":
                    return self._send(204, b"", "image/x-icon")
                files = {"/": ("index.html", "text/html"), "/index.html": ("index.html", "text/html"),
                         "/style.css": ("style.css", "text/css"), "/app.js": ("app.js", "text/javascript")}
                if url.path not in files:
                    raise APIError(404, "Page not found")
                name, content_type = files[url.path]
                self._send(200, (assets / name).read_bytes(), f"{content_type}; charset=utf-8")
            except APIError as error:
                self._json(error.status, {"error": str(error)})
            except OSError as error:
                self._json(500, {"error": f"Unable to read local files: {error}"})

        def _mutate(self, method: str) -> None:
            try:
                origin = self.headers.get("Origin")
                if origin and origin != f"http://{self.headers.get('Host')}":
                    raise APIError(403, "Save from the local annotation page")
                if self.headers.get_content_type() != "application/json":
                    raise APIError(415, "The request must use application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise APIError(400, "The request body is empty or too large")
                try:
                    payload = json.loads(self.rfile.read(length))
                    if not isinstance(payload, dict):
                        raise ValueError()
                except (ValueError, UnicodeDecodeError):
                    raise APIError(400, "The request must be a JSON object") from None
                operation = {("PUT", "/api/annotation"): store.save,
                             ("DELETE", "/api/annotation"): store.delete}.get((method, urlsplit(self.path).path))
                if operation is None:
                    raise APIError(404, "Operation not found")
                self._json(200, operation(payload))
            except APIError as error:
                self._json(error.status, {"error": str(error)})
            except (OSError, ValueError) as error:
                self._json(500, {"error": f"Unable to save local files: {error}"})

        def do_PUT(self) -> None:
            self._mutate("PUT")

        def do_DELETE(self) -> None:
            self._mutate("DELETE")

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, default=GOLD)
    parser.add_argument("--annotations-dir", type=Path, help="defaults to --gold-dir")
    parser.add_argument("--annotator", default="zhou")
    parser.add_argument("--criteria", nargs="+", default=["factual_support"], choices=CRITERIA)
    parser.add_argument("--annotation-set-id", default="gold_rubric_v1")
    parser.add_argument("--protocol-version", default="rubric-v0.2")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    store = RubricStore(args.gold_dir, args.annotations_dir, tuple(args.criteria),
                        args.annotation_set_id, args.protocol_version)
    try:
        initial = store.state(args.annotator)
        server = make_server(store, port=args.port, default_annotator=args.annotator)
    except (APIError, OSError) as error:
        parser.error(str(error))
    print(f"Rubric annotation: http://127.0.0.1:{server.server_port}/?annotator_id={args.annotator}", flush=True)
    print(f"{len(initial['items'])} items; {len(initial['annotations'])} labeled", flush=True)
    print(f"Annotations: {initial['paths']['annotations']}", flush=True)
    print("Keep this terminal running; press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
