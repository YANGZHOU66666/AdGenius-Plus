"""Serve a local pairwise annotation UI backed directly by JSON/JSONL files."""

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


def _nonblank(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


class AnnotationStore:
    def __init__(self, pairs_path: Path, products_path: Path, annotations_dir: Path,
                 annotation_set_id: str = "gold_pairwise_v1",
                 protocol_version: str = "pairwise-v0.1"):
        self.pairs_path = Path(pairs_path).resolve()
        self.products_path = Path(products_path).resolve()
        self.annotations_dir = Path(annotations_dir).resolve()
        self.annotation_set_id = annotation_set_id
        self.protocol_version = protocol_version
        if not _nonblank(annotation_set_id) or not _nonblank(protocol_version):
            raise ValueError("annotation_set_id and protocol_version must be nonempty")

    def _annotator(self, value: Any) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
            raise APIError(400, "Annotator ID must be 1–64 letters, digits, underscores, or hyphens, starting with a letter or digit")
        return value

    def _paths(self, annotator: str) -> tuple[Path, Path]:
        return (self.annotations_dir / f"pairwise_annotations_{annotator}.jsonl",
                self.annotations_dir / f"pairwise_drafts_{annotator}.json")

    @contextmanager
    def _lock(self, annotator: str):
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        with (self.annotations_dir / f".pairwise_{annotator}.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _dataset(self) -> tuple[list[dict], dict[str, dict], str]:
        pairs, pair_raw = _read_jsonl(self.pairs_path)
        products, product_raw = _read_jsonl(self.products_path)
        if not pairs:
            raise APIError(422, "The candidate pairs file is empty")
        product_map = {}
        for product in products:
            identifier = product.get("product_id")
            if (not _nonblank(identifier) or identifier in product_map
                    or not isinstance(product.get("product_info"), dict)):
                raise APIError(422, "Product data contains a missing or duplicate product_id, or invalid product_info")
            product_map[identifier] = product
        pair_map, seen_candidates, visible = {}, set(), []
        for pair in pairs:
            pair_id, product_id = pair.get("pair_id"), pair.get("product_id")
            candidates = pair.get("candidates")
            if (set(pair) != {"pair_id", "product_id", "candidates"}
                    or not _nonblank(pair_id) or pair_id in pair_map
                    or not _nonblank(product_id) or product_id not in product_map
                    or not isinstance(candidates, list) or len(candidates) != 2):
                raise APIError(422, "Invalid candidate pair, duplicate pair_id, or missing product data")
            candidate_ids = set()
            for candidate in candidates:
                if (not isinstance(candidate, dict) or set(candidate) != {"candidate_id", "text"}
                        or not _nonblank(candidate.get("candidate_id"))
                        or not _nonblank(candidate.get("text"))):
                    raise APIError(422, f"Invalid candidate text: {pair_id}")
                candidate_ids.add(candidate["candidate_id"])
            if len(candidate_ids) != 2 or candidate_ids & seen_candidates:
                raise APIError(422, "Duplicate candidate IDs; check the frozen candidate pairs")
            seen_candidates.update(candidate_ids)
            pair_map[pair_id] = pair
            product = product_map[product_id]
            visible.append({**pair,
                            "product_title": _as_text(product["product_info"].get("title")),
                            "product_information": _product_prompt(product)})
        return visible, pair_map, _digest(pair_raw + b"\0" + product_raw)

    def _annotations(self, annotator: str, pair_map: dict) -> tuple[dict, bytes]:
        path, _ = self._paths(annotator)
        rows, raw = _read_jsonl(path, optional=True)
        annotations, ids = {}, set()
        fields = {"annotation_id", "judge_type", "annotation_set_id", "pair_id",
                  "annotator_id", "protocol_version", "presentation", "pairwise_result"}
        for row in rows:
            pair_id = row.get("pair_id")
            if (set(row) != fields or not _nonblank(row.get("annotation_id"))
                    or row["annotation_id"] in ids or not _nonblank(pair_id) or pair_id not in pair_map
                    or pair_id in annotations or row.get("judge_type") != "human"
                    or row.get("annotator_id") != annotator
                    or row.get("annotation_set_id") != self.annotation_set_id
                    or row.get("protocol_version") != self.protocol_version):
                raise APIError(422, f"Annotation identity, protocol, or record format mismatch: {path.name}")
            self._presentation(row.get("presentation"), pair_map[pair_id])
            result = row.get("pairwise_result")
            # Preserve historical tie records; new submissions only accept A/B.
            if (not isinstance(result, dict)
                    or set(result) != {"status", "choice", "winner_candidate_id", "reason", "error"}
                    or result.get("status") != "success" or result.get("error") is not None
                    or not isinstance(result.get("choice"), str) or result["choice"] not in {"A", "B", "tie"}
                    or not (result.get("reason") is None or isinstance(result.get("reason"), str))):
                raise APIError(422, f"Invalid annotation result: {path.name} / {pair_id}")
            winner = (None if result["choice"] == "tie"
                      else row["presentation"][result["choice"]])
            if result.get("winner_candidate_id") != winner:
                raise APIError(422, f"Annotation choice does not match the candidate ID: {pair_id}")
            annotations[pair_id] = row
            ids.add(row["annotation_id"])
        return annotations, raw

    def _drafts(self, annotator: str, pair_map: dict) -> tuple[dict, bytes]:
        _, path = self._paths(annotator)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return {}, b""
        try:
            drafts = json.loads(raw)
            if not isinstance(drafts, dict):
                raise ValueError("Drafts must be a JSON object")
            for pair_id, draft in drafts.items():
                if (pair_id not in pair_map or not isinstance(draft, dict)
                        or set(draft) != {"reason"} or not isinstance(draft["reason"], str)):
                    raise ValueError("Invalid draft record or draft not in the current dataset")
        except (ValueError, UnicodeDecodeError) as error:
            raise APIError(422, f"Invalid draft file; file left unchanged: {path.name} ({error})") from None
        return drafts, raw

    def _presentation(self, presentation: Any, pair: dict) -> None:
        expected = {candidate["candidate_id"] for candidate in pair["candidates"]}
        if (not isinstance(presentation, dict) or set(presentation) != {"A", "B"}
                or not all(isinstance(value, str) for value in presentation.values())
                or set(presentation.values()) != expected):
            raise APIError(400, "The A/B mapping must contain exactly the two distinct candidates for this pair")

    def state(self, annotator_id: str) -> dict:
        annotator = self._annotator(annotator_id)
        pairs, pair_map, dataset_revision = self._dataset()
        annotations, raw = self._annotations(annotator, pair_map)
        drafts, draft_raw = self._drafts(annotator, pair_map)
        path, draft_path = self._paths(annotator)
        return {"annotator_id": annotator, "annotation_set_id": self.annotation_set_id,
                "protocol_version": self.protocol_version, "dataset_revision": dataset_revision,
                "annotations_revision": _digest(raw), "drafts_revision": _digest(draft_raw),
                "pairs": pairs, "annotations": annotations,
                "drafts": {key: value for key, value in drafts.items() if key not in annotations},
                "paths": {"pairs": str(self.pairs_path), "products": str(self.products_path),
                          "annotations": str(path), "drafts": str(draft_path)}}

    def _context(self, payload: dict) -> tuple[str, dict, dict, dict, bytes]:
        annotator = self._annotator(payload.get("annotator_id"))
        _, pairs, revision = self._dataset()
        if payload.get("dataset_revision") != revision:
            raise APIError(409, "Product or candidate pair files have changed; load the latest version from disk first")
        pair_id = payload.get("pair_id")
        if not isinstance(pair_id, str) or pair_id not in pairs:
            raise APIError(400, "Candidate pair not found")
        annotations, raw = self._annotations(annotator, pairs)
        return annotator, pairs, pairs[pair_id], annotations, raw

    def _save_annotations(self, annotator: str, annotations: dict, pairs: dict) -> bytes:
        raw = "".join(json.dumps(annotations[pair_id], ensure_ascii=False) + "\n"
                      for pair_id in pairs if pair_id in annotations).encode("utf-8")
        _atomic_write(self._paths(annotator)[0], raw)
        return raw

    def _clear_draft(self, annotator: str, pair_id: str, pairs: dict) -> None:
        drafts, _ = self._drafts(annotator, pairs)
        if pair_id in drafts:
            del drafts[pair_id]
            _atomic_write(self._paths(annotator)[1], _json_bytes(drafts))

    def save(self, payload: dict) -> dict:
        annotator = self._annotator(payload.get("annotator_id"))
        with self._lock(annotator):
            _, pairs, pair, annotations, _ = self._context(payload)
            pair_id = pair["pair_id"]
            if "expected_annotation" not in payload or payload["expected_annotation"] != annotations.get(pair_id):
                raise APIError(409, "This annotation was changed on disk or in another page; review the latest version first")
            presentation, choice, reason = payload.get("presentation"), payload.get("choice"), payload.get("reason")
            self._presentation(presentation, pair)
            if (not isinstance(choice, str) or choice not in {"A", "B"}
                    or not (reason is None or isinstance(reason, str))):
                raise APIError(400, "Choose A or B; the reason must be text")
            self._drafts(annotator, pairs)  # Validate before touching either file.
            previous = annotations.get(pair_id)
            record = {"annotation_id": previous["annotation_id"] if previous else f"human_pw_{uuid.uuid4().hex}",
                      "judge_type": "human", "annotation_set_id": self.annotation_set_id,
                      "pair_id": pair_id, "annotator_id": annotator,
                      "protocol_version": self.protocol_version, "presentation": presentation,
                      "pairwise_result": {"status": "success", "choice": choice,
                                          "winner_candidate_id": presentation[choice],
                                          "reason": reason, "error": None}}
            annotations[pair_id] = record
            raw = self._save_annotations(annotator, annotations, pairs)
            self._clear_draft(annotator, pair_id, pairs)
            return {"annotation": record, "annotations_revision": _digest(raw)}

    def delete(self, payload: dict) -> dict:
        annotator = self._annotator(payload.get("annotator_id"))
        with self._lock(annotator):
            _, pairs, pair, annotations, _ = self._context(payload)
            pair_id = pair["pair_id"]
            if "expected_annotation" not in payload or payload["expected_annotation"] != annotations.get(pair_id):
                raise APIError(409, "This annotation has changed; load the latest version from disk first")
            self._drafts(annotator, pairs)
            annotations.pop(pair_id, None)
            raw = self._save_annotations(annotator, annotations, pairs)
            self._clear_draft(annotator, pair_id, pairs)
            return {"annotation": None, "annotations_revision": _digest(raw)}

    def save_draft(self, payload: dict) -> dict:
        annotator = self._annotator(payload.get("annotator_id"))
        with self._lock(annotator):
            _, pairs, pair, annotations, _ = self._context(payload)
            pair_id = pair["pair_id"]
            if pair_id in annotations:
                raise APIError(409, "This pair already has an annotation; reload before editing its reason")
            drafts, _ = self._drafts(annotator, pairs)
            if "expected_draft" not in payload or payload["expected_draft"] != drafts.get(pair_id):
                raise APIError(409, "This draft has changed; load the latest version from disk first")
            reason = payload.get("reason")
            if not isinstance(reason, str):
                raise APIError(400, "The reason must be text")
            if reason:
                drafts[pair_id] = {"reason": reason}
            else:
                drafts.pop(pair_id, None)
            raw = _json_bytes(drafts)
            _atomic_write(self._paths(annotator)[1], raw)
            return {"draft": drafts.get(pair_id), "drafts_revision": _digest(raw)}


def make_server(store: AnnotationStore, host: str = "127.0.0.1", port: int = 0,
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
                if url.path == "/api/config":
                    return self._json(200, {"default_annotator": default_annotator})
                if url.path == "/favicon.ico":
                    return self._send(204, b"", "image/x-icon")
                filenames = {"/": ("index.html", "text/html"),
                             "/index.html": ("index.html", "text/html"),
                             "/style.css": ("style.css", "text/css"),
                             "/app.js": ("app.js", "text/javascript")}
                if url.path not in filenames:
                    raise APIError(404, "Page not found")
                filename, content_type = filenames[url.path]
                self._send(200, (assets / filename).read_bytes(), f"{content_type}; charset=utf-8")
            except APIError as error:
                self._json(error.status, {"error": str(error)})
            except OSError as error:
                self._json(500, {"error": f"Unable to read or write local files: {error}"})

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
                path = urlsplit(self.path).path
                operations = {("PUT", "/api/annotation"): store.save,
                              ("DELETE", "/api/annotation"): store.delete,
                              ("PUT", "/api/draft"): store.save_draft}
                operation = operations.get((method, path))
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
    parser.add_argument("--pairs", type=Path, default=ROOT / "data/eval/gold/candidate_pairs.jsonl")
    parser.add_argument("--products", type=Path, default=ROOT / "data/eval/gold/product_items.jsonl")
    parser.add_argument("--annotations-dir", type=Path, default=ROOT / "data/eval/gold")
    parser.add_argument("--annotator", default="zhou")
    parser.add_argument("--annotation-set-id", default="gold_pairwise_v2")
    parser.add_argument("--protocol-version", default="pairwise-v0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    store = AnnotationStore(args.pairs, args.products, args.annotations_dir,
                            args.annotation_set_id, args.protocol_version)
    try:
        initial = store.state(args.annotator)
        server = make_server(store, port=args.port, default_annotator=args.annotator)
    except (APIError, OSError) as error:
        parser.error(str(error))
    print(f"Pairwise annotation: http://127.0.0.1:{server.server_port}/?annotator_id={args.annotator}", flush=True)
    print(f"{len(initial['pairs'])} pairs; current annotator completed {len(initial['annotations'])}", flush=True)
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
