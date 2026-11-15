"""Offline tests for disk-backed human annotation and lossless concurrent edits."""

import copy
import importlib.util
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "pairwise_annotation.py"
SPEC = importlib.util.spec_from_file_location("pairwise_annotation", SCRIPT)
annotation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(annotation)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


class PairwiseAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pairs_path = self.root / "candidate_pairs.jsonl"
        self.products_path = self.root / "product_items.jsonl"
        self.annotations_dir = self.root / "annotations"
        self.products = [
            {"product_id": f"gold_product_{index}", "source_id": f"ASIN-{index}",
             "product_info": {"title": f"Metal stapler {index}",
                              "features": ["Staples up to 20 sheets", "Metal body"],
                              "details": {"Color": "Blue", "Best Sellers Rank": "#1"},
                              "price": 99.0, "average_rating": 5.0}}
            for index in range(2)
        ]
        self.pairs = [
            {"pair_id": f"pair_{index}", "product_id": f"gold_product_{index}",
             "candidates": [
                 {"candidate_id": f"cand_{index}_first", "text": "A metal stapler."},
                 {"candidate_id": f"cand_{index}_second",
                  "text": "Staple up to 20 sheets with a metal stapler."},
             ]}
            for index in range(2)
        ]
        write_rows(self.pairs_path, self.pairs)
        write_rows(self.products_path, self.products)
        self.source_bytes = {path: path.read_bytes()
                             for path in (self.pairs_path, self.products_path)}
        self.store = self.new_store()

    def new_store(self):
        return annotation.AnnotationStore(self.pairs_path, self.products_path,
                                          self.annotations_dir)

    def result_path(self, annotator="zhou"):
        return self.annotations_dir / f"pairwise_annotations_{annotator}.jsonl"

    def payload(self, pair_index=0, annotator="zhou", choice="A", reverse=False):
        state = self.store.state(annotator)
        pair = self.pairs[pair_index]
        candidates = pair["candidates"][::-1] if reverse else pair["candidates"]
        return {
            "annotator_id": annotator,
            "pair_id": pair["pair_id"],
            "dataset_revision": state["dataset_revision"],
            "expected_annotation": state["annotations"].get(pair["pair_id"]),
            "presentation": {"A": candidates[0]["candidate_id"],
                             "B": candidates[1]["candidate_id"]},
            "choice": choice,
            "reason": "具体且有资料支持。",
        }

    def assert_human_record(self, row):
        self.assertEqual(set(row), {
            "annotation_id", "judge_type", "annotation_set_id", "pair_id",
            "annotator_id", "protocol_version", "presentation", "pairwise_result",
        })
        self.assertEqual(row["judge_type"], "human")
        self.assertTrue(row["annotation_id"].strip())
        self.assertEqual(set(row["presentation"]), {"A", "B"})
        result = row["pairwise_result"]
        self.assertEqual(set(result), {"status", "choice", "winner_candidate_id",
                                       "reason", "error"})
        self.assertEqual(result["status"], "success")
        self.assertIsNone(result["error"])
        if result["choice"] == "tie":
            self.assertIsNone(result["winner_candidate_id"])
        else:
            self.assertEqual(result["winner_candidate_id"],
                             row["presentation"][result["choice"]])

    def assert_sources_unchanged(self):
        self.assertEqual({path: path.read_bytes() for path in self.source_bytes},
                         self.source_bytes)

    def test_browsing_never_creates_a_label_and_renders_only_generation_information(self):
        state = self.store.state("zhou")
        self.assertEqual(state["annotations"], {})
        self.assertEqual(len(state["pairs"]), 2)
        pair = state["pairs"][0]
        self.assertEqual(pair["pair_id"], "pair_0")
        self.assertEqual(pair["product_title"], "Metal stapler 0")
        self.assertIn("Staples up to 20 sheets", pair["product_information"])
        self.assertIn("Blue", pair["product_information"])
        self.assertNotIn("Best Sellers Rank", pair["product_information"])
        self.assertNotIn("average_rating", pair["product_information"])
        self.assertFalse(self.result_path().exists())
        self.assert_sources_unchanged()

    def test_choice_is_immediately_persisted_as_human_schema_and_reloaded_from_disk(self):
        payload = self.payload(choice="B")
        saved = self.store.save(payload)
        row = saved["annotation"]
        self.assert_human_record(row)
        self.assertEqual(row["pairwise_result"]["winner_candidate_id"], "cand_0_second")
        self.assertEqual(row["pairwise_result"]["reason"], payload["reason"])
        self.assertEqual(read_rows(self.result_path()), [row])
        state = self.new_store().state("zhou")
        self.assertEqual(state["annotations"]["pair_0"], row)
        self.assertEqual(state["annotations_revision"], saved["annotations_revision"])
        self.assert_sources_unchanged()

    def test_revisiting_then_saving_swapped_display_keeps_correct_candidate_identity(self):
        first = self.store.save(self.payload(choice="A"))["annotation"]
        payload = self.payload(choice="B", reverse=True)
        self.assertEqual(payload["expected_annotation"], first)
        second = self.store.save(payload)["annotation"]
        self.assert_human_record(second)
        self.assertEqual(second["pairwise_result"]["choice"], "B")
        self.assertEqual(second["pairwise_result"]["winner_candidate_id"], "cand_0_first")
        self.assertEqual(second["presentation"], {"A": "cand_0_second", "B": "cand_0_first"})
        self.assertEqual(len(read_rows(self.result_path())), 1)

    def test_clear_removes_choice_and_requires_current_record(self):
        original = self.store.save(self.payload(choice="A"))["annotation"]
        self.assert_human_record(original)
        payload = self.payload()
        self.assertEqual(payload["expected_annotation"], original)
        response = self.store.delete(payload)
        self.assertIsNone(response["annotation"])
        self.assertEqual(read_rows(self.result_path()), [])
        self.assertEqual(self.store.state("zhou")["annotations"], {})
        self.assert_sources_unchanged()

    def test_tie_save_is_rejected_without_changing_draft_or_existing_label(self):
        state = self.store.state("zhou")
        self.store.save_draft({
            "annotator_id": "zhou", "pair_id": "pair_0",
            "dataset_revision": state["dataset_revision"],
            "expected_draft": None, "reason": "Needs review",
        })
        draft_path = self.annotations_dir / "pairwise_drafts_zhou.json"
        draft_bytes = draft_path.read_bytes()
        with self.assertRaises(annotation.APIError) as error:
            self.store.save(self.payload(choice="tie"))
        self.assertEqual(error.exception.status, 400)
        self.assertFalse(self.result_path().exists())
        self.assertEqual(draft_path.read_bytes(), draft_bytes)

        original = self.store.save(self.payload(choice="B"))["annotation"]
        label_bytes = self.result_path().read_bytes()
        with self.assertRaises(annotation.APIError) as error:
            self.store.save(self.payload(choice="tie"))
        self.assertEqual(error.exception.status, 400)
        self.assertEqual(self.result_path().read_bytes(), label_bytes)
        self.assertEqual(self.store.state("zhou")["annotations"]["pair_0"], original)
        self.assert_sources_unchanged()

    def test_external_annotation_edits_are_visible_without_restarting(self):
        original = self.store.save(self.payload())["annotation"]
        before = self.store.state("zhou")
        external = copy.deepcopy(original)
        external["pairwise_result"].update(choice="tie", winner_candidate_id=None,
                                           reason="Externally reviewed")
        write_rows(self.result_path(), [external])
        after = self.store.state("zhou")
        self.assertNotEqual(after["annotations_revision"], before["annotations_revision"])
        self.assertEqual(after["annotations"]["pair_0"], external)

    def test_concurrent_same_pair_writes_cannot_silently_overwrite(self):
        first_payload = self.payload(choice="A")
        second_payload = self.payload(choice="B")
        barrier = Barrier(2)

        def submit(payload):
            barrier.wait()
            try:
                return self.new_store().save(payload)
            except annotation.APIError as error:
                return error

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(submit, [first_payload, second_payload]))
        successes = [result for result in results if isinstance(result, dict)]
        failures = [result for result in results if isinstance(result, annotation.APIError)]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].status, 409)
        self.assertEqual(read_rows(self.result_path()), [successes[0]["annotation"]])

    def test_stale_tabs_on_different_pairs_merge_and_preserve_existing_record(self):
        first_payload = self.payload(0)
        second_payload = self.payload(1, choice="B")
        first = self.store.save(first_payload)["annotation"]
        second = self.store.save(second_payload)["annotation"]
        rows = read_rows(self.result_path())
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["pair_id"]: row for row in rows},
                         {"pair_0": first, "pair_1": second})
        self.assert_sources_unchanged()

    def test_each_annotator_has_independent_labels(self):
        zhou = self.store.save(self.payload(annotator="zhou", choice="A"))["annotation"]
        other = self.store.save(self.payload(annotator="chengkai", choice="B"))["annotation"]
        self.assertEqual(read_rows(self.result_path("zhou")), [zhou])
        self.assertEqual(read_rows(self.result_path("chengkai")), [other])
        self.assertEqual(self.store.state("zhou")["annotations"]["pair_0"], zhou)
        self.assertEqual(self.store.state("chengkai")["annotations"]["pair_0"], other)

    def test_unsubmitted_reason_draft_is_immediately_saved_to_separate_sidecar(self):
        state = self.store.state("zhou")
        payload = {"annotator_id": "zhou", "pair_id": "pair_0",
                   "dataset_revision": state["dataset_revision"],
                   "expected_draft": None, "reason": "先记下待核实的容量声明"}
        saved = self.store.save_draft(payload)
        draft_path = self.annotations_dir / "pairwise_drafts_zhou.json"
        self.assertEqual(json.loads(draft_path.read_text(encoding="utf-8")),
                         {"pair_0": {"reason": payload["reason"]}})
        reloaded = self.new_store().state("zhou")
        self.assertEqual(reloaded["drafts"]["pair_0"], {"reason": payload["reason"]})
        self.assertEqual(reloaded["drafts_revision"], saved["drafts_revision"])
        self.assertEqual(reloaded["annotations"], {})
        self.assertFalse(self.result_path().exists())
        self.assert_sources_unchanged()

    def test_completed_annotation_clears_draft_and_draft_cas_prevents_overwrite(self):
        state = self.store.state("zhou")
        draft = {"annotator_id": "zhou", "pair_id": "pair_0",
                 "dataset_revision": state["dataset_revision"],
                 "expected_draft": None, "reason": "Draft reason"}
        self.store.save_draft(draft)
        stale = copy.deepcopy(draft)
        saved = self.store.save_draft({**draft, "expected_draft": {"reason": "Draft reason"},
                                       "reason": "Updated draft"})
        with self.assertRaises(annotation.APIError) as error:
            self.store.save_draft(stale)
        self.assertEqual(error.exception.status, 409)
        payload = self.payload(choice="B")
        completed = self.store.save(payload)["annotation"]
        current = self.store.state("zhou")
        self.assertEqual(current["annotations"]["pair_0"], completed)
        self.assertNotIn("pair_0", current["drafts"])
        draft_path = self.annotations_dir / "pairwise_drafts_zhou.json"
        if draft_path.exists():
            self.assertNotIn("pair_0", json.loads(draft_path.read_text(encoding="utf-8")))
        self.assertNotEqual(saved["drafts_revision"], current["drafts_revision"])

    def test_changed_product_or_pair_updates_revision_and_rejects_stale_save(self):
        for source, rows in ((self.products_path, self.products), (self.pairs_path, self.pairs)):
            with self.subTest(source=source.name):
                old = self.payload()
                original_bytes = source.read_bytes()
                changed = copy.deepcopy(rows)
                if source == self.products_path:
                    changed[0]["product_info"]["title"] = "Changed product title"
                else:
                    changed[0]["candidates"][0]["text"] = "Updated source advertisement."
                write_rows(source, changed)
                current = self.store.state("zhou")
                self.assertNotEqual(current["dataset_revision"], old["dataset_revision"])
                with self.assertRaises(annotation.APIError) as error:
                    self.store.save(old)
                self.assertEqual(error.exception.status, 409)
                self.assertFalse(self.result_path().exists())
                source.write_bytes(original_bytes)

    def test_invalid_choice_mapping_and_path_traversal_never_write_labels(self):
        invalid = []
        for choice in (None, "left", "unknown", [], {}):
            payload = self.payload()
            payload["choice"] = choice
            invalid.append(payload)
        payload = self.payload()
        payload["presentation"] = {"A": "cand_0_first", "B": "cand_0_first"}
        invalid.append(payload)
        payload = self.payload()
        payload["presentation"] = {"A": "cand_0_first", "B": "cand_1_second"}
        invalid.append(payload)
        payload = self.payload()
        payload["annotator_id"] = "../escaped"
        invalid.append(payload)
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(annotation.APIError):
                self.store.save(payload)
        state = self.store.state("zhou")
        draft = {"annotator_id": "../escaped", "pair_id": "pair_0",
                 "dataset_revision": state["dataset_revision"],
                 "expected_draft": None, "reason": "escape"}
        with self.assertRaises(annotation.APIError):
            self.store.save_draft(draft)
        self.assertFalse(self.result_path().exists())
        self.assertFalse((self.root / "escaped.jsonl").exists())
        self.assert_sources_unchanged()

    def test_malformed_annotation_file_blocks_writes_without_destroying_data(self):
        self.store.save(self.payload())
        path = self.result_path()
        path.write_text(path.read_text() + '{"incomplete":', encoding="utf-8")
        broken = path.read_bytes()
        with self.assertRaises(annotation.APIError):
            self.store.state("zhou")
        payload = {"annotator_id": "zhou", "pair_id": "pair_1",
                   "dataset_revision": "stale", "expected_annotation": None,
                   "presentation": {"A": "cand_1_first", "B": "cand_1_second"},
                   "choice": "A", "reason": "test"}
        with self.assertRaises(annotation.APIError):
            self.store.save(payload)
        self.assertEqual(path.read_bytes(), broken)

    def test_structurally_invalid_external_annotation_is_reported_without_rewrite(self):
        saved = self.store.save(self.payload())["annotation"]
        for field, invalid in (("pair_id", []), ("choice", [])):
            with self.subTest(field=field):
                row = copy.deepcopy(saved)
                if field == "pair_id":
                    row[field] = invalid
                else:
                    row["pairwise_result"][field] = invalid
                write_rows(self.result_path(), [row])
                before = self.result_path().read_bytes()
                with self.assertRaises(annotation.APIError):
                    self.store.state("zhou")
                self.assertEqual(self.result_path().read_bytes(), before)

    def test_duplicate_pair_ids_and_missing_product_fail_without_annotation_writes(self):
        cases = [(self.pairs_path, self.pairs + [self.pairs[0]]),
                 (self.products_path, self.products[:1])]
        for source, rows in cases:
            with self.subTest(source=source.name):
                original = source.read_bytes()
                write_rows(source, rows)
                with self.assertRaises(annotation.APIError):
                    self.store.state("zhou")
                self.assertFalse(self.result_path().exists())
                source.write_bytes(original)

    def test_http_roundtrip_external_sync_cors_and_bad_payload(self):
        server = annotation.make_server(self.store, port=0, default_annotator="zhou")
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        base_url = f"http://127.0.0.1:{server.server_port}"

        def request(path, *, method="GET", body=None, headers=None):
            content = None if body is None else json.dumps(body).encode("utf-8")
            request_headers = dict(headers or {})
            if content is not None:
                request_headers.setdefault("Content-Type", "application/json")
            try:
                with urlopen(Request(base_url + path, data=content, method=method,
                                    headers=request_headers), timeout=2) as response:
                    raw = response.read()
                    return response.status, json.loads(raw) if raw else None
            except HTTPError as error:
                try:
                    raw = error.read()
                    return error.code, json.loads(raw) if raw else None
                finally:
                    error.close()

        status, state = request("/api/state?annotator_id=zhou")
        self.assertEqual(status, 200)
        self.assertEqual(state["annotations"], {})
        payload = {
            "annotator_id": "zhou",
            "pair_id": "pair_0",
            "dataset_revision": state["dataset_revision"],
            "expected_annotation": None,
            "presentation": {"A": "cand_0_first", "B": "cand_0_second"},
            "choice": "B",
            "reason": "Supported facts are more specific.",
        }
        status, result = request("/api/annotation", method="PUT", body=payload)
        self.assertEqual(status, 200)
        self.assert_human_record(result["annotation"])
        status, state = request("/api/state?annotator_id=zhou")
        self.assertEqual(status, 200)
        self.assertEqual(state["annotations"]["pair_0"], result["annotation"])

        external = copy.deepcopy(result["annotation"])
        external["pairwise_result"].update(choice="tie", winner_candidate_id=None,
                                           reason="Edited in JSONL")
        write_rows(self.result_path(), [external])
        status, state = request("/api/state?annotator_id=zhou")
        self.assertEqual(status, 200)
        self.assertEqual(state["annotations"]["pair_0"], external)

        delete_payload = {**payload, "expected_annotation": external}
        status, result = request("/api/annotation", method="DELETE", body=delete_payload)
        self.assertEqual(status, 200)
        self.assertIsNone(result["annotation"])
        self.assertEqual(read_rows(self.result_path()), [])

        status, error = request("/api/annotation", method="PUT", body=payload,
                                headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertIn("error", error)
        status, error = request("/api/annotation", method="PUT", body={"bad": True})
        self.assertEqual(status, 400)
        self.assertIn("error", error)
        status, error = request("/api/annotation", method="PUT", body={"bad": True},
                                headers={"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        self.assertIn("error", error)
        self.assert_sources_unchanged()


if __name__ == "__main__":
    unittest.main()
