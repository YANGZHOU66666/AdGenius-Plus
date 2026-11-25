"""Offline tests for the disk-backed rubric annotation workbench."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "rubric_annotation.py"
SPEC = importlib.util.spec_from_file_location("rubric_annotation", SCRIPT)
annotation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(annotation)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_gold(directory):
    products = [{"product_id": f"p{i}", "source_id": f"S{i}",
                 "product_info": {"title": f"Stapler {i}", "features": ["Metal body"], "description": []}}
                for i in range(3)]
    write_rows(directory / "product_items.jsonl", products)
    for model in ("qwen_local", "longcat"):
        write_rows(directory / f"results_{model}.jsonl",
                   [{"candidate_id": f"cand_{model}_{i}", "product_id": f"p{i}",
                     "generation_result": {"status": "success", "text": f"Ad {model} {i}", "error": None}}
                    for i in range(3)])
    write_rows(directory / "rubric_sample.jsonl", [
        {"item_id": "rubric_item_001", "candidate_id": "cand_longcat_2", "product_id": "p2", "generator": "longcat"},
        {"item_id": "rubric_item_002", "candidate_id": "cand_qwen_local_0", "product_id": "p0", "generator": "qwen_local"},
    ])


class RubricStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.gold = Path(self.temp.name)
        make_gold(self.gold)
        self.store = annotation.RubricStore(self.gold)

    def tearDown(self):
        self.temp.cleanup()

    def payload(self, state, candidate_id, **extra):
        return {"annotator_id": "zhou", "dataset_revision": state["dataset_revision"],
                "candidate_id": candidate_id, "expected_annotation": state["annotations"].get(candidate_id), **extra}

    def test_state_hides_generator_and_keeps_sample_order(self):
        state = self.store.state("zhou")
        self.assertEqual([it["candidate_id"] for it in state["items"]], ["cand_longcat_2", "cand_qwen_local_0"])
        self.assertNotIn("generator", json.dumps(state["items"]))
        self.assertEqual(state["items"][0]["ad"], "Ad longcat 2")

    def test_save_writes_rubric_result_and_reloads(self):
        state = self.store.state("zhou")
        result = self.store.save(self.payload(state, "cand_qwen_local_0", values={"factual_support": False},
                                              reasons={"factual_support": "  30 sheets vs 20  "}))
        record = result["annotation"]
        self.assertEqual(record["judge_type"], "human")
        self.assertEqual(record["product_id"], "p0")
        self.assertEqual(record["rubric_result"], {"status": "success", "error": None, "criteria": [
            {"criterion_id": "factual_support", "value": False, "reason": "30 sheets vs 20"}]})
        reloaded = self.store.state("zhou")
        self.assertEqual(reloaded["annotations"]["cand_qwen_local_0"], record)
        # Editing keeps the annotation ID.
        edited = self.store.save(self.payload(reloaded, "cand_qwen_local_0", values={"factual_support": True}))
        self.assertEqual(edited["annotation"]["annotation_id"], record["annotation_id"])
        self.assertIsNone(edited["annotation"]["rubric_result"]["criteria"][0]["reason"])

    def test_file_follows_sample_order(self):
        state = self.store.state("zhou")
        self.store.save(self.payload(state, "cand_qwen_local_0", values={"factual_support": True}))
        state = self.store.state("zhou")
        self.store.save(self.payload(state, "cand_longcat_2", values={"factual_support": True}))
        rows = [json.loads(line) for line in (self.gold / "rubric_annotations_zhou.jsonl").read_text().splitlines()]
        self.assertEqual([r["candidate_id"] for r in rows], ["cand_longcat_2", "cand_qwen_local_0"])

    def test_stale_expected_annotation_is_rejected(self):
        state = self.store.state("zhou")
        self.store.save(self.payload(state, "cand_longcat_2", values={"factual_support": True}))
        with self.assertRaises(annotation.APIError) as raised:
            self.store.save(self.payload(state, "cand_longcat_2", values={"factual_support": False}))
        self.assertEqual(raised.exception.status, 409)

    def test_invalid_values_are_rejected(self):
        state = self.store.state("zhou")
        for values in ({}, {"factual_support": "false"}, {"factual_support": True, "concrete_product_info": True}):
            with self.assertRaises(annotation.APIError):
                self.store.save(self.payload(state, "cand_longcat_2", values=values))
        self.assertFalse((self.gold / "rubric_annotations_zhou.jsonl").exists())

    def test_delete_removes_label(self):
        state = self.store.state("zhou")
        self.store.save(self.payload(state, "cand_longcat_2", values={"factual_support": True}))
        state = self.store.state("zhou")
        self.store.delete(self.payload(state, "cand_longcat_2"))
        self.assertEqual(self.store.state("zhou")["annotations"], {})

    def test_two_criteria(self):
        store = annotation.RubricStore(self.gold, criteria=("factual_support", "concrete_product_info"))
        state = store.state("zhou")
        record = store.save(self.payload(state, "cand_longcat_2",
                                         values={"factual_support": True, "concrete_product_info": False}))["annotation"]
        self.assertEqual([c["criterion_id"] for c in record["rubric_result"]["criteria"]],
                         ["factual_support", "concrete_product_info"])
        # A file written for two criteria does not load under a one-criterion protocol.
        with self.assertRaises(annotation.APIError):
            self.store.state("zhou")

    def test_sample_mismatch_is_rejected(self):
        write_rows(self.gold / "rubric_sample.jsonl", [
            {"item_id": "rubric_item_001", "candidate_id": "cand_longcat_2", "product_id": "p0", "generator": "longcat"}])
        with self.assertRaises(annotation.APIError):
            self.store.state("zhou")


class RubricServerTest(unittest.TestCase):
    def test_http_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            gold = Path(temp)
            make_gold(gold)
            server = annotation.make_server(annotation.RubricStore(gold))
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                state = json.loads(urlopen(f"{base}/api/state?annotator_id=zhou").read())
                body = json.dumps({"annotator_id": "zhou", "dataset_revision": state["dataset_revision"],
                                   "candidate_id": "cand_longcat_2", "expected_annotation": None,
                                   "values": {"factual_support": True}}).encode()
                request = Request(f"{base}/api/annotation", body, {"Content-Type": "application/json"}, method="PUT")
                self.assertTrue(json.loads(urlopen(request).read())["annotation"]["rubric_result"]["criteria"][0]["value"])
                for page in ("/", "/app.js", "/style.css"):
                    self.assertEqual(urlopen(base + page).status, 200)
                foreign = Request(f"{base}/api/annotation", body,
                                  {"Content-Type": "application/json", "Origin": "http://evil.test"}, method="PUT")
                with self.assertRaises(HTTPError) as raised:
                    urlopen(foreign)
                self.assertEqual(raised.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
