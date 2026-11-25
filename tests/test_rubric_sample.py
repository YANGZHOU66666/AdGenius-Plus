"""Offline tests for the frozen rubric annotation sample."""

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_rubric_sample", ROOT / "scripts" / "build_rubric_sample.py")
sample_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sample_script)
MODELS = ["qwen_local", "gemma_31b", "glm53_flash", "longcat"]


def fake_candidates(count=100):
    products = [f"p{i:03d}" for i in range(count)]
    return products, {m: {p: {"candidate_id": f"cand_{m}_{p}"} for p in products} for m in MODELS}


class BuildSampleTest(unittest.TestCase):
    def test_counts_distinct_products_and_determinism(self):
        products, candidates = fake_candidates()
        rows = sample_script.build_sample(products, candidates, MODELS[0], MODELS[1:], 10, 7)
        self.assertEqual(len(rows), 130)
        self.assertEqual(sum(r["generator"] == "qwen_local" for r in rows), 100)
        others = [r for r in rows if r["generator"] != "qwen_local"]
        self.assertEqual(len({r["product_id"] for r in others}), 30)
        self.assertEqual(rows, sample_script.build_sample(products, candidates, MODELS[0], MODELS[1:], 10, 7))

    def test_rejects_incomplete_model(self):
        products, candidates = fake_candidates()
        del candidates["longcat"]["p000"]
        with self.assertRaises(ValueError):
            sample_script.build_sample(products, candidates, MODELS[0], MODELS[1:], 10, 7)


class FrozenSampleTest(unittest.TestCase):
    def test_frozen_sample_matches_gold(self):
        gold = ROOT / "data" / "eval" / "gold"
        rows = [json.loads(line) for line in (gold / "rubric_sample.jsonl").open(encoding="utf-8")]
        manifest = json.loads((gold / "rubric_sample.manifest.json").read_text(encoding="utf-8"))
        known = {}
        for model in MODELS:
            for line in (gold / f"results_{model}.jsonl").open(encoding="utf-8"):
                record = json.loads(line)
                known[record["candidate_id"]] = (record["product_id"], model)
        self.assertEqual(len(rows), manifest["item_count"])
        self.assertEqual(len({r["candidate_id"] for r in rows}), len(rows))
        for row in rows:
            self.assertEqual(known[row["candidate_id"]], (row["product_id"], row["generator"]))


if __name__ == "__main__":
    unittest.main()
