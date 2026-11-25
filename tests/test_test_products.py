"""Offline tests for the test ProductItem conversion (scripts/build_test_products.py)."""

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_test_products", ROOT / "scripts" / "build_test_products.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class TestProductTests(unittest.TestCase):
    def test_gold_rule_ids_and_unchanged_values(self):
        record = {"parent_asin": "B000TEST01", "title": "Stapler", "price": None, "features": []}
        item = builder.to_product_item(7, record)
        self.assertEqual(item, {"product_id": "test_product_000007", "source_id": "B000TEST01",
                                "product_info": {"title": "Stapler", "price": None, "features": []}})
        self.assertIn("parent_asin", record)  # the source record is not modified

    def test_rebuild_matches_the_frozen_file_and_passes_checks(self):
        items, manifest = builder.build(ROOT)
        checks = builder.validate(ROOT, items)
        self.assertEqual(len(items), 1000)
        self.assertEqual(checks["overlaps"], {"gold_asins": 0, "gold_groups": 0, "dev_train_asins": 0,
                                              "dev_train_groups": 0})
        frozen = ROOT / "data/eval/test/product_items.jsonl"
        expected = "".join(json.dumps(i, ensure_ascii=False, separators=(",", ":")) + "\n" for i in items)
        self.assertEqual(frozen.read_text(encoding="utf-8"), expected)
        stored = json.loads((ROOT / "data/eval/test/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["files"]["product_items.jsonl"]["sha256"], builder.sha256_file(frozen))
        self.assertEqual(stored["descriptive"], manifest["descriptive"])


if __name__ == "__main__":
    unittest.main()
