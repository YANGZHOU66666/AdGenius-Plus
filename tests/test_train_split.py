"""Offline tests for the frozen training/dev split builder."""

import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_train_split.py"
SPEC = importlib.util.spec_from_file_location("build_train_split", SCRIPT)
split = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(split)


def _record(asin, title="A pen", features=("Smooth ink",), description=(), category="Writing"):
    return {"parent_asin": asin, "title": title, "features": list(features),
            "description": list(description), "categories": ["Office Products", category], "details": {}}


class TrainSplitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        records = [_record(f"A{i:03d}", category="Writing" if i % 2 else "Paper") for i in range(60)]
        records[5] = _record("A005", title="")                                  # missing title
        records[7] = _record("A007", features=(), description=())               # no text
        records[9] = _record("A009", features=("x" * 7000,))                    # too long
        records[11] = dict(records[10], parent_asin="A011")                     # same content as A010
        self.source = self.root / "raw.jsonl.gz"
        with gzip.open(self.source, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        self.pool = self.root / "train_pool.jsonl"
        with self.pool.open("w", encoding="utf-8") as handle:
            for line, record in enumerate(records, start=1):
                group = "g010" if record["parent_asin"] in {"A010", "A011"} else f"g{line:03d}"
                handle.write(json.dumps({"product_id": f"train_product_{line - 1:06d}",
                                         "source_id": record["parent_asin"], "source_line": line,
                                         "group_id": group, "split": "train_pool"}) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, out_name="out", seed=7):
        out = self.root / out_name
        manifest = split.build(self.pool, self.source, out, self.root, seed, dev_size=5, train_size=20)
        read = lambda name: split.read_jsonl(out / name)
        return manifest, read("dev_products.jsonl"), read("train_products.jsonl"), read("split_index.jsonl")

    def test_sizes_filters_and_disjointness(self):
        manifest, dev, train, index = self.build()
        self.assertEqual((len(dev), len(train)), (5, 20))
        sources = {item["source_id"] for item in dev + train}
        self.assertEqual(len(sources), 25)
        self.assertTrue({"A005", "A007", "A009"}.isdisjoint(sources))
        self.assertLessEqual(len({"A010", "A011"} & sources), 1)
        self.assertEqual(len({row["group_id"] for row in index}), 25)
        for item in dev + train:
            self.assertEqual(set(item), {"product_id", "source_id", "product_info"})
            self.assertNotIn("parent_asin", item["product_info"])

    def test_deterministic_and_seed_dependent(self):
        _, dev_a, train_a, _ = self.build("a", seed=7)
        _, dev_b, train_b, _ = self.build("b", seed=7)
        _, dev_c, train_c, _ = self.build("c", seed=8)
        self.assertEqual(dev_a + train_a, dev_b + train_b)
        self.assertNotEqual(dev_a + train_a, dev_c + train_c)

    def test_ranks_follow_line_order(self):
        _, dev, train, index = self.build()
        train_rows = [row for row in index if row["split"] == "train"]
        self.assertEqual([row["rank"] for row in train_rows], list(range(20)))
        self.assertEqual([row["product_id"] for row in train_rows], [item["product_id"] for item in train])

    def test_validate_rejects_eval_leakage(self):
        manifest, dev, _, _ = self.build()
        splits = self.root / "data" / "splits" / "v1"
        splits.mkdir(parents=True)
        (splits / "test.jsonl").write_text(json.dumps({"product_id": dev[0]["source_id"]}) + "\n")
        with self.assertRaisesRegex(RuntimeError, "gold or test"):
            split.validate(self.root / "out", self.root)


if __name__ == "__main__":
    unittest.main()
