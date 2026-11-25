"""Offline tests for reproducible, complete candidate-pair construction."""

import copy
import importlib.util
import hashlib
import json
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_candidate_pairs.py"
SPEC = importlib.util.spec_from_file_location("build_candidate_pairs", SCRIPT)
pairs_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pairs_script)
MODELS = list(pairs_script.DEFAULT_MODELS)


def candidate_records(count=100):
    return [
        {
            "candidate_id": f"cand_{model}_{product:03d}",
            "product_id": f"gold_product_{product}",
            "run_id": f"gen_{model}_test",
            "sample_index": 0,
            "generation_result": {"status": "success", "text": f"Ad {model} {product}",
                                  "error": None},
            "trace_id": None,
        }
        for product in range(count)
        for model in MODELS
    ]


class CandidatePairTests(unittest.TestCase):
    def test_input_and_model_order_do_not_change_pairs(self):
        records = candidate_records()
        expected = pairs_script.build_pairs(records, MODELS, 42)
        shuffled = copy.deepcopy(records)
        random.Random(7).shuffle(shuffled)
        actual = pairs_script.build_pairs(shuffled, list(reversed(MODELS)), 42)
        self.assertEqual(actual, expected)

    def test_each_product_has_two_distinct_model_candidates(self):
        records = candidate_records()
        pairs, combinations = pairs_script.build_pairs(records, MODELS, 42)
        sources = {record["candidate_id"]: record for record in records}
        self.assertEqual(len(pairs), 100)
        self.assertEqual(len({pair["pair_id"] for pair in pairs}), 100)
        self.assertEqual({pair["product_id"] for pair in pairs},
                         {record["product_id"] for record in records})
        for pair in pairs:
            self.assertEqual(set(pair), {"pair_id", "product_id", "candidates"})
            self.assertEqual(len(pair["candidates"]), 2)
            selected = [sources[candidate["candidate_id"]] for candidate in pair["candidates"]]
            self.assertEqual(len({record["run_id"] for record in selected}), 2)
            self.assertTrue(all(record["product_id"] == pair["product_id"] for record in selected))
            for candidate, source in zip(pair["candidates"], selected):
                self.assertEqual(set(candidate), {"candidate_id", "text"})
                self.assertEqual(candidate["text"], source["generation_result"]["text"])
        self.assertEqual(len(combinations), 6)
        self.assertEqual(sum(combinations.values()), 100)

    def test_seed_reproducible_and_products_sampled_independently(self):
        records = candidate_records(4)
        for seed in (0, 42, 99):
            expected = pairs_script.build_pairs(records, MODELS, seed)[0]
            self.assertEqual(pairs_script.build_pairs(records, MODELS, seed)[0], expected)
            first_product = [record for record in records if record["product_id"] == "gold_product_0"]
            self.assertEqual(pairs_script.build_pairs(first_product, MODELS, seed)[0], expected[:1])

    def test_duplicate_ids_model_slots_and_nonzero_samples_rejected(self):
        base = candidate_records(2)
        duplicate_id = copy.deepcopy(base)
        duplicate_id[1]["candidate_id"] = duplicate_id[0]["candidate_id"]
        duplicate_slot = copy.deepcopy(base)
        duplicate_slot.append(dict(duplicate_slot[0], candidate_id="another_candidate"))
        nonzero_sample = copy.deepcopy(base)
        nonzero_sample[0]["sample_index"] = 1
        for records, message in ((duplicate_id, "Duplicate candidate_id"),
                                 (duplicate_slot, "Duplicate candidate for"),
                                 (nonzero_sample, "sample_index=0")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                pairs_script.build_pairs(records, MODELS, 42)

    def _invoke(self, records, *, products=None, existing_manifest=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "generations.jsonl"
            output = root / "pairs.jsonl"
            manifest = output.with_suffix(".manifest.json")
            source.write_text("".join(json.dumps(record) + "\n" for record in records))
            args = ["--input", str(source), "--output", str(output)]
            if products is not None:
                product_file = root / "products.jsonl"
                product_file.write_text("".join(json.dumps({"product_id": value}) + "\n"
                                                for value in products))
                args += ["--products", str(product_file)]
            if existing_manifest:
                manifest.write_text("keep existing manifest")
            with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                try:
                    result = pairs_script.main(args)
                except SystemExit as error:
                    result = error.code
            return result, (output.read_text() if output.exists() else None), (
                manifest.read_text() if manifest.exists() else None)

    def test_failed_missing_or_empty_candidate_produces_no_files(self):
        failed = candidate_records(2)
        failed[0]["generation_result"] = {"status": "failed", "text": None, "error": "timeout"}
        missing = candidate_records(2)[1:]
        empty = candidate_records(2)
        empty[0]["generation_result"]["text"] = "  "
        truncated = candidate_records(2)
        truncated[0]["generation_result"] = {"status": "truncated", "text": "Ad cut", "error": "output_too_long"}
        for label, records in (("failed", failed), ("missing", missing), ("empty", empty),
                               ("truncated", truncated)):
            with self.subTest(label=label):
                status, output, manifest = self._invoke(records)
                self.assertEqual(status, 2)
                self.assertIsNone(output)
                self.assertIsNone(manifest)

    def test_product_input_detects_entirely_missing_and_extra_products(self):
        for products in ({"gold_product_0", "gold_product_1", "gold_product_2"},
                         {"gold_product_0"}):
            with self.subTest(products=products):
                status, output, manifest = self._invoke(candidate_records(2), products=products)
                self.assertEqual(status, 2)
                self.assertIsNone(output)
                self.assertIsNone(manifest)

    def test_cli_writes_schema_shaped_pairs_and_manifest(self):
        status, output, manifest_text = self._invoke(
            candidate_records(2), products={"gold_product_0", "gold_product_1"})
        self.assertEqual(status, 0)
        pairs = [json.loads(line) for line in output.splitlines()]
        manifest = json.loads(manifest_text)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(manifest["product_count"], 2)
        self.assertEqual(manifest["pair_count"], 2)
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(len(manifest["input_sha256"]), 64)
        self.assertEqual(len(manifest["products_sha256"]), 64)
        self.assertEqual(sum(manifest["source_combination_counts"].values()), 2)

    def test_existing_manifest_not_overwritten_and_no_pair_output(self):
        status, output, manifest = self._invoke(candidate_records(1), existing_manifest=True)
        self.assertEqual(status, 2)
        self.assertIsNone(output)
        self.assertEqual(manifest, "keep existing manifest")

    def test_reads_four_model_files_without_an_intermediate_combined_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = candidate_records(100)
            paths = []
            for model in MODELS:
                path = root / f"results_{model}.jsonl"
                rows = [row for row in records if row["run_id"] == f"gen_{model}_test"]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                                encoding="utf-8")
                paths.append(path)
            original_bytes = {path: path.read_bytes() for path in paths}
            output = root / "candidate_pairs.jsonl"
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                status = pairs_script.main([
                    "--input", *[str(path) for path in paths],
                    "--output", str(output), "--seed", "42",
                ])
            self.assertEqual(status, 0)
            pairs = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(pairs, pairs_script.build_pairs(records, MODELS, 42)[0])
            manifest = json.loads(output.with_suffix(".manifest.json").read_text())
            self.assertEqual(len(manifest["inputs"]), 4)
            self.assertTrue(all(row["record_count"] == 100 for row in manifest["inputs"]))
            self.assertEqual(manifest["pair_count"], 100)
            self.assertEqual(manifest["output_sha256"],
                             hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual({path: path.read_bytes() for path in paths}, original_bytes)


if __name__ == "__main__":
    unittest.main()
