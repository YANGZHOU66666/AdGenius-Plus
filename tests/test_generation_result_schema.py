"""GenerationResult schema: the three generation states and the shared frozen data."""

import json
import unittest
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "metadata/schemas/generation_result.schema.json").read_text(encoding="utf-8"))
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)


def record(result):
    return {"candidate_id": "cand_1", "product_id": "prod_1", "run_id": "gen_test_v1", "sample_index": 0,
            "generation_result": result, "trace_id": None}


class GenerationResultSchemaTests(unittest.TestCase):
    def test_accepts_success_truncated_and_failed(self):
        for result in ({"status": "success", "text": "Complete ad.", "error": None},
                       {"status": "truncated", "text": "An ad that ran past the", "error": "output_too_long"},
                       {"status": "failed", "text": None, "error": "output_too_long"},
                       {"status": "failed", "text": None, "error": "timeout"}):
            with self.subTest(result=result):
                VALIDATOR.validate(record(result))

    def test_rejects_inconsistent_states(self):
        for result in ({"status": "truncated", "text": None, "error": "output_too_long"},
                       {"status": "truncated", "text": "   ", "error": "output_too_long"},
                       {"status": "truncated", "text": "Cut ad", "error": None},
                       {"status": "truncated", "text": "Cut ad", "error": "timeout"},
                       {"status": "success", "text": "Ad", "error": "output_too_long"},
                       {"status": "failed", "text": "Ad", "error": "timeout"},
                       {"status": "partial", "text": "Ad", "error": None}):
            with self.subTest(result=result):
                self.assertFalse(VALIDATOR.is_valid(record(result)))

    def test_existing_frozen_generation_files_still_validate(self):
        paths = sorted((ROOT / "data/eval/gold").glob("results_*.jsonl")) + sorted(
            (ROOT / "data/train/v1").glob("teacher_*_*.jsonl"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name), path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        VALIDATOR.validate(json.loads(line))


if __name__ == "__main__":
    unittest.main()
