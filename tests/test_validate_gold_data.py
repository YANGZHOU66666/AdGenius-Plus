"""Offline tests for scripts/validate_gold_data.py on a copy of the shared data."""

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("validate_gold_data", ROOT / "scripts" / "validate_gold_data.py")
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)
COPIED = ("data/eval/gold", "metadata/schemas", "results/pairwise", "results/rubric")


class ValidateGoldDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for part in COPIED:
            shutil.copytree(ROOT / part, self.root / part)

    def tearDown(self):
        self.temp.cleanup()

    def rewrite(self, relative, change):
        path = self.root / relative
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        change(rows)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def test_shared_data_validates(self):
        summary = validator.validate(self.root)
        self.assertEqual((summary["rubric_items"], summary["rubric_human_false"],
                          summary["validated_rubric_runs"]), (130, 18, 4))

    def test_changed_rubric_label_is_caught(self):
        def flip(rows):
            criterion = rows[0]["rubric_result"]["criteria"][0]
            criterion["value"], criterion["reason"] = False, "edited"
        self.rewrite("data/eval/gold/rubric_annotations_zhou.jsonl", flip)
        with self.assertRaisesRegex(ValueError, "different human labels"):
            validator.validate(self.root)

    def test_changed_rubric_judge_result_is_caught(self):
        def flip(rows):
            criterion = rows[0]["rubric_result"]["criteria"][0]
            criterion["value"] = not criterion["value"]
        self.rewrite("results/rubric/deepseek41_rubric_v1/results.jsonl", flip)
        with self.assertRaisesRegex(ValueError, "Rubric judge result hash mismatch"):
            validator.validate(self.root)


if __name__ == "__main__":
    unittest.main()
