"""Offline tests for teacher-data generation, using the mock provider (no API calls)."""

import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_teacher_data.py"
SPEC = importlib.util.spec_from_file_location("generate_teacher_data", SCRIPT)
teacher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(teacher)


def _item(index):
    return {"product_id": f"train_product_{index:06d}", "source_id": f"B{index:09d}",
            "product_info": {"title": f"Pen {index}", "features": ["Smooth ink"]}}


class TeacherDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name) / "data"
        self.run = Path(self.tmp.name) / "run"
        self.data.mkdir()
        for name, count, offset in (("dev_products.jsonl", 3, 0), ("train_products.jsonl", 5, 10)):
            (self.data / name).write_text("".join(json.dumps(_item(offset + i)) + "\n" for i in range(count)))

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *extra):
        out = StringIO()
        with redirect_stdout(out):
            code = teacher.main(["--model", "mock", "--data-dir", str(self.data), "--run-dir", str(self.run),
                                 "--workers", "2", *extra])
        return code, out.getvalue()

    def test_freezes_in_input_order_with_manifest(self):
        code, _ = self.run_main()
        self.assertEqual(code, 0)
        for split, name in (("dev", "dev_products.jsonl"), ("train", "train_products.jsonl")):
            inputs = teacher.read_jsonl(self.data / name)
            frozen = teacher.read_jsonl(self.data / f"teacher_mock_{split}.jsonl")
            self.assertEqual([r["product_id"] for r in frozen], [i["product_id"] for i in inputs])
            self.assertTrue(all(r["generation_result"]["status"] == "success" for r in frozen))
        manifest = json.loads((self.data / "teacher_mock.manifest.json").read_text())
        self.assertEqual(set(manifest["splits"]), {"dev", "train"})
        self.assertEqual(manifest["generation"]["temperature"], 1.0)
        self.assertEqual(manifest["splits"]["train"]["products"], 5)

    def test_resume_skips_finished_products(self):
        self.run_main()
        calls = teacher.read_jsonl(self.run / "train" / "calls_mock.jsonl")
        _, output = self.run_main("--splits", "train")
        self.assertIn("5 already done, 0 to generate", output)
        self.assertEqual(len(teacher.read_jsonl(self.run / "train" / "calls_mock.jsonl")), len(calls))

    def test_limit_runs_are_never_frozen(self):
        self.run_main("--limit", "2")
        self.assertFalse((self.data / "teacher_mock_dev.jsonl").exists())
        self.assertEqual(len(teacher.read_jsonl(self.run / "dev_limit2" / "results_mock.jsonl")), 2)


if __name__ == "__main__":
    unittest.main()
