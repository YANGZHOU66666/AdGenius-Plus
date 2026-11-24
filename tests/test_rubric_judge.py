"""Offline tests for the rubric LLM judge and its agreement metrics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adgenius.eval import rubric, rubric_metrics
from adgenius.shared.llm import ModelConfig

ROOT = Path(__file__).resolve().parents[1]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_gold(directory: Path) -> None:
    write_rows(directory / "product_items.jsonl", [
        {"product_id": f"p{i}", "source_id": f"S{i}",
         "product_info": {"title": f"Stapler {i}", "features": ["Staples up to 20 sheets"]}} for i in range(2)])
    write_rows(directory / "results_qwen_local.jsonl", [
        {"candidate_id": f"cand_{i}", "product_id": f"p{i}",
         "generation_result": {"status": "success", "text": f"Ad {i}", "error": None}} for i in range(2)])
    write_rows(directory / "rubric_sample.jsonl", [
        {"item_id": "rubric_item_001", "candidate_id": "cand_1", "product_id": "p1", "generator": "qwen_local"},
        {"item_id": "rubric_item_002", "candidate_id": "cand_0", "product_id": "p0", "generator": "qwen_local"}])


class FakeClient:
    def __init__(self, texts):
        self.texts = texts

    def generate(self, **request):
        text = self.texts.pop(0)
        return SimpleNamespace(text=text, to_dict=lambda: {"usage": {"input_tokens": 5, "output_tokens": 3}})


class RubricJudgeTest(unittest.TestCase):
    def test_configs_load(self):
        for path in sorted((ROOT / "configs" / "eval").glob("rubric_*.toml")):
            settings = rubric.load_settings(path)
            self.assertEqual(settings["criteria"], ["factual_support"])
            self.assertTrue((ROOT / settings["system_prompt"]).is_file())

    def test_settings_reject_other_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.toml"
            text = (ROOT / "configs/eval/rubric_deepseek.toml").read_text(encoding="utf-8")
            path.write_text(text.replace('"product_and_ad"', '"ad_text_only"'), encoding="utf-8")
            with self.assertRaises(ValueError):
                rubric.load_settings(path)

    def test_requests(self):
        chat = [{"role": "user", "content": "x"}]
        claude = ModelConfig(provider="anthropic", model="claude-test", base_url="https://api.anthropic.com",
                             api_key_env="ANTHROPIC_API_KEY")
        request = rubric.judge_request(claude, rubric.load_settings(ROOT / "configs/eval/rubric_claude.toml"), chat)
        self.assertIsNone(request["temperature"])
        self.assertFalse(request["json_mode"])
        self.assertEqual(request["provider_options"]["output_config"]["format"]["schema"], rubric.OUTPUT_SCHEMA)
        openai = ModelConfig(provider="openai_compatible", model="m", base_url="https://example.com/v1", api_key_env="KEY")
        request = rubric.judge_request(openai, rubric.load_settings(ROOT / "configs/eval/rubric_deepseek.toml"), chat)
        self.assertTrue(request["json_mode"])
        self.assertEqual(request["temperature"], 0.0)

    def test_parse_and_record(self):
        verdict = rubric.parse_verdict(json.dumps({"value": False, "reason": "30 vs 20",
                                                   "conflicts": [{"ad": "30 sheets", "information": "20 sheets"}]}))
        item = {"candidate_id": "cand_1", "product_id": "p1"}
        record = rubric.result_record(item, "run", "rubric-v0.2", verdict, "call_rb_1")
        rubric._validator().validate(record)
        reason = record["rubric_result"]["criteria"][0]["reason"]
        self.assertEqual(reason, '30 vs 20 | conflicts: ad "30 sheets" vs information "20 sheets"')
        failed = rubric.result_record(item, "run", "rubric-v0.2", None, None, "judge_failed")
        rubric._validator().validate(failed)
        for text in ('{"value": "false", "conflicts": []}', '{"value": true}', "not json"):
            with self.assertRaises(ValueError):
                rubric.parse_verdict(text)

    def test_run_retries_and_keeps_sample_order(self):
        with tempfile.TemporaryDirectory() as temp:
            gold = Path(temp) / "gold"
            gold.mkdir()
            make_gold(gold)
            texts = ["not json", '{"value": true, "conflicts": [], "reason": "ok"}',
                     '{"value": true, "conflicts": [], "reason": "ok"}']
            args = SimpleNamespace(env_file=None, eval_config=str(ROOT / "configs/eval/rubric_deepseek.toml"),
                                   gold_dir=str(gold), limit=None, run_id="test_run",
                                   output_dir=str(Path(temp) / "out"), calls_dir=str(Path(temp) / "calls"),
                                   dry_run=False, workers=1)
            with patch.object(rubric, "LLMClient", lambda config: FakeClient(texts)), \
                    patch.object(rubric.time, "sleep"), patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}):
                summary = rubric.run(args)
            self.assertEqual(summary["status"], "completed")
            rows = [json.loads(line) for line in (Path(temp) / "out/results.jsonl").read_text().splitlines()]
            self.assertEqual([r["candidate_id"] for r in rows], ["cand_1", "cand_0"])
            calls = [json.loads(line) for line in (Path(temp) / "calls/calls.jsonl").read_text().splitlines()]
            self.assertEqual([c["status"] for c in calls], ["failed", "success", "success"])
            manifest = json.loads((Path(temp) / "out/manifest.json").read_text())
            self.assertEqual(manifest["specification"]["candidate_ids"], ["cand_1", "cand_0"])


class RubricMetricsTest(unittest.TestCase):
    def test_confusion(self):
        pairs = [(False, False), (False, True), (True, False), (True, True), (True, True)]
        stats = rubric_metrics.confusion(pairs)
        self.assertEqual((stats["caught"], stats["missed"], stats["extra_false"]), (1, 1, 1))
        self.assertAlmostEqual(stats["accuracy"], 0.6)
        self.assertAlmostEqual(stats["kappa"], (0.6 - 0.52) / 0.48)

    def test_summarize_counts_unquoted_false(self):
        def record(cid, value, reason):
            return {"candidate_id": cid, "rubric_result": {"status": "success", "error": None, "criteria": [
                {"criterion_id": "factual_support", "value": value, "reason": reason}]}}
        results = [record("a", False, 'x | conflicts: ad "1" vs information "2"'), record("b", False, "no quote"),
                   record("c", True, None)]
        summary = rubric_metrics.summarize(results, {"a": False, "b": True, "c": True},
                                           {"a": "qwen_local", "b": "gemma_31b", "c": "qwen_local"}, iterations=50)
        self.assertEqual(summary["false_without_quoted_conflict"], 1)
        self.assertEqual(summary["by_generator"]["other_generators"]["extra_false"], 1)
        self.assertEqual(summary["caught"], 1)


if __name__ == "__main__":
    unittest.main()
