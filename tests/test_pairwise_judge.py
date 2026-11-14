"""Offline contract tests for the AB/BA judge orchestration."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from adgenius.eval import pairwise
from adgenius.shared.llm import LLMClient, LLMError


class FakeClient:
    def __init__(self, texts, seen):
        self.texts = iter(texts)
        self.seen = seen
        self.config = SimpleNamespace(provider="openai_compatible")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def generate(self, **request):
        self.seen.append(request)
        text = next(self.texts)
        if isinstance(text, BaseException):
            raise text
        return SimpleNamespace(text=text, finish_reason="stop", raw_response={"choices": []},
                               to_dict=lambda: {
                                   "text": text,
                                   "usage": {"input_tokens": 10, "output_tokens": 4,
                                             "total_tokens": 14},
                                   "latency_seconds": 0.01,
                               })


class PairwiseJudgeTests(unittest.TestCase):
    def test_gpt6_luna_http_ab_ba_mapping_and_resume(self):
        received = []

        def handle(request):
            body = json.loads(request.content)
            received.append(body)
            self.assertEqual(str(request.url), "https://api.openai.com/v1/chat/completions")
            self.assertEqual(body["model"], "gpt-6-luna")
            self.assertEqual(body["max_completion_tokens"], 1024)
            self.assertNotIn("max_tokens", body)
            self.assertNotIn("enable_thinking", body)
            self.assertEqual(body["reasoning_effort"], "none")
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["response_format"], {"type": "json_object"})
            self.assertEqual(len(body["messages"]), 2)
            return httpx.Response(200, json={
                "id": f"chat-{len(received)}", "model": "gpt-6-luna",
                "choices": [{"message": {"role": "assistant", "content": json.dumps({
                    "choice": "1" if len(received) == 1 else "2",
                    "reason": "The benefit is clearer."})}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            })

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = pairwise.build_parser().parse_args([
                "--eval-config", str(pairwise.REPO_ROOT / "configs/eval/pairwise_gpt6_luna.toml"),
                "--limit", "1", "--run-id", "gpt6_luna_test", "--output-dir", str(output),
                "--no-human-annotations",
            ])
            factory = lambda config: LLMClient(config, transport=httpx.MockTransport(handle))
            with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-test-secret"}), \
                 patch("adgenius.eval.pairwise.LLMClient", side_effect=factory):
                outcome = pairwise.run(args)
                # A completed run reuses both successful directions without paying again.
                pairwise.run(args)
            self.assertEqual(len(received), 2)
            self.assertEqual(outcome["metrics"]["stable_winner_count"], 1)
            self.assertEqual(outcome["metrics"]["token_usage_all_attempts"]["output_tokens"]["known_sum"], 40)
            result = pairwise._read_jsonl(output / "results.jsonl")[0]
            self.assertEqual([r["choice"] for r in result["pairwise_result"]], ["A", "A"])
            self.assertEqual([r["prompt_order"] for r in result["pairwise_result"]], ["AB", "BA"])
            self.assertNotIn("openai-test-secret", (output / "calls.jsonl").read_text())

    def test_claude_native_schema_and_ab_ba_mapping_over_http(self):
        received = []

        def handle(request):
            body = json.loads(request.content)
            received.append(body)
            self.assertEqual(str(request.url), "https://api.anthropic.com/v1/messages")
            self.assertEqual(body["model"], "claude-haiku-5-5")
            self.assertNotIn("temperature", body)
            self.assertNotIn("response_format", body)
            self.assertNotIn("enable_thinking", body)
            self.assertEqual(body["thinking"], {"type": "adaptive"})
            self.assertEqual(body["output_config"]["effort"], "low")
            schema = body["output_config"]["format"]
            self.assertEqual(schema["type"], "json_schema")
            self.assertEqual(schema["schema"]["properties"]["choice"]["enum"], ["1", "2"])
            self.assertEqual(len(body["messages"]), 1)
            self.assertIn("social media feed", body["system"])
            return httpx.Response(200, json={
                "id": f"msg-{len(received)}", "model": "claude-haiku-5-5",
                "type": "message", "role": "assistant", "stop_reason": "end_turn",
                "content": [{"type": "thinking", "thinking": "Do not treat this as the answer."},
                            {"type": "text", "text": json.dumps({
                                "choice": "1" if len(received) == 1 else "2",
                                "reason": "This ad is more focused."})}],
                "usage": {"input_tokens": 100, "output_tokens": 30},
            })

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = pairwise.build_parser().parse_args([
                "--eval-config", str(pairwise.REPO_ROOT / "configs/eval/pairwise_claude.toml"),
                "--limit", "1", "--run-id", "claude_test", "--output-dir", str(output),
                "--no-human-annotations",
            ])
            factory = lambda config: LLMClient(config, transport=httpx.MockTransport(handle))
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), \
                 patch("adgenius.eval.pairwise.LLMClient", side_effect=factory):
                outcome = pairwise.run(args)
            self.assertEqual(len(received), 2)
            self.assertEqual(outcome["metrics"]["stable_winner_count"], 1)
            self.assertEqual(outcome["metrics"]["token_usage_all_attempts"]["output_tokens"]["known_sum"], 60)
            result = pairwise._read_jsonl(output / "results.jsonl")[0]
            self.assertEqual([r["choice"] for r in result["pairwise_result"]], ["A", "A"])

    def test_claude_rejects_sampling_settings_before_outputs_or_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "judge.toml"
            source = pairwise.REPO_ROOT / "configs/eval/pairwise_claude.toml"
            config_path.write_text(source.read_text().replace(
                'model_alias = "claude"', 'model_alias = "claude"\ntemperature = 0.0'))
            output = Path(directory) / "run"
            args = pairwise.build_parser().parse_args([
                "--eval-config", str(config_path), "--output-dir", str(output),
                "--limit", "1", "--no-human-annotations",
            ])
            with patch("adgenius.eval.pairwise.LLMClient", side_effect=AssertionError("network")), \
                 self.assertRaisesRegex(ValueError, "temperature"):
                pairwise.run(args)
            self.assertFalse(output.exists())

    def execute(self, texts, output_dir, *, human_labels=False, extra_args=()):
        seen = []
        fake = FakeClient(texts, seen)
        cli_args = [
            "--limit", "1", "--run-id", "judge_test", "--output-dir", str(output_dir),
        ]
        if not human_labels:
            cli_args.append("--no-human-annotations")
        args = pairwise.build_parser().parse_args([*cli_args, *extra_args])
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}), \
             patch("adgenius.eval.pairwise.LLMClient", return_value=fake):
            outcome = pairwise.run(args)
        return outcome, seen

    def test_dry_run_needs_no_key_and_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dry-run"
            args = pairwise.build_parser().parse_args([
                "--dry-run", "--limit", "1", "--output-dir", str(output),
                "--no-human-annotations",
            ])
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}), \
                 patch("adgenius.eval.pairwise.LLMClient", side_effect=AssertionError("network")):
                result = pairwise.run(args)
            self.assertEqual(result["initial_judge_calls"], 2)
            self.assertEqual(result["input_mode"], "ad_text_only")
            self.assertEqual(result["protocol_version"], "pairwise-v0.2")
            self.assertFalse(output.exists())

    def test_requests_contain_only_ad_texts_in_both_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            pairs_path = Path(directory) / "pairs.jsonl"
            # No ProductItem file exists for this pair; its product ID is only a reference.
            pair = {"pair_id": "pair_hidden", "product_id": "unprovided_product_123",
                    "candidates": [{"candidate_id": "candidate_hidden_a", "text": "Carry notes anywhere."},
                                   {"candidate_id": "candidate_hidden_b", "text": "Keep your ideas handy."}]}
            pairwise._write_rows(pairs_path, [pair])
            output = Path(directory) / "run"
            _, seen = self.execute([
                '{"choice":"1","reason":"Clear and focused."}',
                '{"choice":"2","reason":"Clear and focused."}',
            ], output, extra_args=["--pairs", str(pairs_path)])
            for request, (first, second) in zip(seen, [
                ("Carry notes anywhere.", "Keep your ideas handy."),
                ("Keep your ideas handy.", "Carry notes anywhere."),
            ]):
                self.assertEqual(request["messages"][1], {"role": "user", "content": (
                    f"Ad 1:\n{first}\n\nAd 2:\n{second}\n\n"
                    "Which ad communicates more effectively in a social media feed?\n")})
                self.assertEqual(len(request["messages"]), 2)
                for hidden in ("unprovided_product_123", "candidate_hidden", "pair_hidden"):
                    self.assertNotIn(hidden, json.dumps(request["messages"]))
                self.assertEqual(request["provider_options"], {"enable_thinking": False})
            manifest = pairwise._strict_json((output / "manifest.json").read_bytes())
            self.assertNotIn("products", manifest["specification"]["input_files"])
            self.assertEqual(manifest["specification"]["settings"]["input_mode"], "ad_text_only")

    def test_existing_human_labels_are_compared_without_relabeling_protocol(self):
        original = pairwise.DEFAULT_HUMAN.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            _, seen = self.execute([
                '{"choice":"1","reason":"More focused."}',
                '{"choice":"2","reason":"More focused."}',
            ], output, human_labels=True)
            metrics = pairwise._strict_json((output / "metrics.json").read_bytes())
            agreement = metrics["human_agreement"]
            self.assertEqual(agreement["annotated_pair_count"], 1)
            self.assertEqual(agreement["judge_protocol_version"], "pairwise-v0.2")
            self.assertEqual(agreement["annotation_protocol_versions"], ["pairwise-v0.1"])
            self.assertIn("earlier protocol", agreement["comparison_note"])
            self.assertEqual(len(seen), 2)
            self.assertEqual(pairwise.DEFAULT_HUMAN.read_bytes(), original)

    def test_product_information_placeholder_is_rejected_before_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "user.txt"
            template.write_text("{product_information}\nAd 1: {ad_1}\nAd 2: {ad_2}", encoding="utf-8")
            config = Path(directory) / "config.toml"
            default_config = pairwise.REPO_ROOT / "configs/eval/pairwise.toml"
            config.write_text(default_config.read_text(encoding="utf-8").replace(
                'user_prompt = "prompts/eval/pairwise/user_v2.txt"',
                f"user_prompt = {json.dumps(str(template))}"), encoding="utf-8")
            output = Path(directory) / "run"
            with self.assertRaisesRegex(ValueError, "product_information"):
                self.execute([], output, extra_args=["--eval-config", str(config)])
            self.assertFalse(output.exists())

    def test_earlier_protocol_run_is_rejected_without_overwriting_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            earlier = {"run_id": "judge_test", "config_fingerprint": "old_fingerprint",
                       "specification": {"settings": {"protocol_version": "pairwise-v0.1"}}}
            pairwise._write_json(output / "manifest.json", earlier)
            original = (output / "manifest.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "changed"):
                self.execute([], output)
            self.assertEqual((output / "manifest.json").read_bytes(), original)
            self.assertFalse((output / "calls.jsonl").exists())

    def test_ab_ba_choices_normalize_to_fixed_candidate_id(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "run")
            outcome, seen = self.execute([
                '{"choice":"1","reason":"Ad 1 is clearer."}',
                '{"choice":"2","reason":"The same ad is more specific."}',
            ], output)
            result = pairwise._read_jsonl(Path(output) / "results.jsonl")[0]
            mapping = result["candidate_mapping"]
            self.assertEqual([item["winner_candidate_id"] for item in result["pairwise_result"]],
                             [mapping["A"], mapping["A"]])
            self.assertEqual(outcome["metrics"]["stable_winner_count"], 1)
            self.assertEqual(len(seen), 2)
            self.assertIn("Ad 1:", seen[0]["messages"][1]["content"])
            self.assertIn("Ad 2:", seen[1]["messages"][1]["content"])
            for request in seen:
                self.assertNotIn("cand_", json.dumps(request["messages"]))
                self.assertNotIn("human_pw_", json.dumps(request["messages"]))
            self.assertEqual(outcome["metrics"]["api_attempt_count"], 2)

    def test_order_disagreement_is_not_collapsed_to_tie(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "run")
            outcome, _ = self.execute([
                '{"choice":"1","reason":"First position."}',
                '{"choice":"1","reason":"First position again."}',
            ], output)
            row = pairwise._read_jsonl(Path(output) / "results.jsonl")[0]
            self.assertIsNone(pairwise.decision(row)["winner_candidate_id"])
            self.assertEqual(pairwise.decision(row)["decision_status"], "order_inconsistent")
            self.assertEqual(outcome["metrics"]["order_inconsistent_count"], 1)

    def test_resume_reuses_success_and_retries_only_failed_order(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "run")
            first, calls1 = self.execute([
                '{"choice":"1","reason":"Ad 1 is better."}', "not json",
            ], output)
            self.assertEqual(first["status"], "completed_with_failures")
            second, calls2 = self.execute([
                '{"choice":"2","reason":"Ad 1 remains better after reversal."}',
            ], output)
            self.assertEqual(len(calls1), 2)
            self.assertEqual(len(calls2), 1)
            self.assertEqual(calls2[0]["messages"][1]["content"].split("Ad 1:")[1].splitlines()[1],
                             calls1[1]["messages"][1]["content"].split("Ad 1:")[1].splitlines()[1])
            self.assertEqual(second["status"], "completed")
            self.assertEqual(second["metrics"]["stable_winner_count"], 1)
            journal = pairwise._read_jsonl(Path(output) / "calls.jsonl")
            self.assertEqual(len(journal), 3)
            self.assertEqual(journal[2]["retry_of_trace_id"], journal[1]["trace_id"])

    def test_invalid_output_is_recorded_and_not_silently_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "run")
            outcome, _ = self.execute([
                '{"choice":"tie","reason":"Tie"}',
                '{"choice":"1","reason":"Winner"}',
            ], output)
            self.assertEqual(outcome["status"], "completed_with_failures")
            calls = pairwise._read_jsonl(Path(output) / "calls.jsonl")
            self.assertEqual(calls[0]["normalized_result"]["error"], "invalid_output")
            self.assertIsNotNone(calls[0]["response"])

    def test_retryable_provider_failure_is_logged_before_retry(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("adgenius.eval.pairwise.time.sleep") as sleep:
            output = os.path.join(directory, "run")
            outcome, seen = self.execute([
                LLMError("server error", status_code=503, attempts=1),
                '{"choice":"1","reason":"Ad 1 is better."}',
                '{"choice":"2","reason":"The same candidate wins in BA."}',
            ], output)
            calls = pairwise._read_jsonl(Path(output) / "calls.jsonl")
            self.assertEqual(len(seen), 3)
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0]["normalized_result"]["error"], "provider_error")
            self.assertEqual(calls[1]["retry_of_trace_id"], calls[0]["trace_id"])
            sleep.assert_called_once()
            self.assertEqual(outcome["metrics"]["failed_attempt_count"], 1)
            self.assertEqual(outcome["metrics"]["stable_winner_count"], 1)


if __name__ == "__main__":
    unittest.main()
