"""Offline tests for per-model candidate generation outputs."""

import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

from adgenius.shared.llm import LLMClient, LLMError, ModelConfig


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_candidates.py"
SPEC = importlib.util.spec_from_file_location("generate_candidates", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


class GenerateCandidateTests(unittest.TestCase):
    def _client(self):
        return SimpleNamespace(config=ModelConfig(provider="mock", model="mock", max_retries=0))

    def test_generation_retries_timeout_and_server_error_then_succeeds(self):
        client = self._client()
        responses = [
            LLMError("read timeout", attempts=1, may_have_completed=True),
            LLMError("server error", attempts=1, status_code=500),
            SimpleNamespace(text="  Accurate product copy.  ", finish_reason="stop",
                            provider="mock", to_dict=lambda: {"text": "response"}),
        ]
        client.generate = Mock(side_effect=responses)
        records = []
        with patch.object(generator.time, "sleep") as sleep:
            outcome, final_trace = generator.generate_with_retries(
                client, model_alias="test", candidate_id="cand_1",
                messages=[{"role": "user", "content": "Write copy."}],
                max_output_tokens=32, temperature=0.2, max_retries=2,
                retry_delay=0, log_call=records.append,
            )

        self.assertEqual(outcome, {"status": "success", "text": "Accurate product copy.",
                                   "error": None})
        self.assertEqual(len(records), 3)
        self.assertEqual(len({record["trace_id"] for record in records}), 3)
        self.assertEqual([record["retry_of_trace_id"] for record in records],
                         [None, records[0]["trace_id"], records[1]["trace_id"]])
        self.assertEqual(records[-1]["trace_id"], final_trace)
        self.assertTrue(records[0]["may_have_completed"])
        self.assertEqual(records[1]["status_code"], 500)
        self.assertEqual(records[-1]["response"], {"text": "response"})
        self.assertTrue(all(record["generation_request"]["max_output_tokens"] == 32
                            for record in records))
        self.assertEqual(sleep.call_count, 2)

    def test_generation_stops_after_bounded_retries_and_preserves_failure(self):
        client = self._client()
        client.generate = Mock(side_effect=LLMError(
            "temporary provider error", attempts=1, status_code=503))
        records = []
        with patch.object(generator.time, "sleep") as sleep:
            outcome, final_trace = generator.generate_with_retries(
                client, model_alias="test", candidate_id="cand_2",
                messages=[{"role": "user", "content": "Write copy."}],
                max_output_tokens=32, temperature=0, max_retries=2,
                retry_delay=0, log_call=records.append,
            )
        self.assertEqual(len(records), 3)
        self.assertEqual(outcome, {"status": "failed", "text": None,
                                   "error": "provider_error"})
        self.assertEqual(final_trace, records[-1]["trace_id"])
        self.assertEqual(sleep.call_count, 2)

    def test_authentication_failure_is_not_retried(self):
        client = self._client()
        client.generate = Mock(side_effect=LLMError(
            "unauthorized", attempts=1, status_code=401))
        records = []
        with patch.object(generator.time, "sleep") as sleep:
            outcome, _ = generator.generate_with_retries(
                client, model_alias="test", candidate_id="cand_3",
                messages=[{"role": "user", "content": "Write copy."}],
                max_output_tokens=32, temperature=0, max_retries=3,
                retry_delay=0, log_call=records.append,
            )
        self.assertEqual(len(records), 1)
        self.assertEqual(outcome["error"], "auth_error")
        sleep.assert_not_called()

    def test_provider_retry_after_survives_disabled_inner_retries(self):
        requests = []

        def respond(request):
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(503, headers={"retry-after": "30"})
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "A metal stapler."},
                             "finish_reason": "stop"}],
            })

        config = ModelConfig(provider="openai_compatible_local", model="test",
                             base_url="http://127.0.0.1:12345/v1", max_retries=0)
        records = []
        with (LLMClient(config, transport=httpx.MockTransport(respond)) as client,
              patch.object(generator.time, "sleep") as sleep,
              redirect_stdout(StringIO())):
            outcome, _ = generator.generate_with_retries(
                client, model_alias="test", candidate_id="cand_retry_after",
                messages=[{"role": "user", "content": "Write copy."}],
                max_output_tokens=32, temperature=0.2, max_retries=1,
                retry_delay=0, log_call=records.append,
            )
        self.assertEqual(outcome["status"], "success")
        self.assertEqual(len(requests), 2)
        self.assertEqual(records[0]["retry_after"], "30")
        sleep.assert_called_once_with(30.0)

    def test_refused_and_truncated_responses_are_not_retried(self):
        for finish_reason, expected in [("SAFETY", "invalid_output"),
                                        ("MAX_TOKENS", "output_too_long")]:
            with self.subTest(finish_reason=finish_reason):
                client = self._client()
                client.generate = Mock(return_value=SimpleNamespace(
                    text="Partial copy", provider="google_generative",
                    finish_reason=finish_reason, to_dict=lambda: {"text": "Partial copy"},
                ))
                records = []
                with patch.object(generator.time, "sleep") as sleep:
                    outcome, _ = generator.generate_with_retries(
                        client, model_alias="test", candidate_id="cand_4",
                        messages=[{"role": "user", "content": "Write copy."}],
                        max_output_tokens=32, temperature=0, max_retries=3,
                        log_call=records.append,
                    )
                self.assertEqual(outcome["error"], expected)
                self.assertEqual(records[0]["generation_result"], outcome)
                client.generate.assert_called_once()
                sleep.assert_not_called()

    def test_generation_main_keeps_one_result_and_all_retry_call_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "products.jsonl"
            source.write_text(json.dumps({
                "product_id": "gold_product_0", "product_info": {"title": "Stapler"},
            }) + "\n", encoding="utf-8")
            config = root / "models.toml"
            config.write_text("[models.mock]\nprovider='mock'\nmodel='mock'\n",
                              encoding="utf-8")
            output = root / "results"
            with (patch.object(generator.time, "sleep"), redirect_stdout(StringIO()),
                  patch.object(generator.LLMClient, "generate") as generate):
                response = SimpleNamespace(
                    text="A metal stapler.", provider="mock", finish_reason="stop",
                    to_dict=lambda: {"text": "A metal stapler."},
                )
                generate.side_effect = [LLMError("read timeout", may_have_completed=True),
                                        LLMError("server error", status_code=500), response]
                self.assertEqual(generator.main([
                    "--input", str(source), "--output-dir", str(output),
                    "--config", str(config), "--models", "mock", "--max-retries", "2",
                    "--timeout-seconds", "180", "--retry-delay", "0",
                ]), 0)
            results = [json.loads(line) for line in
                       (output / "results_mock.jsonl").read_text().splitlines()]
            calls = [json.loads(line) for line in
                     (output / "calls_mock.jsonl").read_text().splitlines()]
            self.assertEqual(len(results), 1)
            self.assertEqual(len(calls), 3)
            self.assertEqual(results[0]["trace_id"], calls[-1]["trace_id"])
            self.assertEqual(results[0]["generation_result"]["status"], "success")
            self.assertTrue(all(record["config"]["max_retries"] == 0 for record in calls))
            self.assertTrue(all(record["config"]["timeout_seconds"] == 180 for record in calls))
            self.assertTrue(all(record["generation_request"]["messages"][0]["content"]
                                == generator.SYSTEM_PROMPT for record in calls))

    def test_writes_separate_schema_results_and_call_logs_for_each_model(self):
        aliases = ["weak", "mid_a", "mid_b", "strong"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "products.jsonl"
            output_dir = root / "run"
            source.write_text("".join(json.dumps({
                "product_id": f"gold_product_{index:06d}",
                "source_id": f"source_{index}",
                "product_info": {"title": f"Item {index}", "features": ["Useful feature"]},
            }) + "\n" for index in range(100)), encoding="utf-8")
            config = root / "models.toml"
            config.write_text("\n".join(
                f"[models.{alias}]\nprovider = 'mock'\nmodel = '{alias}'\n"
                for alias in aliases
            ), encoding="utf-8")

            with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                result = generator.main([
                    "--input", str(source), "--output-dir", str(output_dir),
                    "--config", str(config), "--models", *aliases,
                    "--run-stamp", "test_v1",
                ])

            self.assertEqual(result, 0)
            self.assertEqual(len(list(output_dir.glob("results_*.jsonl"))), 4)
            self.assertEqual(len(list(output_dir.glob("calls_*.jsonl"))), 4)
            for alias in aliases:
                with self.subTest(alias=alias):
                    results_path = output_dir / f"results_{alias}.jsonl"
                    calls_path = output_dir / f"calls_{alias}.jsonl"
                    results = [json.loads(line) for line in results_path.read_text().splitlines()]
                    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
                    self.assertEqual(len(results), 100)
                    self.assertEqual(len(calls), 100)
                    self.assertEqual(len({row["candidate_id"] for row in results}), 100)
                    self.assertEqual(
                        set(results[0]),
                        {"candidate_id", "product_id", "run_id", "sample_index",
                         "generation_result", "trace_id"},
                    )
                    self.assertTrue(all(row["run_id"] == f"gen_{alias}_test_v1"
                                        for row in results))
                    self.assertTrue(all(row["generation_result"]["status"] == "success"
                                        for row in results))
                    self.assertEqual(
                        {row["trace_id"] for row in results},
                        {row["trace_id"] for row in calls},
                    )
                    self.assertEqual(
                        {row["candidate_id"] for row in results},
                        {row["candidate_id"] for row in calls},
                    )

    def test_existing_later_model_log_stops_before_any_call_or_new_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "products.jsonl"
            source.write_text(json.dumps({
                "product_id": "gold_product_0", "product_info": {"title": "Stapler"},
            }) + "\n", encoding="utf-8")
            output_dir = root / "run"
            output_dir.mkdir()
            existing = output_dir / "calls_second.jsonl"
            existing.write_text("keep existing log", encoding="utf-8")
            with (redirect_stderr(StringIO()), patch.object(generator, "LLMClient") as client,
                  self.assertRaises(SystemExit) as error):
                generator.main([
                    "--input", str(source), "--output-dir", str(output_dir),
                    "--models", "first", "second",
                ])
            self.assertEqual(error.exception.code, 2)
            client.assert_not_called()
            self.assertEqual(list(output_dir.iterdir()), [existing])
            self.assertEqual(existing.read_text(), "keep existing log")

    def test_default_gold_run_writes_400_candidates_to_v2_and_preserves_v1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data/eval/gold/v1/product_items.jsonl"
            source.parent.mkdir(parents=True)
            source.write_text("".join(json.dumps({
                "product_id": f"gold_product_{index:06d}",
                "source_id": f"source_{index}",
                "product_info": {"title": f"Item {index}",
                                 "features": ["Useful feature"]},
            }) + "\n" for index in range(100)), encoding="utf-8")
            original_source = source.read_bytes()
            config = root / "configs/models.toml"
            config.parent.mkdir()
            config.write_text("\n".join(
                f"[models.{alias}]\nprovider = 'mock'\nmodel = '{alias}'\n"
                for alias in generator.DEFAULT_MODELS
            ), encoding="utf-8")
            previous = root / "artifacts/eval/gold_generation/v1"
            previous.mkdir(parents=True)
            original_v1 = {}
            for alias in generator.DEFAULT_MODELS:
                for kind in ("results", "calls"):
                    path = previous / f"{kind}_{alias}.jsonl"
                    path.write_text(f"Preserve {kind} for {alias}\n", encoding="utf-8")
                    original_v1[path] = path.read_bytes()

            # Relocate the repository root so the real CLI defaults run entirely
            # inside a disposable repository; all configured models are offline.
            with (patch.object(generator, "__file__", str(root / "scripts/generate_candidates.py")),
                  redirect_stdout(StringIO()), redirect_stderr(StringIO())):
                self.assertEqual(generator.main(["--run-stamp", "test_v2"]), 0)

            output = root / "artifacts/eval/gold_generation/v2"
            self.assertEqual(len(list(output.glob("results_*.jsonl"))), 4)
            self.assertEqual(len(list(output.glob("calls_*.jsonl"))), 4)
            all_candidate_ids = set()
            for alias in generator.DEFAULT_MODELS:
                with self.subTest(alias=alias):
                    rows = [json.loads(line) for line in
                            (output / f"results_{alias}.jsonl").read_text().splitlines()]
                    calls = [json.loads(line) for line in
                             (output / f"calls_{alias}.jsonl").read_text().splitlines()]
                    self.assertEqual(len(rows), 100)
                    self.assertEqual(len(calls), 100)
                    self.assertEqual({row["product_id"] for row in rows},
                                     {f"gold_product_{index:06d}" for index in range(100)})
                    for row in rows:
                        self.assertEqual(set(row), {
                            "candidate_id", "product_id", "run_id", "sample_index",
                            "generation_result", "trace_id",
                        })
                        self.assertEqual(row["run_id"], f"gen_{alias}_test_v2")
                        self.assertEqual(row["generation_result"]["status"], "success")
                        all_candidate_ids.add(row["candidate_id"])
                    self.assertEqual({row["trace_id"] for row in rows},
                                     {call["trace_id"] for call in calls})
                    self.assertTrue(all(call["generation_request"]["max_output_tokens"] == 2048
                                        for call in calls))
                    self.assertTrue(all(call["generation_request"]["messages"][0]["content"]
                                        == generator.SYSTEM_PROMPT for call in calls))
            self.assertEqual(len(all_candidate_ids), 400)
            self.assertEqual(source.read_bytes(), original_source)
            self.assertEqual({path: path.read_bytes() for path in original_v1}, original_v1)

            original_v2 = {path: path.read_bytes() for path in output.iterdir()}
            with (patch.object(generator, "__file__", str(root / "scripts/generate_candidates.py")),
                  patch.object(generator, "LLMClient") as client,
                  redirect_stdout(StringIO()), redirect_stderr(StringIO()),
                  self.assertRaises(SystemExit) as error):
                generator.main([])
            self.assertEqual(error.exception.code, 2)
            client.assert_not_called()
            self.assertEqual({path: path.read_bytes() for path in output.iterdir()}, original_v2)


if __name__ == "__main__":
    unittest.main()
