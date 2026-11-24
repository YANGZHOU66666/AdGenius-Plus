"""Offline recovery tests: preserve completed work and replay original settings."""

import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    SPEC = importlib.util.spec_from_file_location(
        "retry_failed_candidates", SCRIPTS / "retry_failed_candidates.py")
    retry = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(retry)


ALIASES = ("qwen_local", "gemma_31b", "glm53_flash", "longcat")
KEY_ENV = "ADGENIUS_RETRY_OFFLINE_TEST_KEY"
HISTORICAL_SYSTEM = "Historical experiment prompt; preserve this exact text."


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class FakeClient:
    def __init__(self, config):
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


def successful_retry(client, **kwargs):
    """Persist the same recovery metadata as a real successful helper call."""
    result = {"status": "success", "text": "Recovered advertisement.", "error": None}
    trace = "retry_" + kwargs["candidate_id"]
    kwargs["log_call"]({
        "candidate_id": kwargs["candidate_id"], "trace_id": trace,
        "model_alias": kwargs["model_alias"], "status": "success",
        "retry_of_trace_id": kwargs["retry_of_trace_id"], "attempt": 1,
        "config": client.config.snapshot(),
        "generation_request": {key: copy.deepcopy(kwargs[key]) for key in
                               ("messages", "max_output_tokens", "temperature")},
        "generation_result": result,
    })
    return result, trace


class RetryFailedCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "original"
        self.source.mkdir()
        self.output = self.source
        self.backup = self.source.with_name(f"{self.source.name}_original")
        self.input = self.root / "products.jsonl"
        self.products = [{"product_id": f"gold_product_{index}",
                          "product_info": {"title": f"Stapler {index}",
                                           "features": ["Metal body"]}}
                         for index in range(3)]
        self.input.write_text("".join(json.dumps(row) + "\n" for row in self.products))
        for alias in ALIASES:
            results, calls = [], []
            config = retry.ModelConfig(
                provider="google_generative", model=f"historical-{alias}",
                base_url="https://example.test/v1beta",
                api_key_env=KEY_ENV if alias == "gemma_31b" else f"{KEY_ENV}_{alias}",
                timeout_seconds=60, max_retries=2).snapshot()
            for index, product in enumerate(self.products):
                candidate_id, trace = f"cand_{alias}_{index}", f"call_{alias}_{index}"
                failed = alias == "gemma_31b" and index == 1
                result = {"status": "failed" if failed else "success",
                          "text": None if failed else f"Original ad {alias} {index}",
                          "error": "timeout" if failed else None}
                results.append({"candidate_id": candidate_id,
                                "product_id": product["product_id"],
                                "run_id": f"gen_{alias}_frozen_v1", "sample_index": 0,
                                "generation_result": result, "trace_id": trace})
                call = {"candidate_id": candidate_id, "trace_id": trace,
                        "model_alias": alias, "status": result["status"]}
                if failed:
                    # The original failed logs did not preserve request settings.
                    call.update(error="ReadTimeout", error_code="timeout", attempts=1)
                else:
                    call["response"] = {"config": config, "request": {
                        "systemInstruction": {"parts": [{"text": HISTORICAL_SYSTEM}]},
                        "contents": [{"role": "user", "parts": [{
                            "text": retry._product_prompt(product)}]}],
                        "generationConfig": {"maxOutputTokens": 2048,
                                             "temperature": 0.2},
                    }}
                calls.append(call)
            for name, rows in (("results", results), ("calls", calls)):
                (self.source / f"{name}_{alias}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows))
        self.original_bytes = {path.name: path.read_bytes() for path in self.source.iterdir()}
        dotenv = patch.object(retry, "load_dotenv")
        dotenv.start()
        self.addCleanup(dotenv.stop)

    def invoke(self, *extra):
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(StringIO()):
            status = retry.main(["--input", str(self.input),
                                 "--source-dir", str(self.source), *extra])
        return status, output.getvalue()

    def repair(self):
        with (patch.dict(os.environ, {KEY_ENV: "offline-key"}),
              patch.object(retry, "LLMClient", side_effect=FakeClient) as clients,
              patch.object(retry, "generate_with_retries",
                           side_effect=successful_retry) as helper):
            status, output = self.invoke()
        return status, output, clients, helper

    def assert_originals_untouched(self):
        original_directory = self.backup if self.backup.exists() else self.source
        self.assertEqual({path.name: path.read_bytes() for path in original_directory.glob("*.jsonl")},
                         self.original_bytes)

    def test_dry_run_does_not_need_keys_create_files_or_call_models(self):
        with (patch.dict(os.environ, {}, clear=True),
              patch.object(retry, "LLMClient") as clients,
              patch.object(retry, "generate_with_retries") as helper):
            status, output = self.invoke("--dry-run")
        self.assertEqual(status, 0)
        self.assertIn("Total to retry: 1", output)
        self.assertIn("gemma_31b: 1 failed candidates", output)
        self.assertFalse(self.backup.exists())
        self.assertFalse((self.output / "retried").exists())
        self.assertEqual(set(path.name for path in self.source.iterdir()),
                         set(self.original_bytes))
        clients.assert_not_called()
        helper.assert_not_called()
        self.assert_originals_untouched()

    def test_truncated_results_are_kept_and_never_retried(self):
        path = self.source / "results_longcat.jsonl"
        rows = read_rows(path)
        rows[0]["generation_result"] = {"status": "truncated", "text": "Original ad longcat 0, cut",
                                        "error": "output_too_long"}
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.original_bytes = {item.name: item.read_bytes() for item in self.source.iterdir()}
        status, _, _, helper = self.repair()
        self.assertEqual(status, 0)
        helper.assert_called_once()
        self.assertEqual(helper.call_args.kwargs["candidate_id"], "cand_gemma_31b_1")
        self.assertEqual(read_rows(self.output / "results_longcat.jsonl")[0], rows[0])

    def test_invalid_truncated_result_is_rejected(self):
        for result in ({"status": "truncated", "text": None, "error": "output_too_long"},
                       {"status": "truncated", "text": "  ", "error": "output_too_long"},
                       {"status": "truncated", "text": "Cut copy", "error": "timeout"}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                retry._validate_results([{"candidate_id": "cand_x", "generation_result": result}],
                                        Path("results_x.jsonl"))

    def test_repairs_only_failures_preserves_ids_and_replays_logged_settings(self):
        status, _, clients, helper = self.repair()
        self.assertEqual(status, 0)
        self.assertEqual(clients.call_count, 1)
        helper.assert_called_once()
        client = helper.call_args.args[0]
        arguments = helper.call_args.kwargs
        self.assertEqual(client.config.model, "historical-gemma_31b")
        self.assertEqual(client.config.timeout_seconds, 180)
        self.assertEqual(client.config.max_retries, 0)
        self.assertEqual(arguments["messages"], [
            {"role": "system", "content": HISTORICAL_SYSTEM},
            {"role": "user", "content": retry._product_prompt(self.products[1])},
        ])
        self.assertEqual(arguments["max_output_tokens"], 2048)
        self.assertEqual(arguments["temperature"], 0.2)
        self.assertEqual(arguments["candidate_id"], "cand_gemma_31b_1")
        self.assertEqual(arguments["retry_of_trace_id"], "call_gemma_31b_1")
        for alias in ALIASES:
            originals = read_rows(self.backup / f"results_{alias}.jsonl")
            repaired = read_rows(self.output / f"results_{alias}.jsonl")
            self.assertEqual(len(originals), len(repaired))
            for before, after in zip(originals, repaired):
                if before["generation_result"]["status"] == "success":
                    self.assertEqual(before, after)
                else:
                    for key in ("candidate_id", "product_id", "run_id", "sample_index"):
                        self.assertEqual(before[key], after[key])
                    self.assertEqual(after["generation_result"]["status"], "success")
                    self.assertNotEqual(before["trace_id"], after["trace_id"])
            original_calls = read_rows(self.backup / f"calls_{alias}.jsonl")
            repaired_calls = read_rows(self.output / f"calls_{alias}.jsonl")
            self.assertEqual(repaired_calls[:len(original_calls)], original_calls)
            self.assertEqual(len(repaired_calls), 4 if alias == "gemma_31b" else 3)
        self.assert_originals_untouched()
        self.assertFalse((self.source / "retried").exists())
        manifest = json.loads((self.output / "retry_manifest.json").read_text())
        self.assertEqual(manifest["source_dir"], str(self.backup.resolve()))

    def test_resume_skips_repaired_successes_without_keys_or_api_calls(self):
        self.repair()
        before = (self.output / "calls_gemma_31b.jsonl").read_bytes()
        with (patch.dict(os.environ, {}, clear=True),
              patch.object(retry, "LLMClient") as clients,
              patch.object(retry, "generate_with_retries") as helper):
            status, output = self.invoke()
        self.assertEqual(status, 0)
        self.assertIn("Total to retry: 0", output)
        clients.assert_not_called()
        helper.assert_not_called()
        self.assertEqual((self.output / "calls_gemma_31b.jsonl").read_bytes(), before)
        self.assert_originals_untouched()

    def test_complete_run_needs_no_backup_or_manifest(self):
        path = self.source / "results_gemma_31b.jsonl"
        rows = read_rows(path)
        rows[1]["generation_result"] = {"status": "success", "text": "Already complete.",
                                           "error": None}
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        before = {path.name: path.read_bytes() for path in self.source.glob("*.jsonl")}
        with (patch.dict(os.environ, {}, clear=True),
              patch.object(retry, "LLMClient") as clients):
            status, output = self.invoke()
        self.assertEqual(status, 0)
        self.assertIn("no repair or backup needed", output)
        self.assertFalse(self.backup.exists())
        self.assertFalse((self.source / "retry_manifest.json").exists())
        self.assertFalse((self.source / "retried").exists())
        clients.assert_not_called()
        self.assertEqual({path.name: path.read_bytes() for path in self.source.glob("*.jsonl")},
                         before)

    def test_recovers_flushed_success_after_interruption_without_repeating_call(self):
        def interrupted(client, **kwargs):
            successful_retry(client, **kwargs)
            raise KeyboardInterrupt()

        with (patch.dict(os.environ, {KEY_ENV: "offline-key"}),
              patch.object(retry, "LLMClient", side_effect=FakeClient),
              patch.object(retry, "generate_with_retries", side_effect=interrupted),
              self.assertRaises(KeyboardInterrupt)):
            self.invoke()
        rows = read_rows(self.output / "results_gemma_31b.jsonl")
        self.assertEqual(rows[1]["generation_result"]["status"], "failed")
        with (patch.dict(os.environ, {}, clear=True),
              patch.object(retry, "LLMClient") as clients,
              patch.object(retry, "generate_with_retries") as helper):
            status, _ = self.invoke()
        self.assertEqual(status, 0)
        clients.assert_not_called()
        helper.assert_not_called()
        rows = read_rows(self.output / "results_gemma_31b.jsonl")
        self.assertEqual(rows[1]["generation_result"]["status"], "success")
        self.assertEqual(rows[1]["trace_id"], "retry_cand_gemma_31b_1")
        self.assert_originals_untouched()

    def test_missing_key_is_rejected_before_calls_or_copying_original_data(self):
        with (patch.dict(os.environ, {}, clear=True),
              patch.object(retry, "LLMClient") as clients,
              patch.object(retry, "generate_with_retries") as helper,
              self.assertRaises(SystemExit) as error):
            self.invoke()
        self.assertEqual(error.exception.code, 2)
        clients.assert_not_called()
        helper.assert_not_called()
        self.assertFalse((self.output / "retry_manifest.json").exists())
        self.assertFalse(self.backup.exists())
        self.assert_originals_untouched()

    def test_explicit_source_output_repairs_in_place_with_sibling_backup(self):
        with (patch.dict(os.environ, {KEY_ENV: "offline-key"}),
              patch.object(retry, "LLMClient", side_effect=FakeClient),
              patch.object(retry, "generate_with_retries", side_effect=successful_retry)):
            status, _ = self.invoke("--output-dir", str(self.source))
        self.assertEqual(status, 0)
        self.assertTrue(self.backup.is_dir())
        self.assertFalse((self.source / "retried").exists())
        self.assert_originals_untouched()

    def test_refuses_uninitialized_nonempty_separate_output(self):
        separate = self.root / "separate"
        separate.mkdir()
        marker = separate / "unrelated.txt"
        marker.write_text("preserve")
        with (patch.object(retry, "LLMClient") as clients,
              self.assertRaises(SystemExit) as error):
            self.invoke("--output-dir", str(separate))
        self.assertEqual(error.exception.code, 2)
        clients.assert_not_called()
        self.assertEqual(marker.read_text(), "preserve")
        self.assert_originals_untouched()

    def test_optional_separate_output_remains_supported(self):
        separate = self.root / "separate"
        with (patch.dict(os.environ, {KEY_ENV: "offline-key"}),
              patch.object(retry, "LLMClient", side_effect=FakeClient),
              patch.object(retry, "generate_with_retries", side_effect=successful_retry)):
            status, _ = self.invoke("--output-dir", str(separate))
        self.assertEqual(status, 0)
        self.assertEqual(read_rows(separate / "results_gemma_31b.jsonl")[1]
                         ["generation_result"]["status"], "success")
        self.assertFalse(self.backup.exists())
        self.assert_originals_untouched()

    def test_resume_rejects_changed_input_and_changed_original_file(self):
        self.repair()
        input_bytes = self.input.read_bytes()
        # Harmless whitespace still changes the frozen source fingerprint.
        self.input.write_bytes(input_bytes + b"\n")
        with (patch.object(retry, "LLMClient") as clients,
              self.assertRaises(SystemExit) as error):
            self.invoke()
        self.assertEqual(error.exception.code, 2)
        clients.assert_not_called()
        self.input.write_bytes(input_bytes)
        source_result = self.backup / "results_longcat.jsonl"
        source_result.write_bytes(source_result.read_bytes() + b"\n")
        with (patch.object(retry, "LLMClient") as clients,
              self.assertRaises(SystemExit) as error):
            self.invoke()
        self.assertEqual(error.exception.code, 2)
        clients.assert_not_called()

    def test_rejects_changed_product_prompt_before_any_call(self):
        self.products[0]["product_info"]["title"] = "Changed source product"
        self.input.write_text("".join(json.dumps(row) + "\n" for row in self.products))
        with (patch.object(retry, "LLMClient") as clients,
              self.assertRaises(SystemExit) as error):
            self.invoke()
        self.assertEqual(error.exception.code, 2)
        clients.assert_not_called()
        self.assertFalse((self.output / "retry_manifest.json").exists())
        self.assert_originals_untouched()


if __name__ == "__main__":
    unittest.main()
