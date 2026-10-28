"""Contract tests using fake HTTP responses; no keys or paid calls required."""

import json
import os
import unittest
from unittest.mock import patch

import httpx

from adgenius.shared.llm import LLMClient, LLMError, ModelConfig

MESSAGES = [{"role": "system", "content": "Write accurate copy."},
            {"role": "user", "content": "A blue stapler."}]


def compatible_response(**changes):
    body = {"id": "chat-1", "model": "actual-version",
            "choices": [{"message": {"content": "Keep your papers together."},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26,
                      "prompt_tokens_details": {"cached_tokens": 5},
                      "completion_tokens_details": {"reasoning_tokens": 2}}}
    body.update(changes)
    return httpx.Response(200, json=body, headers={"x-request-id": "req-1"})


class LLMTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"LLM_TEST_KEY": "test-secret"})
        env.start()
        self.addCleanup(env.stop)

    def config(self, provider="openai_compatible", **changes):
        values = dict(provider=provider, model="test-model",
                      base_url="https://example.test/v1", api_key_env="LLM_TEST_KEY",
                      max_retries=0)
        values.update(changes)
        return ModelConfig(**values)

    def test_compatible_payload_usage_and_no_authorization_in_record(self):
        received = []

        def handle(request):
            received.append(request)
            return compatible_response()

        config = self.config(request_options={"enable_thinking": False})
        with LLMClient(config, transport=httpx.MockTransport(handle)) as client:
            response = client.generate(MESSAGES, max_output_tokens=100, temperature=0,
                                       json_mode=True)
        request = received[0]
        self.assertEqual(str(request.url), "https://example.test/v1/chat/completions")
        self.assertEqual(request.headers["authorization"], "Bearer test-secret")
        body = json.loads(request.content)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertFalse(body["enable_thinking"])
        self.assertFalse(body["stream"])
        self.assertEqual(body["max_tokens"], 100)
        self.assertEqual(response.model, "actual-version")
        self.assertEqual(response.usage.total_tokens, 26)
        self.assertEqual(response.usage.reasoning_tokens, 2)
        self.assertEqual(response.usage.cache_read_tokens, 5)
        self.assertEqual(response.request_id, "req-1")
        self.assertEqual(response.endpoint, "https://example.test/v1/chat/completions")
        self.assertEqual(response.config["max_retries"], 0)
        self.assertNotIn("test-secret", json.dumps(response.to_dict()))

    def test_claude_system_blocks_and_cache_usage(self):
        received = []

        def handle(request):
            received.append(request)
            return httpx.Response(200, json={
                "id": "msg-1", "model": "claude-version",
                "content": [{"type": "thinking", "thinking": "internal"},
                            {"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 2,
                          "cache_read_input_tokens": 20, "cache_creation_input_tokens": 5}},
                headers={"request-id": "claude-req"})

        with LLMClient(self.config("anthropic"), transport=httpx.MockTransport(handle)) as client:
            response = client.generate(MESSAGES)
        request = received[0]
        body = json.loads(request.content)
        self.assertEqual(str(request.url), "https://example.test/v1/messages")
        self.assertEqual(request.headers["x-api-key"], "test-secret")
        self.assertEqual(request.headers["anthropic-version"], "2023-06-01")
        self.assertEqual(body["system"], MESSAGES[0]["content"])
        self.assertEqual(body["messages"], MESSAGES[1:])
        self.assertNotIn("temperature", body)
        self.assertEqual(response.text, "AB")
        self.assertEqual(response.usage.input_tokens, 35)
        self.assertEqual(response.usage.total_tokens, 37)
        self.assertEqual(response.usage.cache_write_tokens, 5)
        self.assertEqual(response.request_id, "claude-req")

    def test_deepseek_cache_and_provider_options(self):
        def handle(request):
            self.assertEqual(json.loads(request.content)["thinking"], {"type": "disabled"})
            return compatible_response(usage={"prompt_tokens": 20, "completion_tokens": 6,
                                               "prompt_cache_hit_tokens": 8})

        with LLMClient(self.config(), transport=httpx.MockTransport(handle)) as client:
            response = client.generate(MESSAGES, provider_options={"thinking": {"type": "disabled"}})
        self.assertEqual(response.usage.cache_read_tokens, 8)
        self.assertEqual(response.usage.total_tokens, 26)

    @patch("adgenius.shared.llm._backoff")
    def test_transient_error_is_retried(self, backoff):
        calls = []

        def handle(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(429, headers={"retry-after": "3"})
            return compatible_response()

        with LLMClient(self.config(max_retries=2), transport=httpx.MockTransport(handle)) as client:
            response = client.generate(MESSAGES)
        self.assertEqual(response.attempts, 2)
        backoff.assert_called_once_with(1, "3")

    @patch("adgenius.shared.llm._backoff")
    def test_connect_failure_exhausts_finite_retries(self, backoff):
        def handle(request):
            raise httpx.ConnectTimeout("timed out", request=request)

        with LLMClient(self.config(max_retries=1), transport=httpx.MockTransport(handle)) as client:
            with self.assertRaises(LLMError) as caught:
                client.generate(MESSAGES)
        self.assertEqual(caught.exception.attempts, 2)
        backoff.assert_called_once()

    @patch("adgenius.shared.llm._backoff")
    def test_read_timeout_is_not_retried_when_server_may_have_completed(self, backoff):
        def handle(request):
            raise httpx.ReadTimeout("timed out", request=request)

        with LLMClient(self.config(max_retries=2), transport=httpx.MockTransport(handle)) as client:
            with self.assertRaises(LLMError) as caught:
                client.generate(MESSAGES)
        self.assertEqual(caught.exception.attempts, 1)
        self.assertTrue(caught.exception.may_have_completed)
        backoff.assert_not_called()

    @patch("adgenius.shared.llm._backoff")
    def test_auth_error_is_not_retried_or_leaked(self, backoff):
        def handle(request):
            return httpx.Response(401, json={"error": "test-secret"})

        with LLMClient(self.config(max_retries=2), transport=httpx.MockTransport(handle)) as client:
            with self.assertRaises(LLMError) as caught:
                client.generate(MESSAGES)
        self.assertEqual(caught.exception.status_code, 401)
        self.assertEqual(caught.exception.attempts, 1)
        self.assertNotIn("test-secret", str(caught.exception))
        self.assertEqual(caught.exception.raw_response, {"error": "[REDACTED]"})
        backoff.assert_not_called()

    def test_invalid_generated_json_is_returned_without_retry(self):
        calls = []

        def handle(request):
            calls.append(request)
            return compatible_response()

        with LLMClient(self.config(), transport=httpx.MockTransport(handle)) as client:
            response = client.generate(MESSAGES, json_mode=True)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(response.text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(response.raw_response["id"], "chat-1")

    def test_missing_usage_is_unknown_instead_of_zero(self):
        with LLMClient(self.config(), transport=httpx.MockTransport(
                lambda _: compatible_response(usage={}))) as client:
            response = client.generate(MESSAGES)
        self.assertIsNone(response.usage.input_tokens)
        self.assertIsNone(response.usage.output_tokens)
        self.assertIsNone(response.usage.total_tokens)

    def test_null_content_is_preserved_for_inspection(self):
        with LLMClient(self.config(), transport=httpx.MockTransport(lambda _: compatible_response(
                choices=[{"message": {"content": None, "refusal": "refused"},
                          "finish_reason": "stop"}]))) as client:
            response = client.generate(MESSAGES)
        self.assertEqual(response.text, "")
        self.assertTrue(response.content_missing)
        self.assertEqual(response.raw_response["choices"][0]["message"]["refusal"], "refused")

    def test_malformed_envelope_is_not_retried(self):
        calls = []

        def handle(request):
            calls.append(request)
            return httpx.Response(200, json={"choices": []})

        with LLMClient(self.config(max_retries=2), transport=httpx.MockTransport(handle)) as client:
            with self.assertRaises(LLMError) as caught:
                client.generate(MESSAGES)
        self.assertEqual(len(calls), 1)
        self.assertEqual(caught.exception.raw_response, {"choices": []})

    def test_reject_unsupported_or_conflicting_inputs_before_request(self):
        def handle(request):
            self.fail("Invalid inputs must not reach the network")

        cases = [dict(provider_options={"stream": True}),
                 dict(provider_options={"model": "other"}),
                 dict(provider_options={"n": 2}),
                 dict(max_output_tokens=0), dict(temperature=float("nan")),
                 dict(json_mode=True, provider_options={"response_format": {}})]
        with LLMClient(self.config(), transport=httpx.MockTransport(handle)) as client:
            for kwargs in cases:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    client.generate(MESSAGES, **kwargs)
            with self.assertRaises(ValueError):
                client.generate([{"role": "user", "content": "hi"}, MESSAGES[0]])
        with LLMClient(self.config("anthropic"), transport=httpx.MockTransport(handle)) as client:
            with self.assertRaises(ValueError):
                client.generate(MESSAGES, json_mode=True)

    def test_mock_does_not_require_key_or_transport(self):
        def handle(request):
            self.fail("Mock must not reach the network")

        with LLMClient(ModelConfig(provider="mock", model="mock"),
                       transport=httpx.MockTransport(handle)) as client:
            response = client.generate(MESSAGES, json_mode=True)
        self.assertEqual(json.loads(response.text), {"mock": True})
        self.assertEqual(response.attempts, 0)
        self.assertIsNone(response.usage.total_tokens)

    def test_toml_base_url_can_use_workspace_endpoint_from_environment(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "models.toml")
        with patch.dict(os.environ, {"DASHSCOPE_BASE_URL":
                                     "https://llm-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"}):
            config = ModelConfig.from_toml(config_path, "qwen")
        self.assertEqual(config.base_url,
                         "https://llm-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(config.snapshot()["base_url_env"], "DASHSCOPE_BASE_URL")

    def test_blank_optional_base_url_environment_uses_config_default(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "models.toml")
        with patch.dict(os.environ, {"DASHSCOPE_BASE_URL": "  "}):
            config = ModelConfig.from_toml(config_path, "qwen")
        self.assertEqual(config.base_url,
                         "https://dashscope.aliyuncs.com/compatible-mode/v1")


if __name__ == "__main__":
    unittest.main()
