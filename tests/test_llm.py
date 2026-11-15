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

    def test_openai_modern_token_limit_and_reasoning_options(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "models.toml")
        config = ModelConfig.from_toml(config_path, "gpt6_luna")

        def handle(request):
            self.assertEqual(str(request.url), "https://api.openai.com/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer openai-test-secret")
            body = json.loads(request.content)
            self.assertEqual(body["model"], "gpt-6-luna")
            self.assertEqual(body["max_completion_tokens"], 100)
            self.assertNotIn("max_tokens", body)
            self.assertEqual(body["reasoning_effort"], "none")
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["response_format"], {"type": "json_object"})
            return compatible_response()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-test-secret"}), \
             LLMClient(config, transport=httpx.MockTransport(handle)) as client:
            response = client.generate(MESSAGES, max_output_tokens=100, temperature=0,
                                       json_mode=True)
            for field in ("max_tokens", "max_completion_tokens"):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    client.generate(MESSAGES, provider_options={field: 999})
        self.assertEqual(response.config["max_tokens_parameter"], "max_completion_tokens")
        self.assertNotIn("openai-test-secret", json.dumps(response.to_dict()))

    def test_token_parameter_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "max_tokens_parameter"):
            self.config(max_tokens_parameter="max_output_tokens")
        with self.assertRaisesRegex(ValueError, "OpenAI-compatible"):
            self.config("anthropic", max_tokens_parameter="max_completion_tokens")

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

    def test_candidate_model_configs_use_openai_compatible_routes(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "models.toml")
        expected = {
            "qwen": ("qwen3.7-flash", "DASHSCOPE_API_KEY",
                     "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
            "mistral": ("mistral-small-2603", "MISTRAL_API_KEY",
                        "https://api.mistral.ai/v1/chat/completions"),
            "glm53_flash": ("glm-5.3-flash", "ZAI_API_KEY",
                            "https://open.bigmodel.cn/api/paas/v4/chat/completions"),
            "longcat": ("LongCat-2.5", "LONGCAT_API_KEY",
                        "https://api.longcat.chat/openai/v1/chat/completions"),
        }
        for alias, (model, key_env, endpoint) in expected.items():
            with self.subTest(alias=alias):
                config = ModelConfig.from_toml(config_path, alias)
                self.assertEqual(config.model, model)
                self.assertEqual(config.api_key_env, key_env)
                self.assertEqual(f"{config.base_url}/chat/completions", endpoint)

    def test_google_generative_model_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "models.toml")
        config = ModelConfig.from_toml(config_path, "gemma_31b")
        self.assertEqual(config.provider, "google_generative")
        self.assertEqual(config.model, "gemma-4-31b-it")
        self.assertEqual(config.api_key_env, "GEMINI_API_KEY")
        self.assertEqual(config.base_url,
                         "https://generativelanguage.googleapis.com/v1beta")

    def test_local_openai_compatible_config_does_not_require_a_key(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "models.toml")
        config = ModelConfig.from_toml(config_path, "qwen_local")
        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.model, "qwen3.5:2b")
        self.assertEqual(config.api_key_env, "")
        self.assertEqual(f"{config.base_url}/api/chat", "http://127.0.0.1:11434/api/chat")

    def test_local_provider_rejects_non_loopback_http(self):
        with self.assertRaises(ValueError):
            ModelConfig(provider="openai_compatible_local", model="local",
                        base_url="http://example.test/v1")

    def test_local_provider_can_call_without_authorization(self):
        received = []

        def handle(request):
            received.append(request)
            return compatible_response()

        config = ModelConfig(provider="openai_compatible_local", model="local",
                             base_url="http://127.0.0.1:11434/v1", max_retries=0)
        with LLMClient(config, transport=httpx.MockTransport(handle)) as client:
            response = client.generate(MESSAGES)
        self.assertEqual(response.text, "Keep your papers together.")
        self.assertNotIn("authorization", received[0].headers)

    def test_ollama_native_payload_and_response(self):
        received = []

        def handle(request):
            received.append(request)
            return httpx.Response(200, json={
                "model": "qwen3.5:2b",
                "message": {"role": "assistant", "content": "A sturdy stapler."},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 4,
            })

        config = ModelConfig(provider="ollama", model="qwen3.5:2b",
                             base_url="http://127.0.0.1:11434", max_retries=0,
                             request_options={"think": False})
        with LLMClient(config, transport=httpx.MockTransport(handle)) as client:
            response = client.generate(MESSAGES, max_output_tokens=64, temperature=0)
        body = json.loads(received[0].content)
        self.assertEqual(str(received[0].url), "http://127.0.0.1:11434/api/chat")
        self.assertEqual(body["options"], {"num_predict": 64, "temperature": 0})
        self.assertFalse(body["think"])
        self.assertEqual(response.text, "A sturdy stapler.")
        self.assertEqual(response.usage.total_tokens, 16)

    def test_google_generative_native_payload_usage_and_key_redaction(self):
        received = []

        def handle(request):
            received.append(request)
            return httpx.Response(200, json={
                "modelVersion": "gemma-4-31b-it",
                "responseId": "google-response-1",
                "candidates": [{
                    "content": {"role": "model", "parts": [{"text": "A clear product ad."}]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {
                    "promptTokenCount": 31,
                    "candidatesTokenCount": 7,
                    "totalTokenCount": 38,
                },
            }, headers={"x-goog-request-id": "google-request-1"})

        config = ModelConfig(provider="google_generative", model="gemma-4-31b-it",
                             base_url="https://generativelanguage.googleapis.com/v1beta",
                             api_key_env="LLM_TEST_KEY", max_retries=0)
        with LLMClient(config, transport=httpx.MockTransport(handle)) as client:
            response = client.generate(MESSAGES, max_output_tokens=120,
                                       temperature=0.2, json_mode=False)
        request = received[0]
        body = json.loads(request.content)
        self.assertEqual(str(request.url),
                         "https://generativelanguage.googleapis.com/v1beta/models/"
                         "gemma-4-31b-it:generateContent")
        self.assertEqual(request.headers["x-goog-api-key"], "test-secret")
        self.assertNotIn("authorization", request.headers)
        self.assertEqual(body["systemInstruction"],
                         {"parts": [{"text": "Write accurate copy."}]})
        self.assertEqual(body["contents"], [
            {"role": "user", "parts": [{"text": "A blue stapler."}]},
        ])
        self.assertEqual(body["generationConfig"],
                         {"maxOutputTokens": 120, "temperature": 0.2})
        self.assertEqual(response.text, "A clear product ad.")
        self.assertEqual(response.provider, "google_generative")
        self.assertEqual(response.model, "gemma-4-31b-it")
        self.assertEqual(response.usage.input_tokens, 31)
        self.assertEqual(response.usage.output_tokens, 7)
        self.assertEqual(response.usage.total_tokens, 38)
        self.assertEqual(response.finish_reason, "STOP")
        self.assertEqual(response.request_id, "google-request-1")
        self.assertNotIn("test-secret", json.dumps(response.to_dict()))


if __name__ == "__main__":
    unittest.main()
