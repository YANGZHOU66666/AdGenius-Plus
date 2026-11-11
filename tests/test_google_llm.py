"""Google generateContent contract tests using only a fake HTTP transport."""

import copy
import json
import os
import unittest
from unittest.mock import patch

import httpx

from adgenius.shared.llm import LLMClient, LLMError, ModelConfig


MESSAGES = [{"role": "user", "content": "Write an ad for a metal stapler."}]


def google_response(**changes):
    body = {
        "modelVersion": "gemma-4-31b-it-version-1",
        "responseId": "google-response-1",
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "A metal stapler."}]},
            "finishReason": "STOP",
        }],
    }
    body.update(changes)
    return httpx.Response(200, json=body)


class GoogleLLMTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"GOOGLE_LLM_TEST_KEY": "google-test-secret"})
        env.start()
        self.addCleanup(env.stop)

    def config(self, **changes):
        values = dict(provider="google_generative", model="gemma-4-31b-it",
                      base_url="https://generativelanguage.googleapis.com/v1beta",
                      api_key_env="GOOGLE_LLM_TEST_KEY", max_retries=0)
        values.update(changes)
        return ModelConfig(**values)

    def test_multiturn_mapping_and_public_text_usage_and_provenance(self):
        received = []
        messages = [
            {"role": "system", "content": "Write factual ads."},
            {"role": "system", "content": "Use one sentence."},
            {"role": "user", "content": "A metal stapler."},
            {"role": "assistant", "content": "A stapler for paperwork."},
            {"role": "user", "content": "Mention the metal body."},
        ]

        def handle(request):
            received.append(request)
            return google_response(
                candidates=[{
                    "content": {"role": "model", "parts": [
                        {"text": "Consider the product facts.", "thought": True},
                        {"text": "Keep papers together "},
                        {"text": "with this metal stapler.", "thought": False},
                    ]},
                    "finishReason": "STOP",
                }],
                usageMetadata={"promptTokenCount": 40, "candidatesTokenCount": 12,
                               "thoughtsTokenCount": 8, "totalTokenCount": 60,
                               "cachedContentTokenCount": 10},
            )

        with LLMClient(self.config(), transport=httpx.MockTransport(handle)) as client:
            response = client.generate(messages, max_output_tokens=120, temperature=0.2)

        request = received[0]
        body = json.loads(request.content)
        self.assertEqual(str(request.url),
                         "https://generativelanguage.googleapis.com/v1beta/models/"
                         "gemma-4-31b-it:generateContent")
        self.assertEqual(request.headers["x-goog-api-key"], "google-test-secret")
        self.assertNotIn("authorization", request.headers)
        self.assertEqual(body["systemInstruction"],
                         {"parts": [{"text": "Write factual ads.\n\nUse one sentence."}]})
        self.assertEqual(body["contents"], [
            {"role": "user", "parts": [{"text": "A metal stapler."}]},
            {"role": "model", "parts": [{"text": "A stapler for paperwork."}]},
            {"role": "user", "parts": [{"text": "Mention the metal body."}]},
        ])
        self.assertEqual(body["generationConfig"],
                         {"maxOutputTokens": 120, "temperature": 0.2})
        self.assertEqual(response.text, "Keep papers together with this metal stapler.")
        self.assertEqual(response.requested_model, "gemma-4-31b-it")
        self.assertEqual(response.model, "gemma-4-31b-it-version-1")
        self.assertEqual(response.request_id, "google-response-1")
        self.assertEqual(response.usage.input_tokens, 40)
        self.assertEqual(response.usage.output_tokens, 20)
        self.assertEqual(response.usage.reasoning_tokens, 8)
        self.assertEqual(response.usage.cache_read_tokens, 10)
        self.assertEqual(response.usage.total_tokens, 60)
        self.assertFalse(response.content_missing)
        self.assertNotIn("google-test-secret", json.dumps(response.to_dict()))

    def test_prompt_block_without_candidates_is_an_empty_response(self):
        def handle(_request):
            return httpx.Response(200, json={
                "promptFeedback": {"blockReason": "SAFETY"},
                "usageMetadata": {"promptTokenCount": 20, "totalTokenCount": 20},
                "responseId": "blocked-response",
            })

        with LLMClient(self.config(), transport=httpx.MockTransport(handle)) as client:
            response = client.generate(MESSAGES)
        self.assertEqual(response.text, "")
        self.assertTrue(response.content_missing)
        self.assertEqual(response.finish_reason, "SAFETY")
        self.assertEqual(response.request_id, "blocked-response")
        self.assertIsNone(response.usage.output_tokens)
        self.assertEqual(response.usage.input_tokens, 20)

    def test_http_error_redacts_google_key_and_retains_request_id(self):
        def handle(request):
            self.assertEqual(request.headers["x-goog-api-key"], "google-test-secret")
            return httpx.Response(400, json={
                "error": {"message": "Bad request with google-test-secret"},
            }, headers={"x-goog-request-id": "invalid-google-request"})

        with LLMClient(self.config(), transport=httpx.MockTransport(handle)) as client:
            with self.assertRaises(LLMError) as caught:
                client.generate(MESSAGES)
        error = caught.exception
        self.assertEqual(error.status_code, 400)
        self.assertEqual(error.request_id, "invalid-google-request")
        self.assertEqual(error.attempts, 1)
        self.assertNotIn("google-test-secret", str(error))
        self.assertNotIn("google-test-secret", json.dumps(error.raw_response))
        self.assertIn("[REDACTED]", json.dumps(error.raw_response))

    def test_native_generation_options_and_json_mode_preserve_inputs(self):
        options = {"generationConfig": {
            "thinkingConfig": {"thinkingLevel": "minimal"},
            "responseSchema": {"type": "OBJECT", "properties": {"ad": {"type": "STRING"}}},
        }}
        original = copy.deepcopy(options)
        received = []

        def handle(request):
            received.append(json.loads(request.content))
            return google_response()

        with LLMClient(self.config(), transport=httpx.MockTransport(handle)) as client:
            client.generate(MESSAGES, max_output_tokens=99, json_mode=True,
                            provider_options=options)
        config = received[0]["generationConfig"]
        self.assertEqual(config["thinkingConfig"], {"thinkingLevel": "minimal"})
        self.assertEqual(config["responseSchema"], options["generationConfig"]["responseSchema"])
        self.assertEqual(config["maxOutputTokens"], 99)
        self.assertEqual(config["responseMimeType"], "application/json")
        self.assertEqual(options, original)

    def test_core_overrides_and_conflicting_json_mode_rejected_before_request(self):
        def handle(_request):
            self.fail("Invalid parameters must not be sent to Google")

        cases = [
            {"provider_options": {"generationConfig": {"maxOutputTokens": 1}}},
            {"provider_options": {"generationConfig": {"temperature": 0.5}}},
            {"provider_options": {"generationConfig": {"candidateCount": 2}}},
            {"provider_options": {"contents": []}},
            {"provider_options": {"systemInstruction": {"parts": []}}},
            {"provider_options": {"generationConfig": []}},
            {"json_mode": True, "provider_options": {
                "generationConfig": {"responseMimeType": "text/plain"}}},
        ]
        with LLMClient(self.config(), transport=httpx.MockTransport(handle)) as client:
            for kwargs in cases:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    client.generate(MESSAGES, **kwargs)

    def test_truncation_finish_reason_preserved_for_caller(self):
        def handle(_request):
            return google_response(candidates=[{
                "content": {"role": "model", "parts": [{"text": "A metal stapler for"}]},
                "finishReason": "MAX_TOKENS",
            }])

        with LLMClient(self.config(), transport=httpx.MockTransport(handle)) as client:
            response = client.generate(MESSAGES)
        self.assertEqual(response.finish_reason, "MAX_TOKENS")
        self.assertEqual(response.text, "A metal stapler for")
        self.assertFalse(response.content_missing)
        self.assertIsNone(response.usage.total_tokens)


if __name__ == "__main__":
    unittest.main()
