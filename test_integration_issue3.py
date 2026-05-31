"""End-to-end regression test for issue #3.

Symptom: after selecting the ``gemini-3.5-flash`` model and sending a message,
Claude Code reported "There's an issue with the selected model
(gemini-3.5-flash). It may not exist or you may not have access to it."

Root cause: the adapter only handled ``GET /v1/models`` and ``POST /v1/messages``
and answered 404 for everything else. Claude Code validates the selected model
with ``GET /v1/models/{id}`` and counts tokens with
``POST /v1/messages/count_tokens`` before sending; those 404s are surfaced as
the model being unavailable.

This test runs the real anthropic2openai adapter (and, for the chat path, the
real gemini-web2api backend with only the upstream Gemini call mocked) and
exercises the exact endpoints Claude Code hits.

Run with:  python3 test_integration_issue3.py
"""
import json
import os
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

import anthropic2openai as adapter

_BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini-web2api")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from gemini_web2api import server as backend_server  # noqa: E402

BACKEND_PORT = 8081  # adapter forwards here (see anthropic2openai.GEMINI_API_URL)
ADAPTER_PORT = adapter.LISTEN_PORT  # 8082


def _fake_generate(prompt, model_id, think_mode, file_refs=None, extra_fields=None):
    return "Привет! Чем могу помочь?"


def _fake_generate_stream(prompt, model_id, think_mode, file_refs=None, extra_fields=None):
    for piece in ["При", "вет", "!"]:
        yield piece


class TestIssue3EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock only the upstream Gemini call; everything else is the real code.
        cls._orig_generate = backend_server.generate
        cls._orig_generate_stream = backend_server.generate_stream
        backend_server.generate = _fake_generate
        backend_server.generate_stream = _fake_generate_stream

        cls.backend = backend_server.ThreadedServer(
            ("127.0.0.1", BACKEND_PORT), backend_server.GeminiHandler)
        cls.backend_thread = threading.Thread(target=cls.backend.serve_forever, daemon=True)
        cls.backend_thread.start()

        from http.server import HTTPServer
        cls.adapter = HTTPServer(("127.0.0.1", ADAPTER_PORT), adapter.AnthropicProxyHandler)
        cls.adapter_thread = threading.Thread(target=cls.adapter.serve_forever, daemon=True)
        cls.adapter_thread.start()
        time.sleep(0.3)  # let both servers bind

    @classmethod
    def tearDownClass(cls):
        cls.backend.shutdown()
        cls.adapter.shutdown()
        backend_server.generate = cls._orig_generate
        backend_server.generate_stream = cls._orig_generate_stream

    def _request(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{ADAPTER_PORT}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    # --- The endpoints that triggered the issue (used to 404) ---------------

    def test_retrieve_selected_model(self):
        # Exact scenario from issue #3: Claude Code validates the selected model.
        status, raw = self._request("GET", "/v1/models/gemini-3.5-flash")
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw)["id"], "gemini-3.5-flash")

    def test_retrieve_arbitrary_model(self):
        status, raw = self._request("GET", "/v1/models/claude-opus-4-7")
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw)["id"], "claude-opus-4-7")

    def test_count_tokens(self):
        status, raw = self._request("POST", "/v1/messages/count_tokens", {
            "model": "gemini-3.5-flash",
            "messages": [{"role": "user", "content": "Привет"}],
        })
        self.assertEqual(status, 200, raw)
        self.assertIsInstance(json.loads(raw)["input_tokens"], int)

    def test_models_list(self):
        status, raw = self._request("GET", "/v1/models")
        self.assertEqual(status, 200, raw)
        ids = [m["id"] for m in json.loads(raw)["data"]]
        self.assertIn("gemini-3.5-flash", ids)

    # --- The actual chat path still works -----------------------------------

    def test_non_stream_message(self):
        status, raw = self._request("POST", "/v1/messages", {
            "model": "gemini-3.5-flash",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Привет"}],
        })
        self.assertEqual(status, 200, raw)
        data = json.loads(raw)
        self.assertEqual(data["type"], "message")
        text = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        self.assertIn("Привет", text)

    def test_stream_message(self):
        status, raw = self._request("POST", "/v1/messages", {
            "model": "gemini-3.5-flash",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "Привет"}],
        })
        self.assertEqual(status, 200, raw)
        self.assertIn("message_start", raw)
        self.assertIn("Привет", raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
