"""End-to-end regression test for issue #3.

Runs the *real* gemini-web2api backend and the anthropic2openai adapter in
threads (mocking only the upstream Gemini call) and sends an Anthropic
/v1/messages request exactly like Claude Code does when the user has selected
the ``gemini-3.5-flash`` model.

Before the fix the backend answered ``model 'gemini-3.5-flash' not found``
(HTTP 400) and Claude Code reported the model as unavailable. After the fix the
adapter translates the advertised name to a real backend model and the request
succeeds.

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
    # Echo the model id back so the test can assert which backend model was hit.
    return f"Привет! (model_id={model_id})"


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

    def _post_messages(self, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{ADAPTER_PORT}/v1/messages",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def test_non_stream_selected_gemini_flash(self):
        status, raw = self._post_messages({
            "model": "gemini-3.5-flash",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Привет"}],
        })
        self.assertEqual(status, 200, raw)
        data = json.loads(raw)
        self.assertEqual(data["type"], "message")
        text = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        self.assertIn("Привет", text)
        # The real backend model must have been a valid one (2.5 family).
        self.assertIn("gemini-2.5-flash", text)
        self.assertNotIn("not found", raw)

    def test_stream_selected_gemini_flash(self):
        status, raw = self._post_messages({
            "model": "gemini-3.5-flash",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "Привет"}],
        })
        self.assertEqual(status, 200, raw)
        self.assertIn("message_start", raw)
        self.assertIn("Привет", raw)
        self.assertNotIn("not found", raw)

    def test_models_endpoint_lists_selected_model(self):
        req = urllib.request.Request(f"http://127.0.0.1:{ADAPTER_PORT}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        ids = [m["id"] for m in data["data"]]
        self.assertIn("gemini-3.5-flash", ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
