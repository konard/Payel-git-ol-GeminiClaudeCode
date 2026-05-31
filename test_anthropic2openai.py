"""Unit tests for the Anthropic<->OpenAI translation in anthropic2openai.py.

Run with:  python3 test_anthropic2openai.py
"""
import json
import os
import sys
import unittest

import anthropic2openai as adapter

# Make the bundled gemini-web2api backend importable so we can assert the
# adapter only ever forwards model names the backend actually understands.
_BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini-web2api")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
from gemini_web2api.models import resolve_model as backend_resolve_model  # noqa: E402


class TestRequestConversion(unittest.TestCase):
    def test_simple_text_message(self):
        body = {
            "model": "claude-opus-4-7",
            "system": "you are helpful",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "max_tokens": 100,
        }
        out = adapter.anthropic_messages_to_openai(body)
        # claude model name is mapped to a gemini model
        self.assertEqual(out["model"], "gemini-3.5-flash-thinking")
        self.assertTrue(out["stream"])
        self.assertEqual(out["max_tokens"], 100)
        self.assertEqual(out["messages"][0], {"role": "system", "content": "you are helpful"})
        self.assertEqual(out["messages"][1], {"role": "user", "content": "hi"})

    def test_system_as_block_list(self):
        body = {
            "model": "gemini-3.5-flash",
            "system": [{"type": "text", "text": "block sys"}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "yo"}]}],
        }
        out = adapter.anthropic_messages_to_openai(body)
        self.assertEqual(out["messages"][0], {"role": "system", "content": "block sys"})
        self.assertEqual(out["messages"][1], {"role": "user", "content": "yo"})

    def test_tools_converted(self):
        body = {
            "model": "gemini-3.5-flash",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
            "tool_choice": {"type": "auto"},
        }
        out = adapter.anthropic_messages_to_openai(body)
        self.assertEqual(out["tools"][0]["type"], "function")
        fn = out["tools"][0]["function"]
        self.assertEqual(fn["name"], "get_weather")
        self.assertEqual(fn["parameters"]["properties"]["city"]["type"], "string")
        self.assertEqual(out["tool_choice"], "auto")

    def test_tool_choice_variants(self):
        self.assertEqual(adapter.convert_tool_choice({"type": "any"}), "required")
        self.assertEqual(adapter.convert_tool_choice({"type": "none"}), "none")
        self.assertEqual(
            adapter.convert_tool_choice({"type": "tool", "name": "x"}),
            {"type": "function", "function": {"name": "x"}},
        )

    def test_assistant_tool_use_and_tool_result(self):
        body = {
            "model": "gemini-3.5-flash",
            "messages": [
                {"role": "user", "content": "weather in Tokyo?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "let me check"},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                     "input": {"city": "Tokyo"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1",
                     "content": [{"type": "text", "text": "sunny"}]},
                ]},
            ],
        }
        out = adapter.anthropic_messages_to_openai(body)
        msgs = out["messages"]
        # user, assistant(with tool_calls), tool
        self.assertEqual(msgs[0], {"role": "user", "content": "weather in Tokyo?"})
        assistant = msgs[1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"], "let me check")
        self.assertEqual(assistant["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(assistant["tool_calls"][0]["function"]["arguments"]),
                         {"city": "Tokyo"})
        tool_msg = msgs[2]
        self.assertEqual(tool_msg, {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"})


class TestResponseConversion(unittest.TestCase):
    def test_text_response(self):
        resp = {
            "id": "chatcmpl-abc",
            "model": "gemini-3.5-flash",
            "choices": [{"message": {"role": "assistant", "content": "hello"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        out = adapter.openai_response_to_anthropic(resp)
        self.assertEqual(out["id"], "msg-abc")
        self.assertEqual(out["content"], [{"type": "text", "text": "hello"}])
        self.assertEqual(out["stop_reason"], "end_turn")
        self.assertEqual(out["usage"], {"input_tokens": 5, "output_tokens": 2})

    def test_tool_call_response(self):
        resp = {
            "id": "chatcmpl-xyz",
            "model": "gemini-3.5-flash",
            "choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "get_weather",
                                             "arguments": '{"city": "Tokyo"}'}}],
            }, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        out = adapter.openai_response_to_anthropic(resp)
        self.assertEqual(out["stop_reason"], "tool_use")
        block = out["content"][0]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["id"], "call_1")
        self.assertEqual(block["name"], "get_weather")
        self.assertEqual(block["input"], {"city": "Tokyo"})

    def test_text_and_tool_call(self):
        resp = {
            "id": "chatcmpl-1",
            "model": "m",
            "choices": [{"message": {
                "role": "assistant", "content": "checking",
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "f", "arguments": "{}"}}],
            }, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        out = adapter.openai_response_to_anthropic(resp)
        self.assertEqual(out["content"][0]["type"], "text")
        self.assertEqual(out["content"][1]["type"], "tool_use")


def _collect_stream(chunks):
    """Parse adapter SSE strings into (event, data) tuples."""
    events = []
    for raw in chunks:
        ev = None
        data = None
        for ln in raw.strip().split("\n"):
            if ln.startswith("event: "):
                ev = ln[7:]
            elif ln.startswith("data: "):
                data = json.loads(ln[6:])
        events.append((ev, data))
    return events


class TestStreamConversion(unittest.TestCase):
    """Patch the upstream urlopen to feed canned OpenAI SSE lines."""

    def _run_stream(self, openai_sse_lines):
        class FakeResp:
            def __init__(self, lines):
                self._lines = [l.encode() for l in lines]
            def __iter__(self):
                return iter(self._lines)

        orig = adapter.urllib.request.urlopen
        adapter.urllib.request.urlopen = lambda *a, **k: FakeResp(openai_sse_lines)
        try:
            return _collect_stream(list(adapter.proxy_stream({"model": "m"})))
        finally:
            adapter.urllib.request.urlopen = orig

    def test_text_stream(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}\n',
            'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            'data: [DONE]\n',
        ]
        events = self._run_stream(lines)
        types = [e for e, _ in events]
        self.assertEqual(types[0], "message_start")
        self.assertIn("content_block_start", types)
        deltas = [d for e, d in events if e == "content_block_delta"]
        text = "".join(d["delta"]["text"] for d in deltas)
        self.assertEqual(text, "Hello")
        last = events[-1]
        self.assertEqual(last[0], "message_stop")
        msg_delta = [d for e, d in events if e == "message_delta"][0]
        self.assertEqual(msg_delta["delta"]["stop_reason"], "end_turn")

    def test_tool_call_stream(self):
        # gemini-web2api emits the whole tool call in a single chunk.
        lines = [
            'data: {"choices":[{"delta":{"role":"assistant","content":null,'
            '"tool_calls":[{"id":"call_1","type":"function",'
            '"function":{"name":"get_weather","arguments":"{\\"city\\": \\"Tokyo\\"}"}}]},'
            '"finish_reason":"tool_calls"}]}\n',
            'data: [DONE]\n',
        ]
        events = self._run_stream(lines)
        types = [e for e, _ in events]
        self.assertIn("content_block_start", types)
        start = [d for e, d in events if e == "content_block_start"][0]
        self.assertEqual(start["content_block"]["type"], "tool_use")
        self.assertEqual(start["content_block"]["name"], "get_weather")
        json_delta = [d for e, d in events if e == "content_block_delta"][0]
        self.assertEqual(json_delta["delta"]["type"], "input_json_delta")
        self.assertEqual(json.loads(json_delta["delta"]["partial_json"]), {"city": "Tokyo"})
        msg_delta = [d for e, d in events if e == "message_delta"][0]
        self.assertEqual(msg_delta["delta"]["stop_reason"], "tool_use")


class TestModelMappingMatchesBackend(unittest.TestCase):
    """Regression for issue #3.

    Claude Code reported "the selected model (gemini-3.5-flash) may not exist"
    because the adapter advertised/forwarded model names (gemini-3.5-flash,
    gemini-flash-lite, ...) that the gemini-web2api backend does not know, so
    the backend answered ``model '...' not found`` (HTTP 400).

    Contract: every model the adapter advertises or maps to must resolve
    cleanly on the backend.
    """

    def _assert_backend_accepts(self, requested):
        forwarded = adapter.resolve_model(requested)
        _, _, _, err, _ = backend_resolve_model(forwarded)
        self.assertIsNone(
            err,
            f"backend rejected '{forwarded}' (resolved from '{requested}'): {err}",
        )

    def test_advertised_models_are_resolvable(self):
        self.assertTrue(adapter.AVAILABLE_MODELS)
        for model in adapter.AVAILABLE_MODELS:
            self._assert_backend_accepts(model["id"])

    def test_mapped_model_names_are_resolvable(self):
        for name in adapter.MODEL_MAP:
            self._assert_backend_accepts(name)

    def test_default_model_is_resolvable(self):
        # Default applied when a request omits the model field
        # (see anthropic_messages_to_openai).
        body = {"messages": [{"role": "user", "content": "hi"}]}
        out = adapter.anthropic_messages_to_openai(body)
        _, _, _, err, _ = backend_resolve_model(out["model"])
        self.assertIsNone(err, f"default model '{out['model']}' rejected: {err}")

    def test_issue_scenario_gemini_3_5_flash(self):
        # The exact model the user selected in issue #3.
        self._assert_backend_accepts("gemini-3.5-flash")


if __name__ == "__main__":
    unittest.main(verbosity=2)
