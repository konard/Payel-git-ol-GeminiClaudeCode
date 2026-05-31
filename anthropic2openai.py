import json
import urllib.request
import urllib.error
import http.server
import uuid
import time
import re
import sys
from urllib.parse import urlparse

GEMINI_API_URL = "http://localhost:8081/v1"
GEMINI_API_KEY = "sk-gemini"
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8082

MODEL_MAP = {
    "claude-opus-4-7": "gemini-3.5-flash-thinking",
    "claude-sonnet-4-7": "gemini-3.5-flash",
    "claude-sonnet-4-6": "gemini-3.5-flash",
    "claude-haiku-4-5": "gemini-flash-lite",
}

AVAILABLE_MODELS = [
    {
        "type": "model",
        "id": "gemini-3.5-flash-thinking",
        "display_name": "Gemini Flash Thinking (Opus)",
        "created": 1700000000,
    },
    {
        "type": "model",
        "id": "gemini-3.5-flash",
        "display_name": "Gemini Flash (Sonnet)",
        "created": 1700000000,
    },
    {
        "type": "model",
        "id": "gemini-flash-lite",
        "display_name": "Gemini Flash Lite (Haiku)",
        "created": 1700000000,
    },
]


def resolve_model(model: str) -> str:
    return MODEL_MAP.get(model, model)


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            c.get("text", "") for c in content
            if isinstance(c, dict) and c.get("type") in ("text", "input_text")
        )
    return str(content) if content else ""


# Map Anthropic stop_reason <-> OpenAI finish_reason
_FINISH_TO_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


def convert_tools(tools) -> list:
    """Anthropic tool definitions -> OpenAI function tools."""
    out = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        # Already in OpenAI shape (defensive passthrough).
        if t.get("type") == "function" and "function" in t:
            out.append(t)
            continue
        name = t.get("name")
        if not name:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return out


def convert_tool_choice(choice):
    """Anthropic tool_choice -> OpenAI tool_choice."""
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def anthropic_messages_to_openai(body: dict) -> dict:
    msgs = []
    system = extract_text(body.get("system", ""))

    for m in body.get("messages", []):
        role = m.get("role")
        content = m.get("content")

        if role == "system":
            system += "\n" + extract_text(content)
            continue

        # Plain string content: forward as-is.
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            msgs.append({"role": role, "content": extract_text(content)})
            continue

        if role == "assistant":
            text_parts = []
            tool_calls = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype in ("text", "input_text"):
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
            msg = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            msgs.append(msg)
            continue

        # role == "user" (or anything else): tool_result blocks become tool
        # messages; remaining text becomes a user message.
        text_parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_result":
                msgs.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": extract_text(block.get("content", "")) or "",
                })
            elif btype in ("text", "input_text"):
                text_parts.append(block.get("text", ""))
        if text_parts:
            msgs.append({"role": "user", "content": "".join(text_parts)})

    openai_body = {
        "model": resolve_model(body.get("model", "gemini-3.5-flash")),
        "messages": msgs,
        "stream": body.get("stream", False),
        "max_tokens": body.get("max_tokens", 4096),
    }
    if body.get("temperature"):
        openai_body["temperature"] = body["temperature"]
    if body.get("top_p"):
        openai_body["top_p"] = body["top_p"]
    tools = convert_tools(body.get("tools"))
    if tools:
        openai_body["tools"] = tools
        tc = convert_tool_choice(body.get("tool_choice"))
        if tc is not None:
            openai_body["tool_choice"] = tc
    if system.strip():
        openai_body["messages"].insert(
            0, {"role": "system", "content": system.strip()}
        )
    return openai_body


def openai_response_to_anthropic(resp: dict) -> dict:
    choice = resp["choices"][0]
    message = choice.get("message", {})
    content_blocks = []
    if message.get("content"):
        content_blocks.append({"type": "text", "text": message["content"]})
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            tool_input = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            tool_input = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
            "name": fn.get("name", ""),
            "input": tool_input,
        })
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    finish = choice.get("finish_reason", "stop")
    if message.get("tool_calls"):
        finish = "tool_calls"
    stop_reason = _FINISH_TO_STOP.get(finish, "end_turn")

    usage = resp.get("usage", {}) or {}
    return {
        "id": resp.get("id", f"msg_{uuid.uuid4().hex[:12]}").replace("chatcmpl", "msg"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": resp.get("model", ""),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def proxy_non_stream(openai_body: dict) -> tuple:
    data = json.dumps(openai_body).encode()
    req = urllib.request.Request(
        f"{GEMINI_API_URL}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GEMINI_API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return json.dumps({"error": {"message": e.read().decode(), "type": "api_error"}}).encode(), e.code
    anthropic = openai_response_to_anthropic(result)
    return json.dumps(anthropic).encode(), 200


def proxy_stream(openai_body: dict):
    openai_body["stream"] = True
    data = json.dumps(openai_body).encode()
    req = urllib.request.Request(
        f"{GEMINI_API_URL}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GEMINI_API_KEY}",
        },
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=180)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        yield f"event: error\ndata: {json.dumps({'error': {'message': error_body}})}\n\n"
        return

    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    model_name = openai_body["model"]

    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start','message': {'id': msg_id,'type': 'message','role': 'assistant','content': [],'model': model_name,'stop_reason': None,'stop_sequence': None,'usage': {'input_tokens': 0,'output_tokens': 0}}})}\n\n"

    text_block_open = False
    text_index = None
    next_index = 0
    full_text = ""
    # Accumulate tool calls by their OpenAI streaming index.
    tool_calls = {}
    finish_reason = "stop"
    usage_out = 0

    for line in resp:
        line = line.decode().strip()
        if not line or line == "data: [DONE]":
            continue
        if not line.startswith("data: "):
            continue
        try:
            chunk = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or [{}]
        choice = choices[0]
        delta = choice.get("delta", {}) or {}

        text = delta.get("content")
        if text:
            if not text_block_open:
                text_index = next_index
                next_index += 1
                text_block_open = True
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start','index': text_index,'content_block': {'type': 'text','text': ''}})}\n\n"
            full_text += text
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta','index': text_index,'delta': {'type': 'text_delta','text': text}})}\n\n"

        for i, tc in enumerate(delta.get("tool_calls") or []):
            idx = tc.get("index", i)
            acc = tool_calls.setdefault(idx, {"id": None, "name": None, "arguments": ""})
            if tc.get("id"):
                acc["id"] = tc["id"]
            fn = tc.get("function", {}) or {}
            if fn.get("name"):
                acc["name"] = fn["name"]
            if fn.get("arguments"):
                acc["arguments"] += fn["arguments"]

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        cu = chunk.get("usage") or {}
        if cu.get("completion_tokens"):
            usage_out = cu["completion_tokens"]

    if text_block_open:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop','index': text_index})}\n\n"

    for idx in sorted(tool_calls):
        acc = tool_calls[idx]
        block_index = next_index
        next_index += 1
        tool_id = acc["id"] or f"toolu_{uuid.uuid4().hex[:12]}"
        args = acc["arguments"] or "{}"
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start','index': block_index,'content_block': {'type': 'tool_use','id': tool_id,'name': acc['name'] or '','input': {}}})}\n\n"
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta','index': block_index,'delta': {'type': 'input_json_delta','partial_json': args}})}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop','index': block_index})}\n\n"

    if tool_calls:
        finish_reason = "tool_calls"
    stop_reason = _FINISH_TO_STOP.get(finish_reason, "end_turn")
    if not usage_out:
        usage_out = len(full_text.split())
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta','delta': {'stop_reason': stop_reason,'stop_sequence': None},'usage': {'output_tokens': usage_out}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


class AnthropicProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": AVAILABLE_MODELS}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/v1/messages":
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len))
            openai_body = anthropic_messages_to_openai(body)

            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    for chunk in proxy_stream(openai_body):
                        try:
                            self.wfile.write(chunk.encode())
                            self.wfile.flush()
                        except BrokenPipeError:
                            break
                finally:
                    self.close_connection = True
            else:
                data, status = proxy_non_stream(openai_body)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        try:
            sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")
        except (TypeError, IndexError):
            sys.stderr.write(f"[{self.log_date_time_string()}] {format} {args}\n")


def main():
    server = http.server.HTTPServer((LISTEN_HOST, LISTEN_PORT), AnthropicProxyHandler)
    print(f"[*] Anthropic->OpenAI adapter running on http://{LISTEN_HOST}:{LISTEN_PORT}")
    print(f"[*] Point Claude Code: ANTHROPIC_BASE_URL=http://{LISTEN_HOST}:{LISTEN_PORT}")
    print(f"[*] Proxying to gemini-web2api at {GEMINI_API_URL}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
