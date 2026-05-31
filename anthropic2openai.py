import json
import urllib.request
import urllib.error
import http.server
import uuid
import time
import re
import sys

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


def anthropic_messages_to_openai(body: dict) -> dict:
    msgs = []
    system = extract_text(body.get("system", ""))

    for m in body.get("messages", []):
        if m["role"] == "system":
            system += "\n" + extract_text(m["content"])
        else:
            msgs.append({"role": m["role"], "content": extract_text(m["content"])})

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
    if system.strip():
        openai_body["messages"].insert(
            0, {"role": "system", "content": system.strip()}
        )
    return openai_body


def openai_response_to_anthropic(resp: dict) -> dict:
    choice = resp["choices"][0]
    content_text = choice["message"]["content"]
    stop_reason = {"stop": "end_turn", "length": "max_tokens"}.get(
        choice.get("finish_reason", "stop"), "end_turn"
    )
    return {
        "id": resp["id"].replace("chatcmpl", "msg"),
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content_text}],
        "model": resp["model"],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": resp["usage"]["prompt_tokens"],
            "output_tokens": resp["usage"]["completion_tokens"],
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
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start','index': 0,'content_block': {'type': 'text','text': ''}})}\n\n"

    full_text = ""
    for line in resp:
        line = line.decode().strip()
        if not line or line == "data: [DONE]":
            continue
        if line.startswith("data: "):
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta:
                full_text += delta
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta','index': 0,'delta': {'type': 'text_delta','text': delta}})}\n\n"

    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop','index': 0})}\n\n"
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta','delta': {'stop_reason': 'end_turn','stop_sequence': None},'usage': {'output_tokens': len(full_text.split())}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


class AnthropicProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": AVAILABLE_MODELS}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/messages":
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
        sys.stderr.write(f"[{self.log_date_time_string()}] {args[0]} {args[1]} {args[2]}\n")


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
