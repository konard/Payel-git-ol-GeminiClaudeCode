"""Mock OpenAI-compatible backend standing in for gemini-web2api at :8081."""
import json, time, uuid
from http.server import HTTPServer, BaseHTTPRequestHandler

MODELS = ["gemini-3.5-flash","gemini-3.5-flash-thinking","gemini-flash-lite"]

class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def _j(self,d,s=200):
        b=json.dumps(d).encode()
        self.send_response(s); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path=="/v1/models":
            self._j({"object":"list","data":[{"id":n,"object":"model","created":1700000000,"owned_by":"google"} for n in MODELS]})
        else: self._j({"error":"nf"},404)
    def do_POST(self):
        ln=int(self.headers.get("Content-Length",0)); body=json.loads(self.rfile.read(ln)) if ln else {}
        if self.path=="/v1/chat/completions":
            model=body.get("model","?")
            wants_tool = "weather" in json.dumps(body).lower() and body.get("tools")
            if wants_tool:
                tc=[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Tokyo\"}"}}]
                if body.get("stream"):
                    self.send_response(200); self.send_header("Content-Type","text/event-stream"); self.end_headers()
                    cid=f"chatcmpl-{uuid.uuid4().hex[:12]}"
                    ch={"id":cid,"object":"chat.completion.chunk","created":int(time.time()),"model":model,"choices":[{"index":0,"delta":{"role":"assistant","content":None,"tool_calls":tc},"finish_reason":"tool_calls"}]}
                    self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode()); self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                else:
                    self._j({"id":f"chatcmpl-{uuid.uuid4().hex[:12]}","object":"chat.completion","created":int(time.time()),"model":model,"choices":[{"index":0,"message":{"role":"assistant","content":None,"tool_calls":tc},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}})
                return
            if body.get("stream"):
                self.send_response(200); self.send_header("Content-Type","text/event-stream"); self.end_headers()
                cid=f"chatcmpl-{uuid.uuid4().hex[:12]}"
                for piece in ["Привет","! ","Чем ","помочь?"]:
                    ch={"id":cid,"object":"chat.completion.chunk","created":int(time.time()),"model":model,"choices":[{"index":0,"delta":{"content":piece},"finish_reason":None}]}
                    self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode()); self.wfile.flush()
                end={"id":cid,"object":"chat.completion.chunk","created":int(time.time()),"model":model,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
                self.wfile.write(f"data: {json.dumps(end)}\n\n".encode()); self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            else:
                self._j({"id":f"chatcmpl-{uuid.uuid4().hex[:12]}","object":"chat.completion","created":int(time.time()),"model":model,"choices":[{"index":0,"message":{"role":"assistant","content":"Привет! Чем помочь?"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}})
        else: self._j({"error":"nf"},404)

HTTPServer(("127.0.0.1",8081),H).serve_forever()
