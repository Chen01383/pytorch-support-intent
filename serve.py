"""Dependency-free localhost demo and JSON inference API."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import torch

from predict import IntentService

ROOT = Path(__file__).resolve().parent


def make_handler(service):
    inference_lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                return self.send_json(200, {"status": "ready", "classes": len(service.labels), "language": "en"})
            if self.path != "/":
                return self.send_json(404, {"error": "Not found"})
            payload = (ROOT / "web/index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            if self.path != "/predict":
                return self.send_json(404, {"error": "Not found"})
            # This local demo accepts same-origin JSON requests only.
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json":
                return self.send_json(415, {"error": "Use application/json"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 8192:
                    return self.send_json(413, {"error": "Request must be 1..8192 bytes"})
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict) or not isinstance(body.get("text"), str):
                    raise ValueError("Expected a JSON object with a text string")
                with inference_lock:
                    result = service.predict(body["text"])
                self.send_json(200, result)
            except (ValueError, UnicodeError) as error:
                self.send_json(400, {"error": str(error)})

        def log_message(self, format, *args):
            # Do not persist submitted customer query text.
            return
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    service = IntentService(args.device)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service))
    print(f"Ready: http://127.0.0.1:{server.server_port}   Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
