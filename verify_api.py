"""Exercise the real inference service through HTTP, using an ephemeral local port."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request
import torch
from predict import IntentService
from serve import make_handler

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(IntentService()))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    checks = []
    def post(value, expected=200, content_type="application/json"):
        payload = json.dumps(value).encode()
        request = urllib.request.Request(base + "/predict", data=payload, headers={"Content-Type": content_type})
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as response_error:
            response = response_error
        with response:
            assert response.code == expected, (response.code, expected)
            return json.load(response)
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as response:
            assert json.load(response)["classes"] == 77
        checks.append("health_ready")
        actual = post({"text": "I lost my card. How can I freeze it?"})
        assert actual["top3"][0]["intent"] == "lost_or_stolen_card" and actual["route"] == "auto"
        checks.append("real_prediction_and_auto_route")
        human = post({"text": "Can you help me?"})
        assert human["route"] == "human"
        checks.append("ambiguous_query_human_route")
        limitation = post({"text": "I need help with my account."})
        assert limitation["route"] == ("auto" if limitation["confidence"] >= limitation["threshold"] else "human")
        checks.append("routing_matches_probability_even_for_ambiguous_high_confidence_failure")
        assert post({"text": "我想挂失银行卡"})["route"] == "human"
        checks.append("unsupported_input_human_route")
        post({"text": ""}, expected=400)
        post({"text": "a" * 2001}, expected=400)
        post(["wrong-shape"], expected=400)
        post({"text": "a"}, expected=415, content_type="text/plain")
        checks.append("empty_oversized_wrong_shape_and_content_type_rejected")
        result = {"status": "passed", "checks": checks, "real_prediction": actual, "ambiguous_prediction": human,
                  "documented_high_confidence_limitation": limitation}
        (ROOT / "reports/api_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "passed", "checks": checks}, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
