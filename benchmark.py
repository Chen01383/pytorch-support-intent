"""Warm, batch-one latency, including tokenization through routing JSON creation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from predict import IntentService

ROOT = Path(__file__).resolve().parent
QUERIES = ["I lost my card. How can I freeze it?", "Where can I find my card PIN?",
           "Why is my transfer still pending?", "Can I change my account's personal details?",
           "I was charged twice for the same purchase."]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--requests", type=int, default=100)
    args = parser.parse_args()
    if args.requests < 1:
        raise ValueError("Use at least one request")
    torch.set_num_threads(4)
    service = IntentService(args.device)
    for index in range(10):
        service.predict(QUERIES[index % len(QUERIES)])
    timings = []
    for index in range(args.requests):
        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        service.predict(QUERIES[index % len(QUERIES)])
        if args.device == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter_ns() - start) / 1e6)
    result = {"device": args.device, "threads": 4, "batch_size": 1, "concurrency": 1,
              "warmups": 10, "requests": args.requests, "queries": QUERIES,
              "p50_ms": float(np.quantile(timings, .5)), "p95_ms": float(np.quantile(timings, .95)),
              "scope": "Warm local request: tokenization, frozen encoder, CPU classifier head, calibration, ranking, routing and Python dict construction. Excludes model loading, disk/network I/O, HTTP and serialization."}
    (ROOT / f"reports/latency_{args.device}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
