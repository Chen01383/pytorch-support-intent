"""Cached local inference, including calibration and an explicit routing action."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from encoder import FrozenEncoder
from metrics import probabilities
from model import IntentHead

ROOT = Path(__file__).resolve().parent


class IntentService:
    def __init__(self, device="cpu", checkpoint_path=None):
        checkpoint = torch.load(checkpoint_path or ROOT / "artifacts/head_seed42.pt", map_location="cpu", weights_only=True)
        self.labels, self.temperature, self.threshold = checkpoint["labels"], checkpoint["temperature"], checkpoint["threshold"]
        sources = json.loads((ROOT / "SOURCES.json").read_text(encoding="utf-8"))
        if checkpoint["encoder"] != sources:
            raise ValueError("The checkpoint and encoder source manifest disagree.")
        self.head = IntentHead(checkpoint["input_dim"], len(self.labels), checkpoint["hidden_dim"], checkpoint["dropout"])
        self.head.load_state_dict(checkpoint["state_dict"])
        self.head.eval()
        self.encoder = FrozenEncoder(device)

    @torch.inference_mode()
    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please enter a non-empty English customer-service query.")
        if len(text) > 2000:
            raise ValueError("Query too long: use at most 2,000 characters.")
        if not any("a" <= char.lower() <= "z" for char in text):
            return {"route": "human", "reason": "English-only benchmark: unsupported input.", "top3": []}
        features = self.encoder.encode([text])
        probs = probabilities(self.head(torch.from_numpy(features)).numpy(), self.temperature)[0]
        top = np.argsort(probs)[-3:][::-1]
        return {"input": text, "route": "auto" if probs[top[0]] >= self.threshold else "human",
                "confidence": float(probs[top[0]]), "threshold": self.threshold,
                "top3": [{"intent": self.labels[i], "probability": float(probs[i])} for i in top],
                "note": "Routing predicts an intent; it does not answer banking questions. Confidence is not a guarantee of correctness or an out-of-domain detector."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    service = IntentService(args.device)
    if args.text is not None:
        print(json.dumps(service.predict(args.text), ensure_ascii=False, indent=2))
        return
    print("English customer-service intent demo. Blank = example; q = quit.")
    while True:
        try:
            text = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"q", "quit", "exit"}:
            break
        try:
            print(json.dumps(service.predict(text or "I lost my card. How can I freeze it?"), ensure_ascii=False, indent=2))
        except ValueError as error:
            print(error)


if __name__ == "__main__":
    main()
