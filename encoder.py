"""Frozen Transformer encoder with attention-mask-aware mean pooling."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent


def mean_pool(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return F.normalize((hidden * weights).sum(1) / weights.sum(1).clamp(min=1), dim=1)


class FrozenEncoder:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        path = ROOT / "models/minilm"
        if not (path / "model.safetensors").exists():
            raise FileNotFoundError("Run python prepare_model.py once to download the pinned encoder.")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to(self.device).eval()
        self.model.requires_grad_(False)
        self.max_tokens = json.loads((ROOT / "experiment_plan.json").read_text())["max_tokens"]

    @torch.inference_mode()
    def encode(self, texts, batch_size=64):
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        batches = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                    max_length=self.max_tokens, return_tensors="pt").to(self.device)
            output = self.model(**tokens).last_hidden_state
            batches.append(mean_pool(output, tokens["attention_mask"]).cpu().numpy())
        return np.concatenate(batches)


def cache_embeddings(splits, device="cpu"):
    path = ROOT / "data/embeddings.npz"
    source = json.loads((ROOT / "SOURCES.json").read_text())
    signature = hashlib.sha256(json.dumps({"sources": source,
        "max_tokens": json.loads((ROOT / "experiment_plan.json").read_text())["max_tokens"],
        "splits": {k: [(r["id"], r["text"]) for r in rows] for k, rows in splits.items()}}, sort_keys=True).encode()).hexdigest()
    signature_path = ROOT / "data/embedding_signature.txt"
    if path.exists() and signature_path.exists() and signature_path.read_text() == signature:
        with np.load(path) as cache:
            return {key: cache[key].copy() for key in splits}
    encoder = FrozenEncoder(device)
    features = {}
    for name, rows in splits.items():
        features[name] = encoder.encode([row["text"] for row in rows])
        print(f"Encoded {name}: {features[name].shape}", flush=True)
    np.savez_compressed(path, **features)
    signature_path.write_text(signature)
    return features
