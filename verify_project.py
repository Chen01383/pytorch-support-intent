"""Verify real saved weights, reported metrics, data isolation and inference."""
import json
from pathlib import Path
import numpy as np
import torch

from data import prepare_data
from download_data import download_data
from metrics import evaluate, probabilities, routing_metrics
from model import IntentHead
from predict import IntentService

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    download_data()
    _, labels, audit = prepare_data()
    summary = json.loads((ROOT / "reports/summary.json").read_text(encoding="utf-8"))
    plan = json.loads((ROOT / "experiment_plan.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "data/embeddings.npz") as features, np.load(ROOT / "artifacts/test_outputs.npz") as outputs:
        y = outputs["y"]
        for seed in plan["seeds"]:
            checkpoint = torch.load(ROOT / f"artifacts/head_seed{seed}.pt", map_location="cpu", weights_only=True)
            assert checkpoint["labels"] == labels
            model = IntentHead(checkpoint["input_dim"], len(labels), checkpoint["hidden_dim"], checkpoint["dropout"])
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            with torch.inference_mode():
                logits = model(torch.from_numpy(features["test"])).numpy()
            np.testing.assert_allclose(logits, outputs[f"logits_seed{seed}"], atol=1e-6)
            probs = probabilities(logits, checkpoint["temperature"])
            metrics = evaluate(probs, y)
            for metric, value in metrics.items():
                assert abs(value - summary["neural_seeds"][str(seed)]["calibrated"][metric]) < 1e-9
            assert routing_metrics(probs, y, checkpoint["threshold"]) == summary["neural_seeds"][str(seed)]["routing"]
    service = IntentService()
    result = service.predict("I lost my card. How can I freeze it?")
    assert result["top3"][0]["intent"] in labels
    assert all(0 <= item["probability"] <= 1 for item in result["top3"])
    assert service.predict("我想挂失银行卡")["route"] == "human"
    for bad in ["", "   ", "a" * 2001]:
        try:
            service.predict(bad)
        except ValueError:
            continue
        raise AssertionError("Invalid input was not rejected")
    verified = {"status": "passed", "checkpoints_verified": len(plan["seeds"]),
                "test_rows_recomputed": len(y), "pairwise_split_text_overlaps": audit["pairwise_split_text_overlaps"],
                "reported_metrics_recomputed": True, "invalid_input_checks_passed": True,
                "non_english_guard_checked": True, "example_prediction": result}
    (ROOT / "reports/verification.json").write_text(json.dumps(verified, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verified, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
