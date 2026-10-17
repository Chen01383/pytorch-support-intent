"""Run the predeclared experiment, lock decisions, then evaluate the test set."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from data import prepare_data
from download_data import download_data
from encoder import cache_embeddings
from metrics import evaluate, fit_temperature, paired_bootstrap, probabilities, routing_metrics, select_threshold
from model import train_head

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu or install a compatible CUDA PyTorch wheel.")
    torch.set_num_threads(4)
    started = time.perf_counter()
    plan_path = ROOT / "experiment_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    download_data()
    splits, labels, audit = prepare_data()
    targets = {name: np.array([row["label"] for row in rows]) for name, rows in splits.items()}
    features = cache_embeddings(splits, device=args.device)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    train_tfidf = vectorizer.fit_transform([row["text"] for row in splits["train"]])
    select_tfidf = vectorizer.transform([row["text"] for row in splits["selection"]])
    baseline_candidates = []
    best_baseline = None
    best_baseline_score = -1.0
    for c in plan["baseline_C_candidates"]:
        classifier = LogisticRegression(C=c, max_iter=1000, solver="lbfgs")
        classifier.fit(train_tfidf, targets["train"])
        selection = evaluate(classifier.predict_proba(select_tfidf), targets["selection"])
        baseline_candidates.append({"C": c, "selection": selection})
        if selection["macro_f1"] > best_baseline_score:
            best_baseline, best_baseline_score, best_c = classifier, selection["macro_f1"], c
        print(f"TF-IDF C={c}, selection_macro_f1={selection['macro_f1']:.5f}", flush=True)
    candidate_heads, head_runs = {}, []
    for hidden in plan["hidden_dim_candidates"]:
        model, run = train_head(features, targets, hidden, plan["deployment_seed"], plan, args.device)
        candidate_heads[hidden] = (model, run)
        head_runs.append(run)
    best_hidden = max(candidate_heads, key=lambda hidden: candidate_heads[hidden][1]["selection_macro_f1"])
    final_heads = {plan["deployment_seed"]: candidate_heads[best_hidden]}
    for seed in plan["seeds"]:
        if seed not in final_heads:
            final_heads[seed] = train_head(features, targets, best_hidden, seed, plan, args.device)
    calibration = {}
    for seed, (model, run) in final_heads.items():
        with torch.inference_mode():
            logits = model(torch.from_numpy(features["calibration"])).numpy()
        temperature = fit_temperature(logits, targets["calibration"], plan["temperature_bounds"])
        probs = probabilities(logits, temperature)
        threshold = select_threshold(probs, targets["calibration"], plan["routing_thresholds"],
                                     plan["routing_target_calibration_accuracy"], plan["routing_min_calibration_accepted"])
        calibration[seed] = {"temperature": temperature, "threshold": threshold,
                             "raw": evaluate(probabilities(logits), targets["calibration"]),
                             "calibrated": evaluate(probs, targets["calibration"]),
                             "routing": routing_metrics(probs, targets["calibration"], threshold)}
        checkpoint = {"state_dict": model.state_dict(), "hidden_dim": best_hidden, "labels": labels,
                      "dropout": plan["dropout"], "seed": seed, "temperature": temperature,
                      "threshold": threshold, "input_dim": features["train"].shape[1],
                      "encoder": json.loads((ROOT / "SOURCES.json").read_text(encoding="utf-8"))}
        (ROOT / "artifacts").mkdir(exist_ok=True)
        torch.save(checkpoint, ROOT / f"artifacts/head_seed{seed}.pt")
    lock = {"selected_hidden_dim": best_hidden, "selected_baseline_C": best_c,
            "deployment_seed": plan["deployment_seed"], "calibration": calibration,
            "baseline_candidates": baseline_candidates, "head_candidates": head_runs,
            "final_training": {seed: run for seed, (_, run) in final_heads.items()},
            "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "locked_before_test_evaluation": True, "audit": audit}
    write_json(ROOT / "reports/selection_locked.json", lock)
    print("Locked hyperparameters, fitted temperatures and routing policies. Starting test evaluation.", flush=True)
    test_y = targets["test"]
    baseline_probs = best_baseline.predict_proba(vectorizer.transform([row["text"] for row in splits["test"]]))
    baseline_result = evaluate(baseline_probs, test_y)
    test_runs, test_logits = {}, {}
    for seed, (model, _) in final_heads.items():
        with torch.inference_mode():
            logits = model(torch.from_numpy(features["test"])).numpy()
        test_logits[seed] = logits
        probs = probabilities(logits, calibration[seed]["temperature"])
        test_runs[seed] = {"raw": evaluate(probabilities(logits), test_y), "calibrated": evaluate(probs, test_y),
                           "routing": routing_metrics(probs, test_y, calibration[seed]["threshold"])}
    deployment_seed = plan["deployment_seed"]
    deployed = probabilities(test_logits[deployment_seed], calibration[deployment_seed]["temperature"])
    differences = paired_bootstrap(deployed.argmax(1) == test_y, baseline_probs.argmax(1) == test_y,
                                   repetitions=plan["bootstrap_repetitions"])
    summary = {"baseline": baseline_result, "neural_seeds": test_runs,
               "neural_mean_std": {metric: {"mean": float(np.mean([r["calibrated"][metric] for r in test_runs.values()])),
                                            "sample_std": float(np.std([r["calibrated"][metric] for r in test_runs.values()], ddof=1))}
                                   for metric in ("accuracy", "macro_f1", "top3_accuracy")},
               "deployed_seed": deployment_seed, "paired_accuracy_bootstrap": differences,
               "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                               "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                               "torch_threads": torch.get_num_threads()},
               "experiment_seconds": time.perf_counter() - started}
    write_json(ROOT / "reports/summary.json", summary)
    np.savez_compressed(ROOT / "artifacts/test_outputs.npz", y=test_y, baseline=baseline_probs,
                        **{f"logits_seed{seed}": logits for seed, logits in test_logits.items()})
    class_report = classification_report(test_y, deployed.argmax(1), labels=np.arange(len(labels)),
                                         target_names=labels, output_dict=True, zero_division=0)
    write_json(ROOT / "reports/per_intent_metrics.json", class_report)
    matrix = confusion_matrix(test_y, deployed.argmax(1), labels=np.arange(len(labels)))
    pairs = sorted(((int(matrix[i, j]), labels[i], labels[j]) for i in range(len(labels))
                    for j in range(len(labels)) if i != j and matrix[i, j]), reverse=True)[:20]
    write_json(ROOT / "reports/confusion_pairs.json", [{"count": count, "true": true, "predicted": pred} for count, true, pred in pairs])
    with (ROOT / "reports/test_predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "true_intent", "neural_intent", "baseline_intent", "confidence", "route"])
        writer.writeheader()
        for row, probs, base in zip(splits["test"], deployed, baseline_probs):
            confidence = float(probs.max())
            writer.writerow({"id": row["id"], "true_intent": labels[row["label"]], "neural_intent": labels[probs.argmax()],
                             "baseline_intent": labels[base.argmax()], "confidence": confidence,
                             "route": "auto" if confidence >= calibration[deployment_seed]["threshold"] else "human"})
    errors = [{"id": row["id"], "text": row["text"], "true": labels[row["label"]],
               "predicted": labels[probs.argmax()], "confidence": float(probs.max())}
              for row, probs in zip(splits["test"], deployed) if row["label"] != probs.argmax()]
    write_json(ROOT / "reports/error_examples.json", sorted(errors, key=lambda r: -r["confidence"])[:30])
    from build_report import build_report
    build_report()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
