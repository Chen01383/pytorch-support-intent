"""Classification, confidence calibration, selective routing and uncertainty."""
from math import sqrt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import accuracy_score, f1_score, log_loss


def probabilities(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return softmax(np.asarray(logits, dtype=np.float64) / temperature, axis=1)


def fit_temperature(logits, labels, bounds=(0.25, 4.0)):
    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels)
    def objective(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))
    result = minimize_scalar(objective, bounds=np.log(bounds), method="bounded")
    candidate = float(np.exp(result.x))
    # Never worsen calibration NLL relative to the identity transform.
    return candidate if objective(np.log(candidate)) < objective(0.0) else 1.0


def expected_calibration_error(probs, labels, bins=15):
    confidences, predicted = probs.max(1), probs.argmax(1)
    correct = predicted == np.asarray(labels)
    ece = 0.0
    for index in range(bins):
        mask = (confidences >= index / bins) & (confidences <= (index + 1) / bins if index == bins - 1 else confidences < (index + 1) / bins)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def evaluate(probs, labels):
    labels = np.asarray(labels)
    return {"accuracy": float(accuracy_score(labels, probs.argmax(1))),
            "macro_f1": float(f1_score(labels, probs.argmax(1), labels=np.arange(probs.shape[1]), average="macro", zero_division=0)),
            "top3_accuracy": float(np.mean(np.any(np.argsort(probs, axis=1)[:, -3:] == labels[:, None], axis=1))),
            "nll": float(log_loss(labels, probs, labels=np.arange(probs.shape[1]))),
            "ece_15_bins": expected_calibration_error(probs, labels)}


def select_threshold(probs, labels, thresholds, target_accuracy=0.95, min_accepted=50):
    # Calibration-only empirical policy selection; test is never an argument here.
    for threshold in sorted(thresholds):
        accepted = probs.max(1) >= threshold
        if accepted.sum() >= min_accepted and (probs.argmax(1)[accepted] == np.asarray(labels)[accepted]).mean() >= target_accuracy:
            return float(threshold)
    return 1.000001  # Explicitly route everything to a person if target cannot be met.


def wilson_interval(correct, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    p = correct / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [float(center - half), float(center + half)]


def routing_metrics(probs, labels, threshold):
    accepted = probs.max(1) >= threshold
    count = int(accepted.sum())
    correct = int((probs.argmax(1)[accepted] == np.asarray(labels)[accepted]).sum())
    return {"threshold": float(threshold), "accepted": count, "total": len(labels),
            "coverage": float(accepted.mean()), "human_review_fraction": float(1 - accepted.mean()),
            "accepted_accuracy": float(correct / count) if count else None,
            "accepted_accuracy_wilson_95": wilson_interval(correct, count)}


def paired_bootstrap(first_correct, second_correct, repetitions=2000, seed=2026):
    differences = np.asarray(first_correct, dtype=np.float64) - np.asarray(second_correct, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions)
    for index in range(repetitions):
        values[index] = rng.choice(differences, len(differences), replace=True).mean()
    return {"accuracy_difference": float(differences.mean()),
            "percentile_95": np.quantile(values, [0.025, 0.975]).tolist(),
            "repetitions": repetitions, "unit": "test query; paired; conditional on fitted models"}
