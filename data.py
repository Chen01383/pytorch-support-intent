"""Data isolation, deduplication and reproducible stratified partitions."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent


def normalize_text(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def clean_training(rows, test_texts):
    """Inspect test inputs only for exact decontamination; never inspect test labels."""
    test_keys = {normalize_text(text) for text in test_texts}
    groups = defaultdict(list)
    for row in rows:
        groups[normalize_text(row["text"])].append(row)
    kept, counts = [], Counter()
    for key, group in groups.items():
        if not key:
            counts["empty_rows_removed"] += len(group)
        elif key in test_keys:
            counts["test_overlap_train_rows_removed"] += len(group)
        elif len({row["label"] for row in group}) > 1:
            counts["conflicting_train_rows_removed"] += len(group)
        else:
            kept.append(group[0])
            counts["duplicate_train_rows_removed"] += len(group) - 1
    return kept, dict(counts)


def prepare_data():
    labels = json.loads((ROOT / "data/raw/categories.json").read_text(encoding="utf-8"))
    label_to_id = {label: i for i, label in enumerate(labels)}
    def read(name):
        with (ROOT / f"data/raw/{name}.csv").open(encoding="utf-8-sig", newline="") as file:
            return [{"id": f"{name}:{i}", "text": row["text"], "label": label_to_id[row["category"]]}
                    for i, row in enumerate(csv.DictReader(file))]
    official_train, test = read("train"), read("test")
    clean, removed = clean_training(official_train, [row["text"] for row in test])
    plan = json.loads((ROOT / "experiment_plan.json").read_text(encoding="utf-8"))
    train, held = train_test_split(clean, test_size=plan["selection_fraction"] + plan["calibration_fraction"],
                                  random_state=plan["split_seed"], stratify=[r["label"] for r in clean])
    select, calibration = train_test_split(held, test_size=plan["calibration_fraction"] / (
        plan["selection_fraction"] + plan["calibration_fraction"]), random_state=plan["split_seed"] + 1,
        stratify=[r["label"] for r in held])
    splits = {"train": train, "selection": select, "calibration": calibration, "test": test}
    keys = {name: {normalize_text(row["text"]) for row in rows} for name, rows in splits.items()}
    for i, name in enumerate(keys):
        for other in list(keys)[i + 1:]:
            assert not keys[name] & keys[other], f"Text leakage between {name} and {other}"
    audit = {"official_train_rows": len(official_train), "official_test_rows": len(test),
             "classes": len(labels), "removed_training_rows": removed,
             "split_sizes": {name: len(rows) for name, rows in splits.items()},
             "test_internal_normalized_duplicates": len(test) - len(keys["test"]),
             "decontamination": "Full official test preserved; matching normalized inputs removed from training candidates, without consulting test labels.",
             "pairwise_split_text_overlaps": 0,
             "split_id_sha256": {name: hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest()
                                 for name, rows in splits.items()}}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (ROOT / "data/split_ids.json").write_text(json.dumps({k: [r["id"] for r in v] for k, v in splits.items()}), encoding="utf-8")
    return splits, labels, audit


if __name__ == "__main__":
    print(json.dumps(prepare_data()[2], indent=2))
