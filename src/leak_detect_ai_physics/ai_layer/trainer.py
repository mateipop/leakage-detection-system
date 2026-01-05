import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from leak_detect_ai_physics import config

LOG = logging.getLogger(__name__)


def _iter_training_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_feature_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        normalized = record.get("normalized") or {}
        rows.append({key: float(value) for key, value in normalized.items()})
    return rows


def _train_binary_classifier(x_rows: list[dict], y_labels: list[int]) -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                ),
            ),
        ]
    ).fit(x_rows, y_labels)


def _train_multiclass_classifier(x_rows: list[dict], y_labels: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                ),
            ),
        ]
    ).fit(x_rows, y_labels)


def _report_binary_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_training() -> int:
    parser = argparse.ArgumentParser(
        description="Train anomaly and pinpointer models from JSONL training data."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.TRAINING_DATA_PATH,
        help="Path to training dataset JSONL file.",
    )
    parser.add_argument(
        "--anomaly-model",
        type=Path,
        default=config.ANOMALY_MODEL_PATH,
        help="Path to write the anomaly model.",
    )
    parser.add_argument(
        "--pinpointer-model",
        type=Path,
        default=config.PINPOINTER_MODEL_PATH,
        help="Path to write the pinpointer model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout fraction for metrics.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        LOG.error("Training dataset not found: %s", args.dataset)
        return 1

    records = list(_iter_training_records(args.dataset))
    if not records:
        LOG.error("Training dataset is empty: %s", args.dataset)
        return 1

    x_rows = _build_feature_rows(records)
    y_labels = [int(record.get("label", 0)) for record in records]
    unique_labels = sorted(set(y_labels))
    if len(unique_labels) < 2:
        LOG.error(
            "Need at least 2 classes for anomaly training, got: %s",
            unique_labels,
        )
        return 1

    x_train, x_test, y_train, y_test = train_test_split(
        x_rows,
        y_labels,
        test_size=args.test_size,
        random_state=42,
        stratify=y_labels,
    )
    anomaly_model = _train_binary_classifier(x_train, y_train)
    anomaly_preds = anomaly_model.predict(x_test)
    anomaly_report = _report_binary_metrics(y_test, anomaly_preds)

    pin_records = [
        record
        for record in records
        if record.get("entity_type") == "node" and record.get("leak_active")
    ]
    pinpointer_report = {}
    pinpointer_model = None
    if pin_records:
        pin_x = _build_feature_rows(pin_records)
        pin_y = [record["entity_id"] for record in pin_records]
        pin_x_train, pin_x_test, pin_y_train, pin_y_test = train_test_split(
            pin_x,
            pin_y,
            test_size=args.test_size,
            random_state=42,
            stratify=pin_y if len(set(pin_y)) > 1 else None,
        )
        pinpointer_model = _train_multiclass_classifier(pin_x_train, pin_y_train)
        pin_preds = pinpointer_model.predict(pin_x_test)
        pinpointer_report = {
            "accuracy": accuracy_score(pin_y_test, pin_preds),
        }

    args.anomaly_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(anomaly_model, args.anomaly_model)
    if pinpointer_model:
        args.pinpointer_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pinpointer_model, args.pinpointer_model)

    label_counts = Counter(y_labels)
    report = {
        "records": len(records),
        "label_counts": dict(label_counts),
        "anomaly_metrics": anomaly_report,
        "pinpointer_metrics": pinpointer_report,
        "pinpointer_samples": len(pin_records),
    }
    LOG.info("Training report: %s", report)
    LOG.info("Anomaly model saved to %s", args.anomaly_model)
    if pinpointer_model:
        LOG.info("Pinpointer model saved to %s", args.pinpointer_model)
    else:
        LOG.info("Pinpointer model skipped (no leak-active node samples).")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(run_training())
