import argparse
import json
from collections import Counter
from pathlib import Path

from leak_detect_ai_physics import config


def _iter_training_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def run_training() -> int:
    parser = argparse.ArgumentParser(
        description="Train a baseline leak model from JSONL training data."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.TRAINING_DATA_PATH,
        help="Path to training dataset JSONL file.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=config.MODEL_PATH,
        help="Path to write the model JSON artifact.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Training dataset not found: {args.dataset}")
        return 1

    feature_counts: Counter[str] = Counter()
    record_count = 0
    label_counts = Counter()
    for record in _iter_training_records(args.dataset):
        record_count += 1
        label_counts[record.get("label", 0)] += 1
        normalized = record.get("normalized") or {}
        feature_counts.update(normalized.keys())

    if record_count == 0:
        print(f"Training dataset is empty: {args.dataset}")
        return 1

    features = [name for name, _ in feature_counts.most_common()]
    model = {
        "version": 1,
        "dataset": str(args.dataset),
        "record_count": record_count,
        "features": features,
        "label_counts": dict(label_counts),
    }

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    with args.output_model.open("w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Model saved to {args.output_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_training())
