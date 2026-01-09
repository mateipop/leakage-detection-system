import argparse
import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from leak_detect_ai_physics import config

LOG = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    input_dim: int
    hidden_sizes: list[int]
    output_dim: int


class AnomalyNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        return self.head(x)


class CnnClassifier(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(spec.input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, spec.output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x).squeeze(-1)
        return self.fc(x)


class PinpointerRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x).squeeze(-1)
        return self.head(x)


def _iter_training_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _collect_features(records: list[dict]) -> list[str]:
    if records and records[0].get("feature_names"):
        return list(records[0]["feature_names"])
    feature_set = set()
    for record in records:
        normalized = record.get("normalized") or {}
        feature_set.update(normalized.keys())
    return sorted(feature_set)


def _build_feature_tensor(
    records: list[dict],
    features: list[str],
    *,
    include_deltas: bool,
) -> np.ndarray:
    feature_count = len(features)
    x = np.zeros(
        (len(records), len(records[0]["features"]), feature_count),
        dtype=np.float32,
    )
    for idx, record in enumerate(records):
        window = record.get("features") or []
        for tdx, step in enumerate(window):
            for jdx, value in enumerate(step):
                if jdx >= feature_count:
                    break
                x[idx, tdx, jdx] = float(value)
    if include_deltas:
        deltas = np.zeros_like(x)
        deltas[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        x = np.concatenate([x, deltas], axis=2)
    return x


def _split_indices(count: int, *, test_size: float, seed: int):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(count)
    split = int(count * (1.0 - test_size))
    return indices[:split], indices[split:]


def _train_val_split(x, y, *, test_size: float, seed: int):
    train_idx, val_idx = _split_indices(len(x), test_size=test_size, seed=seed)
    return x[train_idx], x[val_idx], y[train_idx], y[val_idx]


def _binary_focal_loss(logits, targets, *, alpha: float = 0.25, gamma: float = 2.0):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    weights = torch.where(targets == 1, alpha, 1 - alpha)
    loss = weights * (1 - pt) ** gamma * bce
    return loss.mean()


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    accuracy = float((y_true == y_pred).mean())
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _pinpointer_distance_report(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    distances = np.linalg.norm(actual - predicted, axis=1)

    if distances.size == 0:
        return {}
    distances = np.sort(distances)
    mean_dist = float(distances.mean())
    median_dist = float(distances[len(distances) // 2])
    return {
        "mean_distance": mean_dist,
        "median_distance": median_dist,
        "samples": int(distances.size),
    }


def _save_model(path: Path, *, state_dict: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "metadata": metadata}, path)


def _train_anomaly(
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    patience: int,
) -> tuple[AnomalyNet, dict]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnomalyNet(x_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_rate = float(y_train.mean()) if len(y_train) else 0.5
    alpha = min(0.75, max(0.5, 1.0 - pos_rate + 0.05))

    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_state = None
    best_metrics = None
    best_f1 = -1.0
    stagnant_epochs = 0

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = _binary_focal_loss(logits, yb, alpha=alpha)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(x_val).to(device))
            val_probs = torch.sigmoid(val_logits).squeeze(1).detach().cpu().numpy()
        val_preds = (val_probs >= 0.5).astype(int)
        report = _binary_metrics(y_val, val_preds)
        if report["f1"] > best_f1:
            best_f1 = report["f1"]
            best_metrics = report
            best_state = copy.deepcopy(model.state_dict())
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
            if patience > 0 and stagnant_epochs >= patience:
                break
        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {"arch": "anomaly_net_v2", "input_dim": int(x_train.shape[2])}
    return model, {"metrics": best_metrics or report, "metadata": metadata}


def _train_pinpointer(
    records: list[dict],
    features: list[str],
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    test_size: float,
    include_deltas: bool,
) -> tuple[PinpointerRegressor | None, dict]:
    pin_records = [record for record in records if record.get("leak_coords")]
    if not pin_records:
        return None, {}

    pin_x = _build_feature_tensor(
        pin_records,
        features,
        include_deltas=include_deltas,
    )
    pin_y = np.array(
        [
            [
                float(record["leak_coords"].get("x", 0.0)),
                float(record["leak_coords"].get("y", 0.0)),
            ]
            for record in pin_records
        ],
        dtype=np.float32,
    )
    train_idx, val_idx = _split_indices(
        len(pin_records),
        test_size=test_size,
        seed=seed,
    )
    pin_x_train, pin_x_val = pin_x[train_idx], pin_x[val_idx]
    pin_y_train, pin_y_val = pin_y[train_idx], pin_y[val_idx]

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PinpointerRegressor(pin_x.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.SmoothL1Loss(beta=1.0)

    train_dataset = TensorDataset(
        torch.from_numpy(pin_x_train),
        torch.from_numpy(pin_y_train),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(torch.from_numpy(pin_x_val).to(device)).detach().cpu().numpy()
    distance_report = _pinpointer_distance_report(
        pin_y_val,
        val_preds,
    )
    return model, {
        "distance": distance_report,
        "metadata": {"spec": {"input_dim": int(pin_x.shape[2]), "output_dim": 2}},
    }


def run_training() -> int:
    parser = argparse.ArgumentParser(
        description="Train anomaly and pinpointer models with PyTorch."
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience based on validation F1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dataset.exists():
        LOG.error("Training dataset not found: %s", args.dataset)
        return 1

    records = list(_iter_training_records(args.dataset))
    if not records:
        LOG.error("Training dataset is empty: %s", args.dataset)
        return 1

    features = _collect_features(records)
    feature_engineering = {"include_deltas": True}
    x_all = _build_feature_tensor(
        records,
        features,
        include_deltas=feature_engineering["include_deltas"],
    )
    y_all = np.array(
        [int(record.get("label", 0)) for record in records],
        dtype=np.int64,
    )

    if len(set(y_all)) < 2:
        LOG.error("Need at least 2 classes for anomaly training, got: %s", set(y_all))
        return 1

    x_train, x_val, y_train, y_val = _train_val_split(
        x_all, y_all, test_size=args.test_size, seed=args.seed
    )
    anomaly_model, anomaly_report = _train_anomaly(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        patience=args.patience,
    )
    _save_model(
        args.anomaly_model,
        state_dict=anomaly_model.state_dict(),
        metadata={
            "features": features,
            "feature_engineering": feature_engineering,
            "window_steps": x_all.shape[1],
            **anomaly_report["metadata"],
        },
    )

    pinpointer_model, pinpointer_report = _train_pinpointer(
        records,
        features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        test_size=args.test_size,
        include_deltas=feature_engineering["include_deltas"],
    )
    if pinpointer_model:
        _save_model(
            args.pinpointer_model,
            state_dict=pinpointer_model.state_dict(),
            metadata={
                "features": features,
                "feature_engineering": feature_engineering,
                "window_steps": x_all.shape[1],
                **pinpointer_report["metadata"],
            },
        )
        LOG.info("Pinpointer model saved to %s", args.pinpointer_model)
    else:
        LOG.info("Pinpointer model skipped (no leak-active node samples).")

    report = {
        "records": len(records),
        "label_counts": {
            "0": int((y_all == 0).sum()),
            "1": int((y_all == 1).sum()),
        },
        "anomaly_metrics": anomaly_report["metrics"],
        "pinpointer_metrics": pinpointer_report,
    }
    LOG.info("Training report: %s", report)
    LOG.info("Anomaly model saved to %s", args.anomaly_model)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(run_training())
