import argparse
import json
import logging
from pathlib import Path

import numpy as np
import redis
import torch

from leak_detect_ai_physics import config
from leak_detect_ai_physics.ai_layer.trainer import AnomalyNet, CnnClassifier, ModelSpec

LOG = logging.getLogger(__name__)


def _load_model(path: Path):
    payload = torch.load(path, map_location="cpu")
    metadata = payload["metadata"]
    arch = metadata.get("arch")
    if arch == "anomaly_net_v2":
        model = AnomalyNet(int(metadata.get("input_dim", 0)))
    else:
        spec = ModelSpec(
            input_dim=metadata["spec"]["input_dim"],
            hidden_sizes=metadata["spec"]["hidden_sizes"],
            output_dim=metadata["spec"]["output_dim"],
        )
        model = CnnClassifier(spec)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, metadata


def _build_feature_vector(payload: dict, features: list[str]) -> list[float]:
    normalized = payload.get("normalized") or {}
    return [float(normalized.get(name, 0.0)) for name in features]


def _apply_feature_engineering(
    window: list[list[float]],
    *,
    include_deltas: bool,
) -> torch.Tensor:
    data = np.array(window, dtype=np.float32)
    if include_deltas:
        deltas = np.zeros_like(data)
        deltas[1:] = data[1:] - data[:-1]
        data = np.concatenate([data, deltas], axis=1)
    return torch.tensor([data], dtype=torch.float32)


def run_inference() -> int:
    parser = argparse.ArgumentParser(
        description="Run inference on live feature streams."
    )
    parser.add_argument(
        "--anomaly-model",
        type=Path,
        default=config.ANOMALY_MODEL_PATH,
        help="Path to the anomaly model.",
    )
    parser.add_argument(
        "--pinpointer-model",
        type=Path,
        default=config.PINPOINTER_MODEL_PATH,
        help="Path to the pinpointer model.",
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=0.5,
        help="Minimum anomaly score before running pinpointer.",
    )
    args = parser.parse_args()

    if not args.anomaly_model.exists():
        LOG.error("Anomaly model not found: %s", args.anomaly_model)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anomaly_model, anomaly_meta = _load_model(args.anomaly_model)
    anomaly_model = anomaly_model.to(device)
    anomaly_features = anomaly_meta.get("features", [])
    anomaly_feature_engineering = anomaly_meta.get("feature_engineering") or {}
    anomaly_include_deltas = bool(
        anomaly_feature_engineering.get("include_deltas", False)
    )
    window_steps = int(anomaly_meta.get("window_steps", 1))
    pinpointer_model = None
    pinpointer_meta = {}
    if args.pinpointer_model.exists():
        pinpointer_model, pinpointer_meta = _load_model(args.pinpointer_model)
        pinpointer_model = pinpointer_model.to(device)
    pinpointer_classes = pinpointer_meta.get("class_names", [])
    pinpointer_feature_engineering = (
        pinpointer_meta.get("feature_engineering") or anomaly_feature_engineering
    )
    pinpointer_include_deltas = bool(
        pinpointer_feature_engineering.get("include_deltas", False)
    )
    pinpointer_window_steps = int(pinpointer_meta.get("window_steps", window_steps))

    r = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        decode_responses=True,
    )

    pubsub = r.pubsub()
    pubsub.subscribe(config.OUTPUT_CHANNEL)
    LOG.info(
        "AI inference listening on '%s' using %s %s",
        config.OUTPUT_CHANNEL,
        args.anomaly_model,
        args.pinpointer_model if pinpointer_model else "(no pinpointer)",
    )
    LOG.info("Pinpointer enabled above anomaly threshold %.3f", args.anomaly_threshold)

    buffers: dict[str, list[list[float]]] = {}

    for message in pubsub.listen():
        if message.get("type") != "message":
            continue
        try:
            payload = json.loads(message["data"])
            entity_id = payload.get("entity_id")
            if not entity_id:
                continue
            key = str(entity_id)
            if key not in buffers:
                buffers[key] = []
            buffers[key].append(_build_feature_vector(payload, anomaly_features))
            if len(buffers[key]) < window_steps:
                continue
            if len(buffers[key]) > window_steps:
                buffers[key] = buffers[key][-window_steps:]
            window_tensor = _apply_feature_engineering(
                buffers[key],
                include_deltas=anomaly_include_deltas,
            ).to(device)
            with torch.no_grad():
                logits = anomaly_model(window_tensor)
                anomaly_score = float(torch.sigmoid(logits)[0][0].item())

            pinpointer_prediction = None
            pinpointer_confidence = 0.0
            if (
                pinpointer_model
                and pinpointer_classes
                and anomaly_score >= args.anomaly_threshold
            ):
                pin_buffer = buffers[key]
                if len(pin_buffer) >= pinpointer_window_steps:
                    pin_window = pin_buffer[-pinpointer_window_steps:]
                    with torch.no_grad():
                        pin_tensor = _apply_feature_engineering(
                            pin_window,
                            include_deltas=pinpointer_include_deltas,
                        ).to(device)
                        logits = pinpointer_model(pin_tensor)
                        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                    best_idx = int(probs.argmax())
                    pinpointer_prediction = pinpointer_classes[best_idx]
                    pinpointer_confidence = float(probs[best_idx])

            prediction = {
                "entity_type": payload.get("entity_type"),
                "entity_id": entity_id,
                "sensor_id": payload.get("sensor_id"),
                "timestamp": payload.get("timestamp"),
                "anomaly_score": anomaly_score,
                "pinpointer_prediction": pinpointer_prediction,
                "pinpointer_confidence": pinpointer_confidence,
                "anomaly_model": str(args.anomaly_model),
                "pinpointer_model": str(args.pinpointer_model),
            }
            r.publish(config.AI_OUTPUT_CHANNEL, json.dumps(prediction))
            LOG.info(
                "Prediction for %s: anomaly=%.4f pinpointer=%s(%.4f)",
                prediction.get("sensor_id") or prediction.get("entity_id"),
                anomaly_score,
                pinpointer_prediction,
                pinpointer_confidence,
            )
        except Exception as exc:
            LOG.exception("Error during inference: %s", exc)

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(run_inference())
