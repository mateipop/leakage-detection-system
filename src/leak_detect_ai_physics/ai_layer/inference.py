import argparse
import json
import logging
from pathlib import Path

import redis
import torch

from leak_detect_ai_physics import config
from leak_detect_ai_physics.ai_layer.trainer import CnnClassifier, ModelSpec

LOG = logging.getLogger(__name__)


def _load_model(path: Path):
    payload = torch.load(path, map_location="cpu")
    metadata = payload["metadata"]
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
    args = parser.parse_args()

    if not args.anomaly_model.exists():
        LOG.error("Anomaly model not found: %s", args.anomaly_model)
        return 1

    anomaly_model, anomaly_meta = _load_model(args.anomaly_model)
    anomaly_features = anomaly_meta.get("features", [])
    window_steps = int(anomaly_meta.get("window_steps", 1))
    pinpointer_model = None
    pinpointer_meta = {}
    if args.pinpointer_model.exists():
        pinpointer_model, pinpointer_meta = _load_model(args.pinpointer_model)
    pinpointer_classes = pinpointer_meta.get("class_names", [])
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
            window_tensor = torch.tensor([buffers[key]], dtype=torch.float32)
            with torch.no_grad():
                logits = anomaly_model(window_tensor)
                anomaly_score = float(torch.sigmoid(logits)[0][0].item())

            pinpointer_prediction = None
            pinpointer_confidence = 0.0
            if pinpointer_model and pinpointer_classes:
                pin_buffer = buffers[key]
                if len(pin_buffer) >= pinpointer_window_steps:
                    pin_window = pin_buffer[-pinpointer_window_steps:]
                    with torch.no_grad():
                        logits = pinpointer_model(
                            torch.tensor([pin_window], dtype=torch.float32)
                        )
                        probs = torch.softmax(logits, dim=1)[0].numpy()
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
