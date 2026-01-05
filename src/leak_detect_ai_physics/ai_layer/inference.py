import argparse
import json
import logging
from pathlib import Path

import joblib
import redis

from leak_detect_ai_physics import config

LOG = logging.getLogger(__name__)


def _load_optional_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def _build_feature_row(payload: dict) -> dict:
    normalized = payload.get("normalized") or {}
    return {key: float(value) for key, value in normalized.items()}


def _pinpointer_score(model, feature_row: dict, entity_id: str | None) -> float:
    if model is None or entity_id is None:
        return 0.0
    classes = getattr(model.named_steps["classifier"], "classes_", [])
    if entity_id not in classes:
        return 0.0
    class_index = list(classes).index(entity_id)
    probs = model.predict_proba([feature_row])[0]
    return float(probs[class_index])


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

    anomaly_model = _load_optional_model(args.anomaly_model)
    if anomaly_model is None:
        LOG.error("Anomaly model not found: %s", args.anomaly_model)
        return 1
    pinpointer_model = _load_optional_model(args.pinpointer_model)

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

    for message in pubsub.listen():
        if message.get("type") != "message":
            continue
        try:
            payload = json.loads(message["data"])
            feature_row = _build_feature_row(payload)
            anomaly_score = float(anomaly_model.predict_proba([feature_row])[0][1])
            entity_id = payload.get("entity_id")
            pinpointer_score = _pinpointer_score(
                pinpointer_model, feature_row, entity_id
            )
            prediction = {
                "entity_type": payload.get("entity_type"),
                "entity_id": entity_id,
                "sensor_id": payload.get("sensor_id"),
                "timestamp": payload.get("timestamp"),
                "anomaly_score": anomaly_score,
                "pinpointer_score": pinpointer_score,
                "anomaly_model": str(args.anomaly_model),
                "pinpointer_model": str(args.pinpointer_model),
            }
            r.publish(config.AI_OUTPUT_CHANNEL, json.dumps(prediction))
            LOG.info(
                "Prediction for %s: anomaly=%.4f pinpointer=%.4f",
                prediction.get("sensor_id") or prediction.get("entity_id"),
                anomaly_score,
                pinpointer_score,
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
