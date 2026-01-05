import argparse
import json
from pathlib import Path

import redis

from leak_detect_ai_physics import config


def _load_model(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _score_features(features: dict, model_features: list[str]) -> float:
    if not features or not model_features:
        return 0.0
    values = [abs(float(features[name])) for name in model_features if name in features]
    if not values:
        return 0.0
    return sum(values) / len(values)


def run_inference() -> int:
    parser = argparse.ArgumentParser(
        description="Run inference on live feature streams."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=config.MODEL_PATH,
        help="Path to the trained model JSON artifact.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        return 1

    model = _load_model(args.model)
    model_features = model.get("features", [])

    r = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        decode_responses=True,
    )

    pubsub = r.pubsub()
    pubsub.subscribe(config.OUTPUT_CHANNEL)
    print(f"AI inference listening on '{config.OUTPUT_CHANNEL}' using {args.model}...")

    for message in pubsub.listen():
        if message.get("type") != "message":
            continue
        try:
            payload = json.loads(message["data"])
            features = payload.get("normalized", {})
            score = _score_features(features, model_features)
            prediction = {
                "entity_type": payload.get("entity_type"),
                "entity_id": payload.get("entity_id"),
                "sensor_id": payload.get("sensor_id"),
                "timestamp": payload.get("timestamp"),
                "score": score,
                "model": str(args.model),
            }
            r.publish(config.AI_OUTPUT_CHANNEL, json.dumps(prediction))
            print(
                "Prediction for {entity}: {score:.4f}".format(
                    entity=prediction.get("sensor_id") or prediction.get("entity_id"),
                    score=score,
                )
            )
        except Exception as exc:
            print(f"Error during inference: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_inference())
