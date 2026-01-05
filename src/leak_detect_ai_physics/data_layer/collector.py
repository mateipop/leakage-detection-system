import json

import redis

from leak_detect_ai_physics import config
from leak_detect_ai_physics.data_layer.processor import FeatureProcessor
from leak_detect_ai_physics.data_layer.storage import append_training_record
from leak_detect_ai_physics.data_layer.utils import validate_telemetry


def run_collector():
    r = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        decode_responses=True,
    )
    processor = FeatureProcessor(window_size=20)

    pubsub = r.pubsub()
    pubsub.subscribe(config.INPUT_CHANNEL)

    print(f"Data Layer Active. Listening on '{config.INPUT_CHANNEL}'...")

    for message in pubsub.listen():
        if message["type"] == "message":
            try:
                data = json.loads(message["data"])

                # 1. VALIDATE
                if not validate_telemetry(data):
                    print(f"Validation failed: {data}")
                    continue

                raw_features = {
                    "pressure": data["pressure"],
                    "head": data["head"],
                    "demand": data["demand"],
                }

                features = processor.process(data["sensor_id"], raw_features)

                training_record = {
                    "sensor_id": data["sensor_id"],
                    "timestamp": data["timestamp"],
                    "network": data.get("network"),
                    "features": raw_features,
                    "normalized": features["normalized"],
                    "elevation": data.get("elevation"),
                    "leak_active": data.get("leak_active", False),
                    "label": int(bool(data.get("leak_active", False))),
                    "leak_node": data.get("leak_node"),
                }
                append_training_record(config.TRAINING_DATA_PATH, training_record)

                features["timestamp"] = data["timestamp"]
                r.publish(config.OUTPUT_CHANNEL, json.dumps(features))
                print(
                    "Streamed features for {sensor}: {normalized}".format(
                        sensor=data["sensor_id"],
                        normalized=features["normalized"],
                    )
                )

            except Exception as e:
                print(f"Error processing message: {e}")


if __name__ == "__main__":
    run_collector()
