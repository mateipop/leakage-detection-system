import json
import redis
from .utils import validate_telemetry
from .processor import FeatureProcessor


# Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
INPUT_CHANNEL = "sensor_telemetry"
OUTPUT_CHANNEL = "live_features"


def run_collector():
    # Initialize Redis and Processor
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    processor = FeatureProcessor(window_size=20)

    pubsub = r.pubsub()
    pubsub.subscribe(INPUT_CHANNEL)

    print(f"🚀 Data Layer Active. Listening on '{INPUT_CHANNEL}'...")

    for message in pubsub.listen():
        if message["type"] == "message":
            try:
                data = json.loads(message["data"])

                # 1. VALIDATE
                if not validate_telemetry(data):
                    print(f"⚠️ Validation Failed: {data}")
                    continue

                # 2. STORE (Simulated: In a real app, write to InfluxDB here)
                # print(f"💾 Storing {data['sensor_id']} to Time-Series DB")

                # 3. PROCESS (Normalization)
                features = processor.process(data["sensor_id"], data["pressure"])

                # 4. STREAM TO AI LAYER
                r.publish(OUTPUT_CHANNEL, json.dumps(features))
                print(
                    f"✅ Streamed Features for {data['sensor_id']}: {features['normalized_pressure']:.4f}"
                )

            except Exception as e:
                print(f"❌ Error processing message: {e}")


if __name__ == "__main__":
    run_collector()
