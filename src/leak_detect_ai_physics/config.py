from pathlib import Path

REDIS_HOST = "localhost"
REDIS_PORT = 6379

INPUT_CHANNEL = "sensor_telemetry"
OUTPUT_CHANNEL = "live_features"
AI_OUTPUT_CHANNEL = "ai_predictions"

NETWORK_NAME = "Net3"
LEAK_NODE_ID = "123"
LEAK_START_SECONDS = 4 * 3600
LEAK_END_SECONDS = 10 * 3600

TRAINING_DATA_PATH = Path("data/training_data.jsonl")
MODEL_PATH = Path("data/models/baseline.json")
