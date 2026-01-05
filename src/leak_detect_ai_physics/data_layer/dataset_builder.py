import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import redis

from leak_detect_ai_physics import config


def _ensure_redis() -> None:
    client = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        decode_responses=True,
    )
    client.ping()


def run() -> int:
    parser = argparse.ArgumentParser(
        description="Run the collector and simulation to build a training dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.TRAINING_DATA_PATH,
        help="Path to write JSONL training data.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete the output file before starting.",
    )
    args = parser.parse_args()

    output_path = args.output
    if args.clear and output_path.exists():
        output_path.unlink()

    env = os.environ.copy()
    env["TRAINING_DATA_PATH"] = str(output_path)

    try:
        _ensure_redis()
    except Exception as exc:
        print(f"Redis unavailable at {config.REDIS_HOST}:{config.REDIS_PORT}: {exc}")
        return 1

    collector = subprocess.Popen(
        [sys.executable, "-m", "leak_detect_ai_physics.data_layer.collector"],
        env=env,
    )
    time.sleep(1.0)

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "leak_detect_ai_physics.simulation_layer.wntr_driver",
            ],
            env=env,
            check=True,
        )
    finally:
        collector.terminate()
        try:
            collector.wait(timeout=5)
        except subprocess.TimeoutExpired:
            collector.kill()

    print(f"Dataset saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
