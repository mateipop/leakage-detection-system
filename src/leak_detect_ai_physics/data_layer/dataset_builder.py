import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import redis
import wntr

from leak_detect_ai_physics import config
from leak_detect_ai_physics.data_layer.collector import (
    build_coordinate_maps,
    build_leak_targets,
    build_training_record,
    select_active_targets,
)
from leak_detect_ai_physics.data_layer.processor import FeatureProcessor
from leak_detect_ai_physics.data_layer.storage import append_training_record
from leak_detect_ai_physics.simulation_layer.wntr_driver import (
    build_leak_plan,
    build_link_payload,
    build_node_payload,
    leak_active_sets,
    link_attributes,
    node_attributes,
)

LOG = logging.getLogger(__name__)


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
    parser.add_argument(
        "--mode",
        choices=["direct", "redis"],
        default="direct",
        help="Dataset builder mode: direct (no Redis) or redis.",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=3,
        help="Window size in hours.",
    )
    parser.add_argument(
        "--stride-steps",
        type=int,
        default=6,
        help="Stride in timesteps between windows.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["any"],
        default="any",
        help="Window label mode (any = leak present anywhere in window).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Simulation duration in seconds.",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=None,
        help="Simulation timestep in seconds.",
    )
    parser.add_argument(
        "--realtime-delay",
        type=float,
        default=None,
        help="Delay between published timesteps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for leak placement.",
    )
    parser.add_argument(
        "--leak-start-min",
        type=int,
        default=None,
        help="Minimum leak start time in seconds.",
    )
    parser.add_argument(
        "--leak-start-max",
        type=int,
        default=None,
        help="Maximum leak start time in seconds.",
    )
    args = parser.parse_args()

    output_path = args.output
    if args.clear and output_path.exists():
        output_path.unlink()

    env = os.environ.copy()
    env["TRAINING_DATA_PATH"] = str(output_path)

    if args.mode == "redis":
        try:
            _ensure_redis()
        except Exception as exc:
            LOG.error(
                "Redis unavailable at %s:%s: %s",
                config.REDIS_HOST,
                config.REDIS_PORT,
                exc,
            )
            return 1

        collector = subprocess.Popen(
            [sys.executable, "-m", "leak_detect_ai_physics.data_layer.collector"],
            env=env,
        )
        time.sleep(1.0)

        try:
            cmd = [
                sys.executable,
                "-m",
                "leak_detect_ai_physics.simulation_layer.wntr_driver",
            ]
            if args.duration is not None:
                cmd += ["--duration", str(args.duration)]
            if args.timestep is not None:
                cmd += ["--timestep", str(args.timestep)]
            if args.realtime_delay is not None:
                cmd += ["--realtime-delay", str(args.realtime_delay)]
            if args.seed is not None:
                cmd += ["--seed", str(args.seed)]
            if args.leak_start_min is not None:
                cmd += ["--leak-start-min", str(args.leak_start_min)]
            if args.leak_start_max is not None:
                cmd += ["--leak-start-max", str(args.leak_start_max)]

            subprocess.run(
                cmd,
                env=env,
                check=True,
            )
        finally:
            collector.terminate()
            try:
                collector.wait(timeout=5)
            except subprocess.TimeoutExpired:
                collector.kill()
    else:
        duration = args.duration or 24 * 3600
        timestep = args.timestep or 300
        seed = args.seed or 42
        leak_start_min = args.leak_start_min
        leak_start_max = args.leak_start_max
        window_steps = int((args.window_hours * 3600) / timestep)
        stride_steps = max(1, args.stride_steps)
        window_seconds = window_steps * timestep
        if leak_start_min is None:
            leak_start_min = max(1, window_seconds)
        if leak_start_max is None:
            leak_start_max = max(leak_start_min, duration - window_seconds)

        wn = wntr.network.WaterNetworkModel(config.NETWORK_NAME)
        leak_plan = build_leak_plan(
            wn,
            duration,
            seed=seed,
            leak_start_min=leak_start_min,
            leak_start_max=leak_start_max,
            max_junction_leaks=2,
            max_pipe_leaks=3,
            max_total_leaks=4,
            non_overlapping=True,
        )
        if not leak_plan:
            leak_plan = build_leak_plan(
                wn,
                duration,
                seed=seed,
                leak_start_min=leak_start_min,
                leak_start_max=leak_start_max,
                max_junction_leaks=2,
                max_pipe_leaks=2,
                max_total_leaks=4,
                non_overlapping=True,
            )
        if not leak_plan:
            LOG.error("Failed to generate a leak plan for dataset build.")
            return 1
        wn.options.time.duration = duration
        wn.options.time.hydraulic_timestep = timestep
        wn.options.time.report_timestep = timestep
        wn.options.time.pattern_timestep = timestep

        LOG.info("Running WNTR simulation for dataset build...")
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()

        node_coords, link_coords = build_coordinate_maps(wn)
        processor = FeatureProcessor(window_size=20)

        node_result_keys = list(results.node.keys())
        link_result_keys = list(results.link.keys())
        timestamps = results.node[node_result_keys[0]].index
        node_ids = list(results.node[node_result_keys[0]].columns)
        link_ids = list(results.link[link_result_keys[0]].columns)

        entity_series = {}

        for timestamp in timestamps:
            active_leaks, active_leak_nodes, active_leak_pipes = leak_active_sets(
                leak_plan, timestamp
            )

            for node_id in node_ids:
                node = wn.get_node(node_id)
                metrics = {
                    key: float(results.node[key].loc[timestamp, node_id])
                    for key in node_result_keys
                }
                payload = build_node_payload(
                    node_id=node_id,
                    timestamp=timestamp,
                    metrics=metrics,
                    attributes=node_attributes(node),
                    leak_plan=leak_plan,
                    active_leaks=active_leaks,
                    active_leak_nodes=active_leak_nodes,
                    active_leak_pipes=active_leak_pipes,
                )
                record = _process_payload(
                    payload,
                    processor,
                    node_coords,
                    link_coords,
                    output_path,
                )
                _store_series(entity_series, record, active_leaks)

            for link_id in link_ids:
                link = wn.get_link(link_id)
                metrics = {
                    key: float(results.link[key].loc[timestamp, link_id])
                    for key in link_result_keys
                }
                payload = build_link_payload(
                    link_id=link_id,
                    timestamp=timestamp,
                    metrics=metrics,
                    attributes=link_attributes(link),
                    leak_plan=leak_plan,
                    active_leaks=active_leaks,
                    active_leak_nodes=active_leak_nodes,
                    active_leak_pipes=active_leak_pipes,
                )
                record = _process_payload(
                    payload,
                    processor,
                    node_coords,
                    link_coords,
                    output_path,
                )
                _store_series(entity_series, record, active_leaks)

        _write_windowed_dataset(
            entity_series,
            output_path,
            window_steps=window_steps,
            stride_steps=stride_steps,
            node_coords=node_coords,
            seed=args.seed,
        )
    LOG.info("Dataset saved to %s", output_path)
    return 0


def _process_payload(
    payload: dict,
    processor: FeatureProcessor,
    node_coords: dict,
    link_coords: dict,
    output_path: Path,
) -> dict:
    if payload.get("entity_type") == "link":
        raw_features = payload["link_metrics"]
        entity_id = payload["link_id"]
        coordinates = link_coords.get(entity_id, {"x": 0.0, "y": 0.0})
    else:
        raw_features = payload["node_metrics"]
        entity_id = payload["node_id"]
        coordinates = node_coords.get(entity_id, {"x": 0.0, "y": 0.0})

    processor_key = f"{payload.get('entity_type', 'node')}:{entity_id}"
    features = processor.process(processor_key, raw_features)

    leak_targets = build_leak_targets(payload.get("leak_plan"), node_coords)
    active_leak_targets = select_active_targets(
        payload.get("active_leaks"), leak_targets
    )
    training_record = build_training_record(
        payload,
        raw_features,
        features["normalized"],
        coordinates,
        leak_targets,
        active_leak_targets,
    )
    append_training_record(output_path, training_record)
    return training_record


def _store_series(entity_series: dict, record: dict, active_leaks: list) -> None:
    entity_key = (record["entity_type"], record["entity_id"])
    if entity_key not in entity_series:
        entity_series[entity_key] = []
    entity_series[entity_key].append(
        {
            "timestamp": record["timestamp"],
            "normalized": record["normalized"],
            "leak_any": bool(active_leaks),
            "active_leak_nodes": record.get("active_leak_nodes", []),
            "coordinates": record.get("coordinates", {}),
            "entity_id": record.get("entity_id"),
            "entity_type": record.get("entity_type"),
        }
    )


def _write_windowed_dataset(
    entity_series: dict,
    output_path: Path,
    *,
    window_steps: int,
    stride_steps: int,
    node_coords: dict,
    seed: int | None,
) -> None:
    if window_steps <= 1:
        return
    feature_names = _collect_feature_names(entity_series)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    leak_records = []
    no_leak_records = []
    for series in entity_series.values():
        if len(series) < window_steps:
            continue
        for start in range(0, len(series) - window_steps + 1, stride_steps):
            window = series[start : start + window_steps]
            leak_nodes = _window_leak_nodes(window)
            if len(leak_nodes) > 1:
                continue
            leak_node_id = next(iter(leak_nodes)) if leak_nodes else None
            leak_coords = None
            if leak_node_id:
                coords = node_coords.get(leak_node_id)
                if coords:
                    leak_coords = {
                        "x": float(coords.get("x", 0.0)),
                        "y": float(coords.get("y", 0.0)),
                    }
            window_features = [
                [float(step["normalized"].get(name, 0.0)) for name in feature_names]
                for step in window
            ]
            record = {
                "entity_type": window[0]["entity_type"],
                "entity_id": window[0]["entity_id"],
                "timestamp_start": window[0]["timestamp"],
                "timestamp_end": window[-1]["timestamp"],
                "features": window_features,
                "feature_names": feature_names,
                "label": int(bool(leak_node_id)),
                "leak_node_id": leak_node_id,
                "leak_coords": leak_coords,
            }
            if leak_node_id:
                leak_records.append(record)
            else:
                no_leak_records.append(record)

    if not leak_records or not no_leak_records:
        LOG.warning(
            "Cannot balance leak labels (leak=%s, no_leak=%s). Using all windows.",
            len(leak_records),
            len(no_leak_records),
        )
        records = leak_records + no_leak_records
    else:
        target = min(len(leak_records), len(no_leak_records))
        rng = random.Random(seed)
        rng.shuffle(leak_records)
        rng.shuffle(no_leak_records)
        records = leak_records[:target] + no_leak_records[:target]
        rng.shuffle(records)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _collect_feature_names(entity_series: dict) -> list[str]:
    names = set()
    for series in entity_series.values():
        for step in series:
            names.update(step["normalized"].keys())
    return sorted(names)


def _window_leak_nodes(window: list[dict]) -> set[str]:
    leak_nodes = set()
    for step in window:
        leak_nodes.update(step.get("active_leak_nodes") or [])
    return leak_nodes


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(run())
