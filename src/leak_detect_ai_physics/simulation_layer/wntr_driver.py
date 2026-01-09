import argparse
import json
import logging
import random
import time

import redis
import wntr

from leak_detect_ai_physics import config

LOG = logging.getLogger(__name__)

SIM_DURATION_SECONDS = 24 * 3600
SIM_TIMESTEP_SECONDS = 60
MAX_JUNCTION_LEAKS = 8
MAX_PIPE_LEAKS = 8
LEAK_AREA_RANGE = (0.01, 0.08)
LEAK_DURATION_RANGE = (2 * 3600, 10 * 3600)
REALTIME_DELAY_SECONDS = 0.01


def random_leak_window(
    rng,
    duration_seconds,
    *,
    start_min: int | None = None,
    start_max: int | None = None,
):
    latest_start = max(1, duration_seconds - LEAK_DURATION_RANGE[0])
    start_low = 1 if start_min is None else max(1, start_min)
    start_high = latest_start if start_max is None else min(start_max, latest_start)
    if start_low > start_high:
        start_low = start_high
    start_time = rng.randint(start_low, start_high)
    max_duration = min(LEAK_DURATION_RANGE[1], duration_seconds - start_time)
    if max_duration <= 0:
        return start_time, start_time
    if max_duration < LEAK_DURATION_RANGE[0]:
        duration = rng.randint(1, max_duration)
    else:
        duration = rng.randint(LEAK_DURATION_RANGE[0], max_duration)
    end_time = start_time + duration
    return start_time, end_time


def build_leak_plan(
    wn,
    duration_seconds,
    seed=42,
    *,
    leak_start_min: int | None = None,
    leak_start_max: int | None = None,
    max_junction_leaks: int = MAX_JUNCTION_LEAKS,
    max_pipe_leaks: int = MAX_PIPE_LEAKS,
    max_total_leaks: int | None = None,
):
    rng = random.Random(seed)
    plan = []
    remaining = None if max_total_leaks is None else max(0, max_total_leaks)
    max_junction_leaks = max(0, max_junction_leaks)
    max_pipe_leaks = max(0, max_pipe_leaks)

    junctions = list(wn.junction_name_list)
    if junctions and max_junction_leaks > 0:
        if remaining == 0:
            return plan
        junction_cap = min(max_junction_leaks, len(junctions))
        if remaining is not None:
            junction_cap = min(junction_cap, remaining)
        junction_leaks = rng.randint(1, junction_cap)
        leak_nodes = rng.sample(junctions, k=junction_leaks)
        for idx, node_id in enumerate(leak_nodes, start=1):
            if remaining == 0:
                return plan
            start_time, end_time = random_leak_window(
                rng,
                duration_seconds,
                start_min=leak_start_min,
                start_max=leak_start_max,
            )
            area = rng.uniform(*LEAK_AREA_RANGE)
            node = wn.get_node(node_id)
            node.add_leak(wn, area=area, start_time=start_time, end_time=end_time)
            plan.append(
                {
                    "leak_id": f"junction_{idx}",
                    "type": "junction",
                    "node_id": node_id,
                    "leak_node_id": node_id,
                    "pipe_id": None,
                    "new_pipe_id": None,
                    "split_fraction": None,
                    "area": area,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )
            if remaining is not None:
                remaining -= 1

    pipes = list(wn.pipe_name_list)
    if pipes and max_pipe_leaks > 0:
        if remaining == 0:
            return plan
        pipe_cap = min(max_pipe_leaks, len(pipes))
        if remaining is not None:
            pipe_cap = min(pipe_cap, remaining)
        pipe_leaks = rng.randint(0, pipe_cap)
        leak_pipes = rng.sample(pipes, k=pipe_leaks)
        for idx, pipe_id in enumerate(leak_pipes, start=1):
            if remaining == 0:
                return plan
            split_fraction = rng.uniform(0.2, 0.8)
            new_pipe_id = f"{pipe_id}_leak_seg_{idx}"
            leak_node_id = f"{pipe_id}_leak_node_{idx}"
            wntr.morph.split_pipe(
                wn,
                pipe_id,
                new_pipe_name=new_pipe_id,
                new_junction_name=leak_node_id,
                add_pipe_at_end=True,
                split_at_point=split_fraction,
                return_copy=False,
            )

            start_time, end_time = random_leak_window(
                rng,
                duration_seconds,
                start_min=leak_start_min,
                start_max=leak_start_max,
            )
            area = rng.uniform(*LEAK_AREA_RANGE)
            leak_node = wn.get_node(leak_node_id)
            leak_node.add_leak(
                wn,
                area=area,
                start_time=start_time,
                end_time=end_time,
            )
            plan.append(
                {
                    "leak_id": f"pipe_{idx}",
                    "type": "pipe",
                    "node_id": leak_node_id,
                    "leak_node_id": leak_node_id,
                    "pipe_id": pipe_id,
                    "new_pipe_id": new_pipe_id,
                    "split_fraction": split_fraction,
                    "area": area,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )
            if remaining is not None:
                remaining -= 1

    return plan


def leak_active_sets(leak_plan, timestamp):
    active = [
        leak
        for leak in leak_plan
        if leak["start_time"] <= timestamp <= leak["end_time"]
    ]
    node_ids = {leak["leak_node_id"] for leak in active}
    pipe_ids = {leak["pipe_id"] for leak in active if leak["pipe_id"]}
    return active, node_ids, pipe_ids


def safe_float(value):
    if value is None:
        return 0.0
    return float(value)


def node_attributes(node):
    return {
        "node_type": type(node).__name__,
        "elevation": safe_float(getattr(node, "elevation", 0.0)),
        "base_demand": safe_float(getattr(node, "base_demand", 0.0)),
        "emitter_coefficient": safe_float(getattr(node, "emitter_coefficient", 0.0)),
    }


def link_attributes(link):
    return {
        "link_type": type(link).__name__,
        "diameter": safe_float(getattr(link, "diameter", 0.0)),
        "length": safe_float(getattr(link, "length", 0.0)),
        "roughness": safe_float(getattr(link, "roughness", 0.0)),
        "minor_loss": safe_float(getattr(link, "minor_loss", 0.0)),
        "status": str(getattr(link, "status", "")),
        "initial_status": str(getattr(link, "initial_status", "")),
    }


def build_node_payload(
    node_id,
    timestamp,
    metrics,
    attributes,
    leak_plan,
    active_leaks,
    active_leak_nodes,
    active_leak_pipes,
):
    return {
        "entity_type": "node",
        "sensor_id": f"SENSOR_{node_id}",
        "node_id": node_id,
        "timestamp": int(timestamp),
        "network": config.NETWORK_NAME,
        "node_attributes": attributes,
        "node_metrics": metrics,
        "pressure": metrics.get("pressure"),
        "head": metrics.get("head"),
        "demand": metrics.get("demand"),
        "leak_active": node_id in active_leak_nodes,
        "active_leaks": active_leaks,
        "active_leak_nodes": list(active_leak_nodes),
        "active_leak_pipes": list(active_leak_pipes),
        "leak_plan": leak_plan,
    }


def build_link_payload(
    link_id,
    timestamp,
    metrics,
    attributes,
    leak_plan,
    active_leaks,
    active_leak_nodes,
    active_leak_pipes,
):
    return {
        "entity_type": "link",
        "link_id": link_id,
        "timestamp": int(timestamp),
        "network": config.NETWORK_NAME,
        "link_attributes": attributes,
        "link_metrics": metrics,
        "leak_active": link_id in active_leak_pipes,
        "active_leaks": active_leaks,
        "active_leak_nodes": list(active_leak_nodes),
        "active_leak_pipes": list(active_leak_pipes),
        "leak_plan": leak_plan,
    }


def run_simulation(
    *,
    duration_seconds: int,
    timestep_seconds: int,
    seed: int,
    realtime_delay_seconds: float,
    leak_start_min: int | None,
    leak_start_max: int | None,
):
    wn = wntr.network.WaterNetworkModel(config.NETWORK_NAME)

    leak_plan = build_leak_plan(
        wn,
        duration_seconds,
        seed=seed,
        leak_start_min=leak_start_min,
        leak_start_max=leak_start_max,
    )

    wn.options.time.duration = duration_seconds
    wn.options.time.hydraulic_timestep = timestep_seconds
    wn.options.time.report_timestep = timestep_seconds
    wn.options.time.pattern_timestep = timestep_seconds

    LOG.info("Running WNTR Hydraulic Engine...")
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    r = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        decode_responses=True,
    )

    node_result_keys = list(results.node.keys())
    link_result_keys = list(results.link.keys())

    timestamps = results.node[node_result_keys[0]].index
    node_ids = list(results.node[node_result_keys[0]].columns)
    link_ids = list(results.link[link_result_keys[0]].columns)

    LOG.info("Starting real-time stream to Data Layer...")
    LOG.debug("Leak plan: %s", leak_plan)

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
            r.publish(config.INPUT_CHANNEL, json.dumps(payload))
            LOG.debug(
                "Node payload sent: %s",
                {
                    "node_id": node_id,
                    "timestamp": int(timestamp),
                    "metrics": list(metrics.keys()),
                },
            )

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
            r.publish(config.INPUT_CHANNEL, json.dumps(payload))
            LOG.debug(
                "Link payload sent: %s",
                {
                    "link_id": link_id,
                    "timestamp": int(timestamp),
                    "metrics": list(metrics.keys()),
                },
            )

        message = (
            "Time: {hours:>4.1f} hrs | Nodes: {node_count} | Links: {link_count}"
        ).format(
            hours=timestamp / 3600,
            node_count=len(node_ids),
            link_count=len(link_ids),
        )
        LOG.info(message)
        time.sleep(realtime_delay_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run WNTR simulation and publish telemetry."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=SIM_DURATION_SECONDS,
        help="Simulation duration in seconds.",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=SIM_TIMESTEP_SECONDS,
        help="Simulation timestep in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for leak placement.",
    )
    parser.add_argument(
        "--realtime-delay",
        type=float,
        default=REALTIME_DELAY_SECONDS,
        help="Delay between published timesteps.",
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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run_simulation(
            duration_seconds=args.duration,
            timestep_seconds=args.timestep,
            seed=args.seed,
            realtime_delay_seconds=args.realtime_delay,
            leak_start_min=args.leak_start_min,
            leak_start_max=args.leak_start_max,
        )
    except Exception as e:
        LOG.exception("Simulation error: %s", e)
