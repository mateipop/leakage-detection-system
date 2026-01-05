import json

import redis
import wntr

from leak_detect_ai_physics import config
from leak_detect_ai_physics.data_layer.processor import FeatureProcessor
from leak_detect_ai_physics.data_layer.storage import append_training_record
from leak_detect_ai_physics.data_layer.utils import validate_telemetry


def build_coordinate_maps(wn):
    node_coords = {}
    for node_id in wn.node_name_list:
        node = wn.get_node(node_id)
        x, y = node.coordinates
        node_coords[node_id] = {"x": float(x), "y": float(y)}

    link_coords = {}
    for link_id in wn.link_name_list:
        link = wn.get_link(link_id)
        start = wn.get_node(link.start_node_name).coordinates
        end = wn.get_node(link.end_node_name).coordinates
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        link_coords[link_id] = {"x": float(mid_x), "y": float(mid_y)}

    return node_coords, link_coords


def build_leak_targets(leak_plan, node_coords):
    targets = []
    for leak in leak_plan or []:
        node_id = leak.get("leak_node_id") or leak.get("node_id")
        coords = node_coords.get(node_id, {"x": 0.0, "y": 0.0})
        targets.append(
            {
                "leak_id": leak.get("leak_id"),
                "type": leak.get("type"),
                "node_id": leak.get("node_id"),
                "leak_node_id": node_id,
                "pipe_id": leak.get("pipe_id"),
                "new_pipe_id": leak.get("new_pipe_id"),
                "split_fraction": leak.get("split_fraction"),
                "area": leak.get("area"),
                "start_time": leak.get("start_time"),
                "end_time": leak.get("end_time"),
                "x": coords["x"],
                "y": coords["y"],
            }
        )
    return targets


def select_active_targets(active_leaks, leak_targets):
    by_id = {target["leak_id"]: target for target in leak_targets}
    active = []
    for leak in active_leaks or []:
        leak_id = leak.get("leak_id")
        if leak_id in by_id:
            active.append(by_id[leak_id])
    return active


def build_training_record(
    data,
    raw_features,
    normalized_features,
    coordinates,
    leak_targets,
    active_leak_targets,
):
    entity_type = data["entity_type"]
    entity_id = data["node_id"] if entity_type == "node" else data["link_id"]

    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "sensor_id": data.get("sensor_id"),
        "timestamp": data["timestamp"],
        "network": data.get("network"),
        "coordinates": coordinates,
        "features": raw_features,
        "normalized": normalized_features,
        "leak_active": data.get("leak_active", False),
        "label": int(bool(data.get("leak_active", False))),
        "active_leak_nodes": data.get("active_leak_nodes", []),
        "active_leak_pipes": data.get("active_leak_pipes", []),
        "leak_targets": leak_targets,
        "active_leak_targets": active_leak_targets,
    }


def run_collector():
    r = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        decode_responses=True,
    )
    processor = FeatureProcessor(window_size=20)
    wn = wntr.network.WaterNetworkModel(config.NETWORK_NAME)
    node_coords, link_coords = build_coordinate_maps(wn)

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

                if data.get("entity_type") == "link":
                    raw_features = data["link_metrics"]
                    entity_id = data["link_id"]
                    coordinates = link_coords.get(entity_id, {"x": 0.0, "y": 0.0})
                else:
                    raw_features = data["node_metrics"]
                    entity_id = data["node_id"]
                    coordinates = node_coords.get(entity_id, {"x": 0.0, "y": 0.0})

                processor_key = f"{data.get('entity_type', 'node')}:{entity_id}"
                features = processor.process(processor_key, raw_features)

                leak_targets = build_leak_targets(data.get("leak_plan"), node_coords)
                active_leak_targets = select_active_targets(
                    data.get("active_leaks"), leak_targets
                )
                training_record = build_training_record(
                    data,
                    raw_features,
                    features["normalized"],
                    coordinates,
                    leak_targets,
                    active_leak_targets,
                )
                append_training_record(config.TRAINING_DATA_PATH, training_record)

                features["timestamp"] = data["timestamp"]
                features["entity_type"] = data.get("entity_type", "node")
                features["entity_id"] = entity_id
                features["sensor_id"] = data.get("sensor_id")
                r.publish(config.OUTPUT_CHANNEL, json.dumps(features))
                print(
                    "Streamed features for {sensor}: {normalized}".format(
                        sensor=features.get("sensor_id") or entity_id,
                        normalized=features["normalized"],
                    )
                )

            except Exception as e:
                print(f"Error processing message: {e}")


if __name__ == "__main__":
    run_collector()
