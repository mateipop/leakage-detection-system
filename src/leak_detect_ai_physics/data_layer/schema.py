from typing import Any

REQUIRED_FIELDS = [
    "entity_type",
    "entity_id",
    "sensor_id",
    "timestamp",
    "network",
    "coordinates",
    "features",
    "normalized",
    "leak_active",
    "label",
    "active_leak_nodes",
    "active_leak_pipes",
    "leak_targets",
    "active_leak_targets",
]

LEAK_TARGET_FIELDS = [
    "leak_id",
    "type",
    "node_id",
    "leak_node_id",
    "pipe_id",
    "new_pipe_id",
    "split_fraction",
    "area",
    "start_time",
    "end_time",
    "x",
    "y",
]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def _validate_coordinates(value: Any, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append("coordinates must be a dict")
        return
    if not _is_number(value.get("x")) or not _is_number(value.get("y")):
        errors.append("coordinates must contain numeric x/y values")


def _validate_leak_targets(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append("leak_targets must be a list")
        return
    for idx, target in enumerate(value):
        if not isinstance(target, dict):
            errors.append(f"leak_targets[{idx}] must be a dict")
            continue
        for field in LEAK_TARGET_FIELDS:
            if field not in target:
                errors.append(f"leak_targets[{idx}] missing {field}")
        if "x" in target and not _is_number(target["x"]):
            errors.append(f"leak_targets[{idx}].x must be numeric")
        if "y" in target and not _is_number(target["y"]):
            errors.append(f"leak_targets[{idx}].y must be numeric")


def validate_training_record(record: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return False, ["record must be a dict"]

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"missing {field}")

    if "entity_type" in record and not isinstance(record["entity_type"], str):
        errors.append("entity_type must be a string")
    if "entity_id" in record and not isinstance(record["entity_id"], str):
        errors.append("entity_id must be a string")
    if "sensor_id" in record and record["sensor_id"] is not None:
        if not isinstance(record["sensor_id"], str):
            errors.append("sensor_id must be a string or null")
    if "timestamp" in record and not isinstance(record["timestamp"], int):
        errors.append("timestamp must be an int")
    if "network" in record and record["network"] is not None:
        if not isinstance(record["network"], str):
            errors.append("network must be a string or null")

    if "coordinates" in record:
        _validate_coordinates(record["coordinates"], errors)

    for key in ("features", "normalized"):
        if key not in record:
            continue
        value = record[key]
        if not isinstance(value, dict):
            errors.append(f"{key} must be a dict")
            continue
        for feature_name, feature_value in value.items():
            if not isinstance(feature_name, str) or not _is_number(feature_value):
                errors.append(f"{key} entries must be string->number")
                break

    if "leak_active" in record and not isinstance(record["leak_active"], bool):
        errors.append("leak_active must be a bool")
    if "label" in record and not isinstance(record["label"], int):
        errors.append("label must be an int")
    for key in ("active_leak_nodes", "active_leak_pipes"):
        if key in record and not isinstance(record[key], list):
            errors.append(f"{key} must be a list")

    if "leak_targets" in record:
        _validate_leak_targets(record["leak_targets"], errors)
    if "active_leak_targets" in record:
        _validate_leak_targets(record["active_leak_targets"], errors)

    return len(errors) == 0, errors
