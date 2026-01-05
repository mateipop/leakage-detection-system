def validate_node_payload(data):
    required_fields = [
        "entity_type",
        "node_id",
        "sensor_id",
        "timestamp",
        "node_metrics",
        "node_attributes",
    ]
    if not all(field in data for field in required_fields):
        return False
    return data.get("entity_type") == "node"


def validate_link_payload(data):
    required_fields = [
        "entity_type",
        "link_id",
        "timestamp",
        "link_metrics",
        "link_attributes",
    ]
    if not all(field in data for field in required_fields):
        return False
    return data.get("entity_type") == "link"


def validate_telemetry(data):
    """Checks if the incoming dictionary has all required fields."""
    if "entity_type" not in data:
        required_fields = ["sensor_id", "timestamp", "pressure", "head", "demand"]
        return all(field in data for field in required_fields)
    if data["entity_type"] == "node":
        return validate_node_payload(data)
    if data["entity_type"] == "link":
        return validate_link_payload(data)
    return False


def calculate_z_score(value, mean, std):
    """Applies standard Z-Score normalization: (x - μ) /"""
    if std == 0:
        return 0.0
    return (value - mean) / std
