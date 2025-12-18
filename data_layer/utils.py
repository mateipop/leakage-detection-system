import numpy as np


def validate_telemetry(data):
    """Checks if the incoming dictionary has all required fields."""
    required_fields = ["sensor_id", "timestamp", "pressure"]
    return all(field in data for field in required_fields)


def calculate_z_score(value, mean, std):
    """Applies standard Z-Score normalization: (x - μ) /"""
    if std == 0:
        return 0.0
    return (value - mean) / std
