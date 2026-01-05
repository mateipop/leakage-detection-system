import numpy as np

from leak_detect_ai_physics.data_layer.utils import (
    calculate_z_score,
    validate_telemetry,
)


def test_validate_telemetry_accepts_required_fields():
    payload = {
        "sensor_id": "S1",
        "timestamp": 123,
        "pressure": 45.6,
        "head": 50.1,
        "demand": 1.2,
    }
    assert validate_telemetry(payload) is True


def test_validate_telemetry_rejects_missing_fields():
    payload = {"sensor_id": "S1", "pressure": 45.6, "head": 50.1}
    assert validate_telemetry(payload) is False


def test_calculate_z_score_handles_zero_std():
    assert calculate_z_score(10.0, mean=10.0, std=0.0) == 0.0


def test_calculate_z_score_matches_expected_value():
    value = 3.0
    mean = 2.0
    std = np.std([1.0, 2.0, 3.0])
    assert np.isclose(calculate_z_score(value, mean, std), (value - mean) / std)
