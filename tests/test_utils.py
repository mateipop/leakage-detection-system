import numpy as np

from leak_detect_ai_physics.data_layer.utils import (
    calculate_z_score,
    validate_telemetry,
)


def test_validate_telemetry_accepts_required_fields():
    payload = {
        "entity_type": "node",
        "node_id": "N1",
        "sensor_id": "S1",
        "timestamp": 123,
        "node_metrics": {"pressure": 45.6, "head": 50.1, "demand": 1.2},
        "node_attributes": {"node_type": "Junction"},
    }
    assert validate_telemetry(payload) is True


def test_validate_telemetry_rejects_missing_fields():
    payload = {
        "entity_type": "node",
        "node_id": "N1",
        "sensor_id": "S1",
        "timestamp": 123,
        "node_metrics": {"pressure": 45.6, "head": 50.1},
    }
    assert validate_telemetry(payload) is False


def test_validate_telemetry_accepts_link_payload():
    payload = {
        "entity_type": "link",
        "link_id": "P1",
        "timestamp": 123,
        "link_metrics": {"flowrate": 1.2},
        "link_attributes": {"link_type": "Pipe"},
    }
    assert validate_telemetry(payload) is True


def test_calculate_z_score_handles_zero_std():
    assert calculate_z_score(10.0, mean=10.0, std=0.0) == 0.0


def test_calculate_z_score_matches_expected_value():
    value = 3.0
    mean = 2.0
    std = np.std([1.0, 2.0, 3.0])
    assert np.isclose(calculate_z_score(value, mean, std), (value - mean) / std)
