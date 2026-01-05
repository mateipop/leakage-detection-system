import numpy as np

from leak_detect_ai_physics.data_layer.processor import FeatureProcessor


def test_process_uses_rolling_window():
    processor = FeatureProcessor(window_size=3)
    processor.process("S1", {"pressure": 1.0})
    processor.process("S1", {"pressure": 2.0})
    result = processor.process("S1", {"pressure": 3.0})

    std = np.std([1.0, 2.0, 3.0])
    expected = (3.0 - 2.0) / std
    assert np.isclose(result["normalized"]["pressure"], expected)

    result = processor.process("S1", {"pressure": 4.0})
    std = np.std([2.0, 3.0, 4.0])
    expected = (4.0 - 3.0) / std
    assert np.isclose(result["normalized"]["pressure"], expected)


def test_process_tracks_sensors_independently():
    processor = FeatureProcessor(window_size=2)
    processor.process("S1", {"pressure": 10.0})
    processor.process("S2", {"pressure": 100.0})
    result = processor.process("S1", {"pressure": 12.0})

    std = np.std([10.0, 12.0])
    expected = (12.0 - 11.0) / std
    assert np.isclose(result["normalized"]["pressure"], expected)
    assert result["sensor_id"] == "S1"
