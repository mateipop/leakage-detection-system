import numpy as np

from data_layer.processor import FeatureProcessor


def test_process_uses_rolling_window():
    processor = FeatureProcessor(window_size=3)
    processor.process("S1", 1.0)
    processor.process("S1", 2.0)
    result = processor.process("S1", 3.0)

    std = np.std([1.0, 2.0, 3.0])
    expected = (3.0 - 2.0) / std
    assert np.isclose(result["normalized_pressure"], expected)

    result = processor.process("S1", 4.0)
    std = np.std([2.0, 3.0, 4.0])
    expected = (4.0 - 3.0) / std
    assert np.isclose(result["normalized_pressure"], expected)


def test_process_tracks_sensors_independently():
    processor = FeatureProcessor(window_size=2)
    processor.process("S1", 10.0)
    processor.process("S2", 100.0)
    result = processor.process("S1", 12.0)

    std = np.std([10.0, 12.0])
    expected = (12.0 - 11.0) / std
    assert np.isclose(result["normalized_pressure"], expected)
    assert result["sensor_id"] == "S1"
