import numpy as np
from collections import deque
from .utils import calculate_z_score


class FeatureProcessor:
    def __init__(self, window_size=50):
        self.window_size = window_size
        # Dictionary to store a rolling buffer for each sensor
        self.buffers = {}

    def process(self, sensor_id, raw_value):
        if sensor_id not in self.buffers:
            self.buffers[sensor_id] = deque(maxlen=self.window_size)

        # Add new value to the buffer
        self.buffers[sensor_id].append(raw_value)

        # Calculate statistics from the window
        window_data = list(self.buffers[sensor_id])
        mean = np.mean(window_data)
        std = np.std(window_data)

        # Normalize
        normalized_value = calculate_z_score(raw_value, mean, std)

        return {
            "sensor_id": sensor_id,
            "normalized_pressure": normalized_value,
            "is_feature": True,
        }
