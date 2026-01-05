from collections import deque

import numpy as np

from .utils import calculate_z_score


class FeatureProcessor:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.buffers = {}

    def process(self, sensor_id, raw_features):
        if sensor_id not in self.buffers:
            self.buffers[sensor_id] = {}

        normalized = {}
        for feature_name, raw_value in raw_features.items():
            if feature_name not in self.buffers[sensor_id]:
                self.buffers[sensor_id][feature_name] = deque(maxlen=self.window_size)

            self.buffers[sensor_id][feature_name].append(raw_value)

            window_data = list(self.buffers[sensor_id][feature_name])
            mean = np.mean(window_data)
            std = np.std(window_data)

            normalized[feature_name] = calculate_z_score(raw_value, mean, std)

        return {
            "sensor_id": sensor_id,
            "normalized": normalized,
            "is_feature": True,
        }
