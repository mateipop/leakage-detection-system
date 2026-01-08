"""
Feature Extractor - Prepare feature vectors for AI inference.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .data_pipeline import DataPipeline
from .telemetry_buffer import TelemetryRecord

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """Feature vector for AI inference."""
    node_id: str
    timestamp: float

    # Raw features
    pressure_zscore: Optional[float] = None
    flow_zscore: Optional[float] = None

    # Derived features
    pressure_trend: Optional[float] = None  # Slope over recent samples
    flow_trend: Optional[float] = None
    pressure_volatility: Optional[float] = None  # Std of recent values
    flow_volatility: Optional[float] = None

    # Contextual features
    neighbor_pressure_avg: Optional[float] = None
    pressure_deviation_from_neighbors: Optional[float] = None

    # Combined anomaly score (raw, before confidence calculation)
    raw_anomaly_score: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.pressure_zscore or 0.0,
            self.flow_zscore or 0.0,
            self.pressure_trend or 0.0,
            self.flow_trend or 0.0,
            self.pressure_volatility or 0.0,
            self.flow_volatility or 0.0,
            self.pressure_deviation_from_neighbors or 0.0
        ])


class FeatureExtractor:
    """
    Extracts feature vectors from processed telemetry data.

    Prepares normalized feature vectors for the AI inference engine.
    """

    def __init__(self, pipeline: DataPipeline, trend_window: int = 10):
        """
        Initialize the feature extractor.

        Args:
            pipeline: The data pipeline to extract features from
            trend_window: Number of samples for trend calculation
        """
        self._pipeline = pipeline
        self._trend_window = trend_window

    def _calculate_trend(
        self,
        records: List[TelemetryRecord]
    ) -> Optional[float]:
        """
        Calculate trend (slope) from recent records.

        Uses simple linear regression.
        """
        valid_records = [
            r for r in records
            if r.is_valid and r.filtered_value is not None
        ]

        if len(valid_records) < 3:
            return None

        # Simple linear regression
        n = len(valid_records)
        x = np.arange(n)
        y = np.array([r.filtered_value for r in valid_records])

        # Slope = (n * sum(xy) - sum(x) * sum(y)) / (n * sum(x^2) - sum(x)^2)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)

        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return float(slope)

    def _calculate_volatility(
        self,
        records: List[TelemetryRecord]
    ) -> Optional[float]:
        """Calculate volatility (standard deviation) of recent values."""
        valid_values = [
            r.filtered_value for r in records
            if r.is_valid and r.filtered_value is not None
        ]

        if len(valid_values) < 3:
            return None

        return float(np.std(valid_values))

    def extract_features(
        self,
        node_id: str,
        timestamp: float,
        neighbor_nodes: List[str] = None
    ) -> FeatureVector:
        """
        Extract feature vector for a single node.

        Args:
            node_id: Node to extract features for
            timestamp: Current timestamp
            neighbor_nodes: List of neighboring node IDs for spatial features

        Returns:
            FeatureVector with all computed features
        """
        features = FeatureVector(
            node_id=node_id,
            timestamp=timestamp
        )

        buffer = self._pipeline.buffer

        # Get recent pressure records
        pressure_records = buffer.get_recent(node_id, 'pressure', self._trend_window)
        if pressure_records:
            latest_pressure = pressure_records[0]
            features.pressure_zscore = latest_pressure.z_score
            features.pressure_trend = self._calculate_trend(pressure_records)
            features.pressure_volatility = self._calculate_volatility(pressure_records)

        # Get recent flow records
        flow_records = buffer.get_recent(node_id, 'flow', self._trend_window)
        if flow_records:
            latest_flow = flow_records[0]
            features.flow_zscore = latest_flow.z_score
            features.flow_trend = self._calculate_trend(flow_records)
            features.flow_volatility = self._calculate_volatility(flow_records)

        # Calculate neighbor-based features
        if neighbor_nodes and features.pressure_zscore is not None:
            neighbor_zscores = []
            for neighbor in neighbor_nodes:
                neighbor_records = buffer.get_recent(neighbor, 'pressure', 1)
                if neighbor_records and neighbor_records[0].z_score is not None:
                    neighbor_zscores.append(neighbor_records[0].z_score)

            if neighbor_zscores:
                features.neighbor_pressure_avg = float(np.mean(neighbor_zscores))
                features.pressure_deviation_from_neighbors = (
                    features.pressure_zscore - features.neighbor_pressure_avg
                )

        # Calculate raw anomaly score (simple weighted combination)
        score_components = []

        if features.pressure_zscore is not None:
            # Negative pressure Z-score indicates pressure drop (leak signature)
            score_components.append(abs(features.pressure_zscore) * 0.4)

        if features.pressure_trend is not None:
            # Negative trend indicates decreasing pressure
            if features.pressure_trend < 0:
                score_components.append(abs(features.pressure_trend) * 0.2)

        if features.pressure_deviation_from_neighbors is not None:
            # Large deviation from neighbors is suspicious
            score_components.append(
                abs(features.pressure_deviation_from_neighbors) * 0.3
            )

        if features.pressure_volatility is not None:
            # High volatility can indicate issues
            score_components.append(features.pressure_volatility * 0.1)

        features.raw_anomaly_score = sum(score_components) if score_components else 0.0

        return features

    def extract_all_features(
        self,
        node_ids: List[str],
        timestamp: float,
        topology_fn=None
    ) -> Dict[str, FeatureVector]:
        """
        Extract feature vectors for multiple nodes.

        Args:
            node_ids: List of node IDs
            timestamp: Current timestamp
            topology_fn: Function(node_id) -> List[str] returning neighbor nodes

        Returns:
            Dict mapping node_id to FeatureVector
        """
        features = {}
        for node_id in node_ids:
            neighbors = topology_fn(node_id) if topology_fn else None
            features[node_id] = self.extract_features(node_id, timestamp, neighbors)
        return features

    def get_top_anomalies(
        self,
        features: Dict[str, FeatureVector],
        top_n: int = 5
    ) -> List[FeatureVector]:
        """
        Get the top N nodes by anomaly score.

        Args:
            features: Dict of node features
            top_n: Number of top anomalies to return

        Returns:
            List of FeatureVectors sorted by anomaly score (descending)
        """
        sorted_features = sorted(
            features.values(),
            key=lambda f: f.raw_anomaly_score,
            reverse=True
        )
        return sorted_features[:top_n]
