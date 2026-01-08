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
    
    # === NEW ADVANCED FEATURES ===
    # Rate-of-change detection (first derivative)
    pressure_rate_of_change: Optional[float] = None  # Instantaneous change rate
    pressure_acceleration: Optional[float] = None    # Second derivative (rate of rate change)
    
    # Gradient-based features (pressure wave detection)
    pressure_gradient_magnitude: Optional[float] = None  # Spatial gradient strength
    gradient_direction_score: Optional[float] = None     # Consistency of gradient direction
    
    # Cross-sensor correlation features
    cross_correlation_score: Optional[float] = None      # Correlation with neighbors over time
    lag_to_neighbors: Optional[float] = None             # Time lag indicating wave propagation
    
    # Adaptive baseline features
    adaptive_zscore: Optional[float] = None              # Z-score with time-of-day baseline
    baseline_deviation: Optional[float] = None           # Deviation from expected baseline
    
    # Cumulative sum (CUSUM) for detecting small persistent shifts
    cusum_score: Optional[float] = None                  # Cumulative deviation score
    
    # Night/day flow anomaly (leaks more detectable at night)
    night_flow_ratio: Optional[float] = None             # Ratio vs expected night minimum

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.pressure_zscore or 0.0,
            self.flow_zscore or 0.0,
            self.pressure_trend or 0.0,
            self.flow_trend or 0.0,
            self.pressure_volatility or 0.0,
            self.flow_volatility or 0.0,
            self.pressure_deviation_from_neighbors or 0.0,
            # New advanced features
            self.pressure_rate_of_change or 0.0,
            self.pressure_acceleration or 0.0,
            self.pressure_gradient_magnitude or 0.0,
            self.cross_correlation_score or 0.0,
            self.adaptive_zscore or 0.0,
            self.cusum_score or 0.0,
            self.night_flow_ratio or 0.0
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
    
    def _calculate_rate_of_change(
        self,
        records: List[TelemetryRecord]
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate instantaneous rate of change and acceleration.
        
        Returns:
            (rate_of_change, acceleration) - first and second derivatives
        """
        valid_records = [
            r for r in records
            if r.is_valid and r.filtered_value is not None
        ]
        
        if len(valid_records) < 2:
            return None, None
        
        # Rate of change: difference between last two values
        values = [r.filtered_value for r in valid_records]
        times = [r.timestamp for r in valid_records]
        
        dt = times[0] - times[1] if len(times) > 1 else 1.0
        if dt == 0:
            dt = 1.0
            
        rate_of_change = (values[0] - values[1]) / dt
        
        # Acceleration: change in rate of change
        acceleration = None
        if len(valid_records) >= 3:
            prev_rate = (values[1] - values[2]) / dt
            acceleration = (rate_of_change - prev_rate) / dt
        
        return float(rate_of_change), float(acceleration) if acceleration else None
    
    def _calculate_cusum(
        self,
        records: List[TelemetryRecord],
        target_mean: float = None,
        k: float = 0.5
    ) -> Optional[float]:
        """
        Calculate CUSUM (Cumulative Sum) score for detecting small shifts.
        
        CUSUM is excellent at detecting small, persistent changes that
        Z-score might miss. Ideal for slow leaks.
        
        Args:
            records: Recent telemetry records
            target_mean: Expected mean (uses window mean if None)
            k: Allowance parameter (slack for normal variation)
        """
        valid_values = [
            r.filtered_value for r in records
            if r.is_valid and r.filtered_value is not None
        ]
        
        if len(valid_values) < 5:
            return None
        
        values = np.array(valid_values)
        
        if target_mean is None:
            # Use first half as baseline
            baseline = values[len(values)//2:]
            target_mean = np.mean(baseline)
        
        std = np.std(values) if np.std(values) > 0 else 1.0
        
        # Calculate CUSUM for negative shift (pressure drop)
        cusum_neg = 0.0
        cusum_pos = 0.0
        
        for val in values:
            normalized = (val - target_mean) / std
            cusum_neg = max(0, cusum_neg - normalized - k)
            cusum_pos = max(0, cusum_pos + normalized - k)
        
        # Return the larger deviation (we care more about pressure drops)
        return float(max(cusum_neg, cusum_pos))
    
    def _calculate_cross_correlation(
        self,
        node_id: str,
        neighbor_nodes: List[str],
        window: int = 10
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate cross-correlation with neighboring sensors.
        
        Leaks cause correlated pressure drops that propagate through the network.
        High correlation with lag suggests a pressure wave.
        
        Returns:
            (correlation_score, lag_estimate)
        """
        buffer = self._pipeline.buffer
        
        # Get our pressure history
        our_records = buffer.get_recent(node_id, 'pressure', window)
        our_values = [
            r.filtered_value for r in our_records
            if r.is_valid and r.filtered_value is not None
        ]
        
        if len(our_values) < 5 or not neighbor_nodes:
            return None, None
        
        correlations = []
        lags = []
        
        for neighbor in neighbor_nodes[:5]:  # Limit to 5 nearest neighbors
            neighbor_records = buffer.get_recent(neighbor, 'pressure', window)
            neighbor_values = [
                r.filtered_value for r in neighbor_records
                if r.is_valid and r.filtered_value is not None
            ]
            
            if len(neighbor_values) < 5:
                continue
            
            # Align lengths
            min_len = min(len(our_values), len(neighbor_values))
            a = np.array(our_values[:min_len])
            b = np.array(neighbor_values[:min_len])
            
            # Normalize
            a = (a - np.mean(a)) / (np.std(a) + 1e-6)
            b = (b - np.mean(b)) / (np.std(b) + 1e-6)
            
            # Cross-correlation
            corr = np.correlate(a, b, mode='full')
            max_corr_idx = np.argmax(np.abs(corr))
            lag = max_corr_idx - (len(a) - 1)
            max_corr = corr[max_corr_idx] / len(a)
            
            correlations.append(max_corr)
            lags.append(lag)
        
        if not correlations:
            return None, None
        
        avg_correlation = float(np.mean(correlations))
        avg_lag = float(np.mean(lags))
        
        return avg_correlation, avg_lag

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
            
            # === NEW: Rate of change and acceleration ===
            rate, accel = self._calculate_rate_of_change(pressure_records)
            features.pressure_rate_of_change = rate
            features.pressure_acceleration = accel
            
            # === NEW: CUSUM for detecting small persistent shifts ===
            features.cusum_score = self._calculate_cusum(pressure_records)

        # Get recent flow records
        flow_records = buffer.get_recent(node_id, 'flow', self._trend_window)
        if flow_records:
            latest_flow = flow_records[0]
            features.flow_zscore = latest_flow.z_score
            features.flow_trend = self._calculate_trend(flow_records)
            features.flow_volatility = self._calculate_volatility(flow_records)
            
            # === NEW: Night flow ratio (leaks more detectable at night) ===
            # Simulate time-of-day awareness (timestamp in seconds from midnight)
            hour = (timestamp / 3600) % 24
            is_night = hour < 6 or hour > 22
            if is_night and latest_flow.filtered_value is not None:
                # At night, flow should be minimal - any significant flow is suspicious
                stats = buffer.get_windowed_statistics(node_id, 'flow')
                if stats and stats['mean'] > 0:
                    features.night_flow_ratio = latest_flow.filtered_value / max(stats['mean'], 0.01)

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
            
            # === NEW: Cross-correlation with neighbors ===
            corr, lag = self._calculate_cross_correlation(node_id, neighbor_nodes)
            features.cross_correlation_score = corr
            features.lag_to_neighbors = lag
            
            # === NEW: Pressure gradient magnitude (spatial derivative) ===
            if neighbor_zscores:
                # Gradient = how different this node is from its neighbors
                # Large gradient suggests pressure wave or leak nearby
                gradient = features.pressure_zscore - features.neighbor_pressure_avg
                features.pressure_gradient_magnitude = abs(gradient)
                
                # Gradient consistency - are all neighbors showing similar pattern?
                if len(neighbor_zscores) > 1:
                    neighbor_std = np.std(neighbor_zscores)
                    # Low neighbor variance + high gradient = consistent directional pressure wave
                    if neighbor_std < 0.5:
                        features.gradient_direction_score = abs(gradient) / (neighbor_std + 0.1)
                    else:
                        features.gradient_direction_score = abs(gradient) / neighbor_std

        # Calculate raw anomaly score (enhanced with new features)
        score_components = []

        if features.pressure_zscore is not None:
            # Negative pressure Z-score indicates pressure drop (leak signature)
            score_components.append(abs(features.pressure_zscore) * 0.3)

        if features.pressure_trend is not None:
            # Negative trend indicates decreasing pressure
            if features.pressure_trend < 0:
                score_components.append(abs(features.pressure_trend) * 0.15)

        if features.pressure_deviation_from_neighbors is not None:
            # Large deviation from neighbors is suspicious
            score_components.append(
                abs(features.pressure_deviation_from_neighbors) * 0.2
            )

        if features.pressure_volatility is not None:
            # High volatility can indicate issues
            score_components.append(features.pressure_volatility * 0.05)
        
        # === NEW: Enhanced scoring with advanced features ===
        if features.pressure_rate_of_change is not None:
            # Rapid pressure drops are highly suspicious
            if features.pressure_rate_of_change < 0:
                score_components.append(abs(features.pressure_rate_of_change) * 0.1)
        
        if features.cusum_score is not None:
            # CUSUM detects small persistent shifts
            score_components.append(features.cusum_score * 0.1)
        
        if features.cross_correlation_score is not None:
            # High correlation with lagged neighbors suggests pressure wave
            if features.cross_correlation_score > 0.7:
                score_components.append(features.cross_correlation_score * 0.1)

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
