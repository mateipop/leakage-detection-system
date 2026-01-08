"""
Inference Engine - Anomaly detection and confidence calculation.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..config import AIConfig, DEFAULT_CONFIG
from ..data_layer.feature_extractor import FeatureVector

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Type of detected anomaly."""
    NONE = "none"
    PRESSURE_DROP = "pressure_drop"
    FLOW_ANOMALY = "flow_anomaly"
    COMBINED = "combined"


@dataclass
class InferenceResult:
    """Result of anomaly inference for a single node."""
    node_id: str
    timestamp: float
    confidence: float  # 0.0 to 1.0
    anomaly_type: AnomalyType
    pressure_contribution: float
    flow_contribution: float
    spatial_contribution: float
    explanation: str

    @property
    def is_anomaly(self) -> bool:
        """Check if this result indicates an anomaly."""
        return self.anomaly_type != AnomalyType.NONE


class InferenceEngine:
    """
    AI inference engine for anomaly detection.

    Uses a rule-based approach with Z-scores and spatial analysis
    to detect potential leaks without prior knowledge of leak locations.
    
    Supports leak signature masking to ignore confirmed leaks when
    detecting new ones.
    """

    def __init__(self, config: AIConfig = None):
        """
        Initialize the inference engine.

        Args:
            config: AI configuration
        """
        self.config = config or DEFAULT_CONFIG.ai
        self._threshold = self.config.anomaly_threshold

        # Exponential moving average of confidence per node
        self._confidence_ema: Dict[str, float] = {}
        
        # Leak signature masking
        self._pressure_signature: Dict[str, float] = {}  # Node -> expected pressure drop from confirmed leaks
        self._exclusion_zones: set = set()  # Nodes to exclude from detection

        logger.info(f"InferenceEngine initialized with threshold={self._threshold}")

    def set_leak_mask(
        self,
        pressure_signature: Dict[str, float],
        exclusion_zones: set = None
    ):
        """
        Set the leak mask to account for confirmed leaks.
        
        Args:
            pressure_signature: Expected pressure drop at each sensor from confirmed leaks
            exclusion_zones: Set of node IDs to exclude from detection entirely
        """
        self._pressure_signature = pressure_signature or {}
        self._exclusion_zones = exclusion_zones or set()
        
        if pressure_signature:
            logger.info(f"Leak mask set: {len(pressure_signature)} nodes with signature, "
                       f"{len(self._exclusion_zones)} nodes excluded")

    def clear_leak_mask(self):
        """Clear all leak masking."""
        self._pressure_signature.clear()
        self._exclusion_zones.clear()
        logger.info("Leak mask cleared")

    def add_exclusion_zone(self, node_id: str):
        """Add a single node to exclusion zones."""
        self._exclusion_zones.add(node_id)
        logger.debug(f"Added {node_id} to inference exclusion zones")

    def _sigmoid(self, x: float, scale: float = 1.0) -> float:
        """Sigmoid function to map scores to [0, 1]."""
        return 1.0 / (1.0 + np.exp(-scale * x))

    def _calculate_confidence(self, features: FeatureVector) -> tuple[float, Dict[str, float]]:
        """
        Calculate anomaly confidence from features.

        Returns:
            Tuple of (confidence, contribution_breakdown)
        """
        contributions = {
            'pressure': 0.0,
            'flow': 0.0,
            'spatial': 0.0,
            'temporal': 0.0,      # NEW: rate-of-change based
            'correlation': 0.0,   # NEW: cross-sensor correlation
            'cusum': 0.0          # NEW: cumulative sum detection
        }

        # Pressure-based detection
        # A significant negative Z-score indicates pressure drop
        if features.pressure_zscore is not None:
            # More negative Z-score = more suspicious
            # Lower the threshold - Z-score < -1 starts to be suspicious
            pressure_score = max(0, -features.pressure_zscore)
            # Sigmoid centered at 0.5 instead of 1.5 for more sensitivity
            contributions['pressure'] = self._sigmoid(pressure_score - 0.5, scale=2.0)

        # Flow-based detection
        # Increased flow can indicate leak
        if features.flow_zscore is not None:
            flow_score = max(0, features.flow_zscore)  # Positive = increased flow
            contributions['flow'] = self._sigmoid(flow_score - 1.0, scale=2.0) * 0.6

        # Spatial deviation
        # If this node deviates significantly from neighbors
        if features.pressure_deviation_from_neighbors is not None:
            # Negative deviation = this node has lower pressure than neighbors
            spatial_score = max(0, -features.pressure_deviation_from_neighbors)
            contributions['spatial'] = self._sigmoid(spatial_score - 0.5, scale=2.0)

        # Trend-based detection - rapidly changing pressure is a strong signal
        if features.pressure_trend is not None:
            trend_score = max(0, -features.pressure_trend)  # Negative trend = dropping pressure
            # Strong negative trend is highly suspicious
            if trend_score > 1.0:
                contributions['pressure'] = max(
                    contributions['pressure'],
                    self._sigmoid(trend_score - 0.5, scale=1.5)
                )

        # Raw anomaly score based detection - use the combined score from features
        if features.raw_anomaly_score > 1.0:
            raw_contribution = self._sigmoid(features.raw_anomaly_score - 1.0, scale=1.0)
            contributions['pressure'] = max(contributions['pressure'], raw_contribution)
        
        # === NEW: Rate-of-change detection (catches sudden pressure drops) ===
        if features.pressure_rate_of_change is not None:
            # Rapid negative rate of change = sudden pressure drop
            roc_score = max(0, -features.pressure_rate_of_change)
            if roc_score > 0.1:  # Significant rate of change
                contributions['temporal'] = self._sigmoid(roc_score * 10 - 1.0, scale=1.5)
        
        # === NEW: Acceleration detection (pressure dropping faster and faster) ===
        if features.pressure_acceleration is not None:
            # Negative acceleration = pressure is dropping at increasing rate
            accel_score = max(0, -features.pressure_acceleration)
            if accel_score > 0.01:
                contributions['temporal'] = max(
                    contributions['temporal'],
                    self._sigmoid(accel_score * 50 - 0.5, scale=1.0) * 0.8
                )
        
        # === NEW: Cross-correlation detection (coordinated anomalies) ===
        if features.cross_correlation_score is not None:
            # High correlation with neighbors + anomaly = leak propagating
            if features.cross_correlation_score > 0.6 and contributions['pressure'] > 0.3:
                contributions['correlation'] = features.cross_correlation_score * 0.5
        
        # === NEW: Gradient-based detection (pressure wave front) ===
        if features.pressure_gradient_magnitude is not None:
            # Large gradient = pressure difference with neighbors = likely leak location
            grad_score = features.pressure_gradient_magnitude
            if grad_score > 1.0:
                contributions['spatial'] = max(
                    contributions['spatial'],
                    self._sigmoid(grad_score - 1.0, scale=1.5) * 0.8
                )
        
        # === NEW: CUSUM detection (small persistent shifts) ===
        if features.cusum_score is not None:
            # CUSUM catches slow leaks that Z-score misses
            if features.cusum_score > 2.0:
                contributions['cusum'] = self._sigmoid(features.cusum_score - 2.0, scale=0.8) * 0.7
        
        # === NEW: Night flow anomaly (leaks more visible at night) ===
        if features.night_flow_ratio is not None:
            # Flow at night should be very low; high ratio = suspicious
            if features.night_flow_ratio > 1.5:
                contributions['flow'] = max(
                    contributions['flow'],
                    self._sigmoid(features.night_flow_ratio - 1.5, scale=2.0) * 0.7
                )

        # Combine contributions - pressure remains primary, others boost detection
        # Use max-based combination so strong signals aren't diluted
        base_confidence = contributions['pressure'] * 0.7 + contributions['flow'] * 0.15 + contributions['spatial'] * 0.15
        
        # Advanced features provide additive boost (not replacement)
        advanced_boost = (
            contributions['temporal'] * 0.3 +
            contributions['correlation'] * 0.2 +
            contributions['cusum'] * 0.2
        )
        
        # Final confidence: base + boost, capped at 1.0
        confidence = min(1.0, base_confidence + advanced_boost)

        # Apply volatility boost - high volatility indicates something changing
        if features.pressure_volatility is not None and features.pressure_volatility > 5.0:
            confidence = min(1.0, confidence * 1.2)
        
        # === NEW: Gradient consistency boost ===
        # If gradient is consistent (all neighbors show similar deviation), 
        # this is strong evidence of a real pressure wave
        if features.gradient_direction_score is not None and features.gradient_direction_score > 2.0:
            confidence = min(1.0, confidence * 1.2)

        return confidence, contributions

    def _apply_temporal_smoothing(
        self,
        node_id: str,
        raw_confidence: float
    ) -> float:
        """
        Apply exponential moving average to smooth confidence over time.

        This prevents spurious single-sample detections while allowing
        rapid response to significant anomalies.
        """
        if node_id not in self._confidence_ema:
            self._confidence_ema[node_id] = raw_confidence
        else:
            # Use asymmetric smoothing: fast rise, slow fall
            # This allows quick response to anomalies while avoiding false positives
            current = self._confidence_ema[node_id]
            if raw_confidence > current:
                # Rising: use faster alpha (50%) for quick detection
                alpha = 0.5
            else:
                # Falling: use slower alpha based on config
                alpha = 1.0 - self.config.confidence_decay

            self._confidence_ema[node_id] = (
                alpha * raw_confidence +
                (1.0 - alpha) * current
            )

        return self._confidence_ema[node_id]

    def _determine_anomaly_type(
        self,
        confidence: float,
        contributions: Dict[str, float]
    ) -> AnomalyType:
        """Determine the type of anomaly based on contributions."""
        if confidence < self._threshold:
            return AnomalyType.NONE

        # Determine primary contributor
        # Filter to only the main categories for type determination
        main_contribs = {
            'pressure': contributions.get('pressure', 0) + contributions.get('temporal', 0) + contributions.get('cusum', 0),
            'flow': contributions.get('flow', 0),
            'spatial': contributions.get('spatial', 0) + contributions.get('correlation', 0)
        }
        primary = max(main_contribs, key=main_contribs.get)

        if main_contribs['pressure'] > 0.3 and main_contribs['flow'] > 0.1:
            return AnomalyType.COMBINED
        elif primary == 'pressure' or primary == 'spatial':
            return AnomalyType.PRESSURE_DROP
        else:
            return AnomalyType.FLOW_ANOMALY

    def _generate_explanation(
        self,
        features: FeatureVector,
        confidence: float,
        anomaly_type: AnomalyType,
        contributions: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation of inference result."""
        if anomaly_type == AnomalyType.NONE:
            return "No anomaly detected - values within normal range"

        parts = []

        if contributions.get('pressure', 0) > 0.2:
            z = features.pressure_zscore or 0
            parts.append(f"pressure drop (Z={z:.2f})")

        if contributions.get('flow', 0) > 0.1:
            z = features.flow_zscore or 0
            parts.append(f"flow increase (Z={z:.2f})")

        if contributions.get('spatial', 0) > 0.2:
            dev = features.pressure_deviation_from_neighbors or 0
            parts.append(f"spatial deviation ({dev:.2f} from neighbors)")
        
        # === NEW: Include advanced feature explanations ===
        if contributions.get('temporal', 0) > 0.1:
            roc = features.pressure_rate_of_change or 0
            parts.append(f"rapid change (rate={roc:.3f})")
        
        if contributions.get('cusum', 0) > 0.1:
            cusum = features.cusum_score or 0
            parts.append(f"persistent shift (CUSUM={cusum:.2f})")
        
        if contributions.get('correlation', 0) > 0.1:
            corr = features.cross_correlation_score or 0
            parts.append(f"correlated anomaly ({corr:.0%} sync)")

        explanation = f"Anomaly detected ({confidence:.1%} confidence): " + ", ".join(parts)
        return explanation

    def infer(self, features: FeatureVector) -> InferenceResult:
        """
        Run inference on a single feature vector.

        Args:
            features: Feature vector for one node

        Returns:
            InferenceResult with confidence and anomaly classification
        """
        node_id = features.node_id
        
        # Check if this node is in an exclusion zone (near confirmed leak)
        if node_id in self._exclusion_zones:
            # Return a suppressed result - don't flag as anomaly
            return InferenceResult(
                node_id=node_id,
                timestamp=features.timestamp,
                confidence=0.0,
                anomaly_type=AnomalyType.NONE,
                pressure_contribution=0.0,
                flow_contribution=0.0,
                spatial_contribution=0.0,
                explanation="Node in exclusion zone (near confirmed leak)"
            )
        
        # Calculate raw confidence
        raw_confidence, contributions = self._calculate_confidence(features)
        
        # Apply leak signature masking - reduce confidence if pressure drop
        # is explained by known leaks
        if node_id in self._pressure_signature:
            expected_drop = self._pressure_signature[node_id]
            if expected_drop > 0:
                # If the expected signature accounts for most of the anomaly,
                # reduce confidence proportionally
                # Scale factor: how much of the anomaly is explained
                mask_factor = min(1.0, expected_drop / 5.0)  # 5 psi = full masking
                raw_confidence = raw_confidence * (1.0 - mask_factor * 0.8)
                
                if mask_factor > 0.3:
                    contributions['pressure'] *= (1.0 - mask_factor)

        # Apply temporal smoothing
        smoothed_confidence = self._apply_temporal_smoothing(
            features.node_id,
            raw_confidence
        )

        # Determine anomaly type
        anomaly_type = self._determine_anomaly_type(smoothed_confidence, contributions)

        # Generate explanation
        explanation = self._generate_explanation(
            features,
            smoothed_confidence,
            anomaly_type,
            contributions
        )
        
        # Add mask info to explanation
        if node_id in self._pressure_signature:
            explanation += f" [Masked by confirmed leak signature: {self._pressure_signature[node_id]:.1f} psi]"

        return InferenceResult(
            node_id=features.node_id,
            timestamp=features.timestamp,
            confidence=smoothed_confidence,
            anomaly_type=anomaly_type,
            pressure_contribution=contributions['pressure'],
            flow_contribution=contributions['flow'],
            spatial_contribution=contributions['spatial'],
            explanation=explanation
        )

    def infer_batch(
        self,
        features: Dict[str, FeatureVector]
    ) -> Dict[str, InferenceResult]:
        """
        Run inference on multiple feature vectors.

        Args:
            features: Dict mapping node_id to FeatureVector

        Returns:
            Dict mapping node_id to InferenceResult
        """
        return {
            node_id: self.infer(f)
            for node_id, f in features.items()
        }

    def get_anomalies(
        self,
        results: Dict[str, InferenceResult],
        min_confidence: float = None
    ) -> List[InferenceResult]:
        """
        Get all results classified as anomalies.

        Args:
            results: Inference results
            min_confidence: Minimum confidence threshold (default: config threshold)

        Returns:
            List of anomalous results sorted by confidence
        """
        threshold = min_confidence or self._threshold

        anomalies = [
            r for r in results.values()
            if r.confidence >= threshold
        ]

        return sorted(anomalies, key=lambda r: r.confidence, reverse=True)

    def get_top_candidate(
        self,
        results: Dict[str, InferenceResult]
    ) -> Optional[InferenceResult]:
        """
        Get the most likely leak candidate.

        Args:
            results: Inference results

        Returns:
            InferenceResult with highest confidence, or None
        """
        if not results:
            return None

        return max(results.values(), key=lambda r: r.confidence)

    def reset_history(self, node_id: str = None):
        """
        Reset confidence history and leak masks.

        Args:
            node_id: Specific node to reset (None = all)
        """
        if node_id is None:
            self._confidence_ema.clear()
            self.clear_leak_mask()
        elif node_id in self._confidence_ema:
            del self._confidence_ema[node_id]
