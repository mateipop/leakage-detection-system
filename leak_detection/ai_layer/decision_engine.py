"""
Decision Engine - Converts inference results into actionable decisions.
"""

import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from ..config import AIConfig, SamplingMode, SystemStatus, DEFAULT_CONFIG
from .inference_engine import InferenceResult, AnomalyType

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions the decision engine can recommend."""
    MAINTAIN_ECO = auto()        # Keep ECO mode
    SWITCH_HIGH_RES = auto()     # Switch specific nodes to HIGH_RES
    SWITCH_ALL_HIGH_RES = auto() # Switch all nodes to HIGH_RES
    SWITCH_ECO = auto()          # Return to ECO mode
    ALERT = auto()               # Generate alert
    CONFIRM_LEAK = auto()        # Confirm leak at specific location
    REQUEST_TOPOLOGY = auto()    # Request topology analysis


@dataclass
class Decision:
    """A decision made by the decision engine."""
    action: ActionType
    target_nodes: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    estimated_leak_location: Optional[str] = None
    estimated_leak_locations: List[str] = field(default_factory=list)  # For multiple leaks
    timestamp: float = 0.0
    leak_count: int = 0  # Number of detected leaks


class DecisionEngine:
    """
    Decision engine that converts inference results into actions.

    Implements the decision logic:
    - Normal: Maintain ECO mode
    - Anomaly detected: Alert, query topology, request high-res data
    - Leak confirmed: Identify location based on pressure drop analysis
    """

    def __init__(self, config: AIConfig = None):
        """
        Initialize the decision engine.

        Args:
            config: AI configuration
        """
        self.config = config or DEFAULT_CONFIG.ai
        self._current_status = SystemStatus.NORMAL
        self._alert_history: List[Decision] = []
        self._investigation_nodes: List[str] = []
        self._investigation_start_time: Optional[float] = None
        self._samples_since_alert = 0

        logger.info("DecisionEngine initialized")

    @property
    def status(self) -> SystemStatus:
        """Current system status."""
        return self._current_status

    def _check_for_anomalies(
        self,
        results: Dict[str, InferenceResult]
    ) -> List[InferenceResult]:
        """Get results that exceed the anomaly threshold."""
        return [
            r for r in results.values()
            if r.confidence >= self.config.anomaly_threshold
        ]

    def _select_investigation_nodes(
        self,
        anomalies: List[InferenceResult],
        topology_fn: Callable[[str], List[str]] = None
    ) -> List[str]:
        """
        Select nodes to investigate based on anomalies and topology.

        Args:
            anomalies: Detected anomalies
            topology_fn: Function to get neighboring nodes

        Returns:
            List of node IDs to investigate
        """
        nodes = set()

        # Include ALL anomalous nodes (not just top 3)
        for anomaly in anomalies:
            nodes.add(anomaly.node_id)

            # Add neighbors if topology available
            if topology_fn:
                neighbors = topology_fn(anomaly.node_id)
                nodes.update(neighbors)  # Include all neighbors

        return list(nodes)

    def _pinpoint_leak_location(
        self,
        results: Dict[str, InferenceResult],
        candidate_nodes: List[str]
    ) -> Optional[str]:
        """
        Pinpoint the most likely leak location from candidates.

        Uses pressure drop magnitude as primary indicator.
        Also considers ALL results if candidate_nodes doesn't yield a result.

        Args:
            results: Inference results
            candidate_nodes: Candidate leak locations

        Returns:
            Most likely leak node ID
        """
        # First try candidates
        scores = {}
        for node_id in candidate_nodes:
            if node_id in results:
                result = results[node_id]
                # Weight pressure contribution heavily
                scores[node_id] = (
                    result.confidence * 0.5 +
                    result.pressure_contribution * 0.5
                )

        # If no candidates scored, look at ALL results
        if not scores:
            for node_id, result in results.items():
                if result.confidence >= self.config.anomaly_threshold:
                    scores[node_id] = (
                        result.confidence * 0.5 +
                        result.pressure_contribution * 0.5
                    )

        if not scores:
            return None

        return max(scores, key=scores.get)

    def make_decision(
        self,
        results: Dict[str, InferenceResult],
        timestamp: float,
        topology_fn: Callable[[str], List[str]] = None
    ) -> Decision:
        """
        Make a decision based on inference results.

        Args:
            results: Inference results for all monitored nodes
            timestamp: Current timestamp
            topology_fn: Function(node_id) -> List[neighbor_ids]

        Returns:
            Decision with recommended action
        """
        anomalies = self._check_for_anomalies(results)

        # State: NORMAL - No anomalies
        if self._current_status == SystemStatus.NORMAL:
            if not anomalies:
                return Decision(
                    action=ActionType.MAINTAIN_ECO,
                    confidence=1.0 - max((r.confidence for r in results.values()), default=0),
                    reason="No anomalies detected. System operating normally.",
                    timestamp=timestamp
                )

            # Anomaly detected - trigger alert
            self._current_status = SystemStatus.ALERT
            self._investigation_nodes = self._select_investigation_nodes(
                anomalies, topology_fn
            )
            self._investigation_start_time = timestamp
            self._samples_since_alert = 0

            top_anomaly = anomalies[0]

            decision = Decision(
                action=ActionType.SWITCH_HIGH_RES,
                target_nodes=self._investigation_nodes,
                confidence=top_anomaly.confidence,
                reason=f"Anomaly detected at {top_anomaly.node_id}: {top_anomaly.explanation}",
                timestamp=timestamp
            )
            self._alert_history.append(decision)

            logger.warning(f"ALERT: {decision.reason}")
            return decision

        # State: ALERT - Gathering more data
        elif self._current_status == SystemStatus.ALERT:
            self._samples_since_alert += 1

            if not anomalies:
                # Anomaly resolved?
                if self._samples_since_alert >= self.config.min_samples_for_detection:
                    self._current_status = SystemStatus.NORMAL
                    self._investigation_nodes.clear()

                    return Decision(
                        action=ActionType.SWITCH_ECO,
                        confidence=0.9,
                        reason="Anomaly no longer detected. Returning to ECO mode.",
                        timestamp=timestamp
                    )

                return Decision(
                    action=ActionType.MAINTAIN_ECO,
                    confidence=0.5,
                    reason="Monitoring - anomaly may have resolved.",
                    timestamp=timestamp
                )

            # Anomaly persists - move to investigation
            if self._samples_since_alert >= self.config.min_samples_for_detection:
                self._current_status = SystemStatus.INVESTIGATING
                self._samples_since_alert = 0

            return Decision(
                action=ActionType.SWITCH_HIGH_RES,
                target_nodes=self._investigation_nodes,
                confidence=anomalies[0].confidence,
                reason=f"Gathering high-resolution data. Confidence: {anomalies[0].confidence:.1%}",
                timestamp=timestamp
            )

        # State: INVESTIGATING - Pinpointing location
        elif self._current_status == SystemStatus.INVESTIGATING:
            self._samples_since_alert += 1

            if not anomalies:
                # False alarm?
                if self._samples_since_alert >= self.config.min_samples_for_detection * 2:
                    self._current_status = SystemStatus.NORMAL
                    self._investigation_nodes.clear()

                    return Decision(
                        action=ActionType.SWITCH_ECO,
                        confidence=0.8,
                        reason="Investigation complete - false alarm. Returning to ECO mode.",
                        timestamp=timestamp
                    )

                return Decision(
                    action=ActionType.MAINTAIN_ECO,
                    confidence=0.5,
                    reason="Continuing investigation...",
                    timestamp=timestamp
                )

            # Have enough samples to pinpoint?
            if self._samples_since_alert >= self.config.min_samples_for_detection:
                leak_location = self._pinpoint_leak_location(
                    results, self._investigation_nodes
                )

                if leak_location:
                    self._current_status = SystemStatus.LEAK_CONFIRMED
                    top_anomaly = anomalies[0]

                    return Decision(
                        action=ActionType.CONFIRM_LEAK,
                        target_nodes=[leak_location],
                        confidence=top_anomaly.confidence,
                        reason=f"Leak confirmed at {leak_location} with {top_anomaly.confidence:.1%} confidence",
                        estimated_leak_location=leak_location,
                        timestamp=timestamp
                    )

            return Decision(
                action=ActionType.SWITCH_HIGH_RES,
                target_nodes=self._investigation_nodes,
                confidence=anomalies[0].confidence,
                reason=f"Pinpointing leak location... ({self._samples_since_alert} samples)",
                timestamp=timestamp
            )

        # State: LEAK_CONFIRMED - Leak has been identified
        else:  # SystemStatus.LEAK_CONFIRMED
            self._samples_since_alert += 1

            # Only consider leak fixed after sustained normal readings
            # This prevents false "fixed" signals from Z-score normalization
            if not anomalies:
                # Need many consecutive normal readings to declare fixed
                # (The sliding window Z-score will normalize over time)
                if self._samples_since_alert >= self.config.min_samples_for_detection * 5:
                    self._current_status = SystemStatus.NORMAL
                    self._investigation_nodes.clear()
                    self._samples_since_alert = 0

                    return Decision(
                        action=ActionType.SWITCH_ECO,
                        confidence=0.9,
                        reason="Leak appears to be fixed. Returning to normal operation.",
                        timestamp=timestamp
                    )

                # Still in confirmed state, waiting for more evidence of fix
                return Decision(
                    action=ActionType.MAINTAIN_ECO,
                    confidence=0.6,
                    reason=f"Leak confirmed - monitoring for resolution ({self._samples_since_alert} samples).",
                    estimated_leak_location=self._investigation_nodes[0] if self._investigation_nodes else None,
                    timestamp=timestamp
                )

            # Anomalies still present - continue monitoring the leak
            self._samples_since_alert = 0  # Reset counter since anomaly still present
            return Decision(
                action=ActionType.SWITCH_HIGH_RES,
                target_nodes=self._investigation_nodes,
                confidence=anomalies[0].confidence,
                reason="Monitoring confirmed leak location.",
                estimated_leak_location=self._investigation_nodes[0] if self._investigation_nodes else None,
                timestamp=timestamp
            )

    def reset(self):
        """Reset the decision engine state."""
        self._current_status = SystemStatus.NORMAL
        self._alert_history.clear()
        self._investigation_nodes.clear()
        self._investigation_start_time = None
        self._samples_since_alert = 0

    @property
    def alert_history(self) -> List[Decision]:
        """Get history of alerts."""
        return list(self._alert_history)
