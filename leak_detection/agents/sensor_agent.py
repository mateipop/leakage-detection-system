"""
Sensor Agent - Autonomous IoT sensor with local anomaly detection.

Each SensorAgent represents a physical sensor device that:
1. Monitors pressure/flow at a network node
2. Performs LOCAL anomaly detection (edge AI)
3. Autonomously decides sampling rate
4. Alerts the coordinator when anomalies are detected
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import deque

import numpy as np

from .base import Agent, MessageBus, MessageType, Message
from ..config import SamplingMode, DeviceState

logger = logging.getLogger(__name__)


@dataclass
class LocalReading:
    """A sensor reading stored locally."""
    timestamp: float
    value: float
    zscore: Optional[float] = None


class SensorAgent(Agent):
    """
    Autonomous sensor agent with edge AI capabilities.
    
    This agent runs locally on the sensor device and performs:
    - Local data collection and buffering
    - Local anomaly detection (Z-score based)
    - Autonomous sampling rate adjustment
    - Alert broadcasting to coordinator
    
    The agent follows a sense-decide-act loop:
    - Sense: Read pressure/flow from environment
    - Decide: Detect anomalies, determine sampling mode
    - Act: Send readings/alerts, adjust sampling rate
    """
    
    # Local detection thresholds
    LOCAL_ANOMALY_THRESHOLD = 1.5  # Z-score threshold for local alert (lowered for sensitivity)
    ALERT_COOLDOWN = 1  # Min steps between alerts (allow rapid alerting)
    
    def __init__(
        self,
        agent_id: str,
        node_id: str,
        message_bus: MessageBus,
        reading_type: str = "pressure",
        buffer_size: int = 30
    ):
        super().__init__(agent_id, message_bus)
        
        self.node_id = node_id
        self.reading_type = reading_type
        
        # Local data buffer for anomaly detection
        self._buffer: deque = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size
        
        # State
        self._sampling_mode = SamplingMode.ECO
        self._device_state = DeviceState.IDLE
        self._last_alert_step = -self.ALERT_COOLDOWN
        self._current_step = 0
        self._anomaly_confidence = 0.0
        
        # Statistics
        self._alerts_sent = 0
        self._readings_collected = 0
        
        # Subscribe to coordinator messages
        self.subscribe(MessageType.MODE_CHANGE)
        self.subscribe(MessageType.REQUEST_DATA)
        
        logger.info(f"SensorAgent '{agent_id}' monitoring {reading_type} at node {node_id}")
    
    @property
    def sampling_mode(self) -> SamplingMode:
        return self._sampling_mode
    
    def reset(self):
        """Reset sensor agent state."""
        super().reset()
        self._buffer.clear()
        self._sampling_mode = SamplingMode.ECO
        self._device_state = DeviceState.IDLE
        self._last_alert_step = -self.ALERT_COOLDOWN
        self._current_step = 0
        self._anomaly_confidence = 0.0
        self._readings_collected = 0

    def sense(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perceive the environment - read sensor value.
        
        Args:
            environment: Must contain 'readings' dict with node values
            
        Returns:
            Local observations including anomaly assessment
        """
        readings = environment.get("readings", {})
        sim_time = environment.get("sim_time", 0.0)
        
        # Get the true value for our node
        true_value = readings.get(self.node_id, {}).get(self.reading_type, 0.0)
        
        # Add noise (simulating real sensor)
        noise_std = 0.5 if self.reading_type == "pressure" else 0.1
        measured_value = true_value + np.random.normal(0, noise_std)
        
        # Compute local Z-score
        zscore = self._compute_local_zscore(measured_value)
        
        # Store in local buffer
        reading = LocalReading(
            timestamp=sim_time,
            value=measured_value,
            zscore=zscore
        )
        self._buffer.append(reading)
        self._readings_collected += 1
        
        # === ENHANCED: Multi-metric anomaly detection ===
        is_anomaly = False
        anomaly_reasons = []
        
        if zscore is not None:
            # Standard Z-score detection
            if zscore < -self.LOCAL_ANOMALY_THRESHOLD:
                is_anomaly = True
                anomaly_reasons.append(f"zscore={zscore:.2f}")
                self._anomaly_confidence = min(1.0, abs(zscore) / 4.0)
            elif abs(zscore) > self.LOCAL_ANOMALY_THRESHOLD:
                is_anomaly = True
                anomaly_reasons.append(f"zscore={zscore:.2f}")
        
        # Rate of change detection (catches sudden drops)
        roc = self._compute_rate_of_change()
        if roc is not None and roc < -0.5:  # Rapid pressure drop
            is_anomaly = True
            anomaly_reasons.append(f"rapid_drop={roc:.3f}")
            self._anomaly_confidence = max(self._anomaly_confidence, min(1.0, abs(roc) / 2.0))
        
        # CUSUM detection (catches slow persistent leaks)
        cusum = self._compute_local_cusum(measured_value)
        if cusum > 3.0:  # Significant cumulative deviation
            is_anomaly = True
            anomaly_reasons.append(f"cusum={cusum:.2f}")
            self._anomaly_confidence = max(self._anomaly_confidence, min(1.0, cusum / 6.0))
        
        # Decay confidence if no anomaly
        if not is_anomaly:
            self._anomaly_confidence = max(0.0, self._anomaly_confidence - 0.1)
        
        return {
            "node_id": self.node_id,
            "value": measured_value,
            "zscore": zscore,
            "is_anomaly": is_anomaly,
            "confidence": self._anomaly_confidence,
            "sim_time": sim_time,
            "rate_of_change": roc,
            "cusum_score": cusum,
            "anomaly_reasons": anomaly_reasons
        }
    
    def decide(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make decisions based on observations.
        
        Decisions include:
        - Whether to send an alert
        - Whether to adjust local sampling rate
        - What data to report
        """
        actions = {
            "send_alert": False,
            "send_reading": True,
            "adjust_mode": None
        }
        
        self._current_step += 1
        
        # Decision 1: Should we alert the coordinator?
        if observations["is_anomaly"]:
            steps_since_alert = self._current_step - self._last_alert_step
            if steps_since_alert >= self.ALERT_COOLDOWN:
                actions["send_alert"] = True
                self._last_alert_step = self._current_step
        
        # Decision 2: Should we autonomously increase sampling rate?
        # (This is local agent autonomy - doesn't wait for coordinator)
        if observations["confidence"] > 0.6 and self._sampling_mode == SamplingMode.ECO:
            actions["adjust_mode"] = SamplingMode.HIGH_RES
        elif observations["confidence"] < 0.3 and self._sampling_mode == SamplingMode.HIGH_RES:
            # Only go back to ECO if coordinator hasn't overridden
            if not self._state.get("coordinator_override", False):
                actions["adjust_mode"] = SamplingMode.ECO
        
        return actions
    
    def act(self, actions: Dict[str, Any]) -> None:
        """
        Execute decided actions.
        
        Actions include sending messages and adjusting local state.
        """
        # Get latest observation from buffer
        if not self._buffer:
            return
            
        latest = self._buffer[-1]
        
        # Action 1: Send alert to coordinator
        if actions["send_alert"]:
            self.send_message(
                MessageType.ANOMALY_ALERT,
                "coordinator",
                payload={
                    "node_id": self.node_id,
                    "reading_type": self.reading_type,
                    "value": latest.value,
                    "zscore": latest.zscore,
                    "confidence": self._anomaly_confidence,
                    "buffer_stats": self._get_buffer_stats()
                },
                timestamp=latest.timestamp,
                priority=5  # High priority for alerts
            )
            self._alerts_sent += 1
            logger.info(f"SensorAgent '{self.agent_id}': ALERT sent (Z={latest.zscore:.2f})")
        
        # Action 2: Send regular reading update
        elif actions["send_reading"]:
            self.send_message(
                MessageType.READING_UPDATE,
                "coordinator",
                payload={
                    "node_id": self.node_id,
                    "reading_type": self.reading_type,
                    "value": latest.value,
                    "zscore": latest.zscore,
                    "confidence": self._anomaly_confidence
                },
                timestamp=latest.timestamp,
                priority=1  # Low priority for regular updates
            )
        
        # Action 3: Adjust sampling mode
        if actions["adjust_mode"] is not None:
            self._sampling_mode = actions["adjust_mode"]
            logger.debug(f"SensorAgent '{self.agent_id}': Mode -> {self._sampling_mode.name}")
    
    def on_message(self, message: Message):
        """Handle messages from coordinator."""
        if message.msg_type == MessageType.MODE_CHANGE:
            # Coordinator is commanding a mode change
            new_mode = message.payload.get("mode")
            if new_mode:
                self._sampling_mode = SamplingMode[new_mode] if isinstance(new_mode, str) else new_mode
                self._state["coordinator_override"] = True
                logger.debug(f"SensorAgent '{self.agent_id}': Coordinator set mode to {self._sampling_mode.name}")
        
        elif message.msg_type == MessageType.REQUEST_DATA:
            # Coordinator requesting immediate data
            if self._buffer:
                latest = self._buffer[-1]
                self.send_message(
                    MessageType.READING_UPDATE,
                    message.sender_id,
                    payload={
                        "node_id": self.node_id,
                        "value": latest.value,
                        "zscore": latest.zscore,
                        "buffer": [{"t": r.timestamp, "v": r.value} for r in list(self._buffer)[-10:]]
                    },
                    priority=3
                )
    
    def _compute_local_zscore(self, value: float) -> Optional[float]:
        """Compute Z-score using local buffer."""
        if len(self._buffer) < 5:
            return None
        
        values = [r.value for r in self._buffer]
        mean = np.mean(values)
        std = np.std(values)
        
        if std < 1e-6:
            return 0.0
        
        return (value - mean) / std
    
    def _compute_local_cusum(self, value: float) -> float:
        """
        Compute CUSUM score for detecting small persistent shifts.
        
        CUSUM is excellent at detecting slow leaks that Z-score might miss.
        """
        if len(self._buffer) < 10:
            return 0.0
        
        values = [r.value for r in self._buffer]
        
        # Use first half as baseline
        baseline_values = values[:len(values)//2]
        target_mean = np.mean(baseline_values)
        std = np.std(values) if np.std(values) > 0 else 1.0
        k = 0.5  # Allowance parameter
        
        # Calculate CUSUM for negative shift (pressure drop)
        cusum_neg = 0.0
        for val in values:
            normalized = (val - target_mean) / std
            cusum_neg = max(0, cusum_neg - normalized - k)
        
        return float(cusum_neg)
    
    def _compute_rate_of_change(self) -> Optional[float]:
        """Compute rate of change (first derivative) of recent values."""
        if len(self._buffer) < 3:
            return None
        
        recent = list(self._buffer)[-3:]
        if recent[0].timestamp == recent[-1].timestamp:
            return 0.0
        
        dt = recent[0].timestamp - recent[-1].timestamp
        dv = recent[0].value - recent[-1].value
        
        return dv / dt if dt != 0 else 0.0
    
    def _get_buffer_stats(self) -> Dict[str, float]:
        """Get statistics from local buffer."""
        if not self._buffer:
            return {}
        
        values = [r.value for r in self._buffer]
        
        # Enhanced stats with new metrics
        stats = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "trend": float(values[-1] - values[0]) if len(values) > 1 else 0.0
        }
        
        # Add rate of change
        roc = self._compute_rate_of_change()
        if roc is not None:
            stats["rate_of_change"] = roc
        
        # Add CUSUM score
        if len(self._buffer) >= 10:
            latest_value = values[-1]
            stats["cusum_score"] = self._compute_local_cusum(latest_value)
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status for monitoring."""
        return {
            "agent_id": self.agent_id,
            "node_id": self.node_id,
            "reading_type": self.reading_type,
            "sampling_mode": self._sampling_mode.name,
            "anomaly_confidence": self._anomaly_confidence,
            "readings_collected": self._readings_collected,
            "alerts_sent": self._alerts_sent,
            "buffer_size": len(self._buffer)
        }
