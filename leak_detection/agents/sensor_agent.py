
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
    timestamp: float
    value: float
    zscore: Optional[float] = None
    
@dataclass
class AdaptiveThreshold:
    mean: float = 0.0
    std_dev: float = 0.5  # Start with a relaxed threshold to avoid startup false positives
    alpha: float = 0.05   # Faster learning rate initially (was 0.01)
    min_std: float = 0.02 
    initialized: bool = False
    
    def update(self, value: float):
        if not self.initialized:
            self.mean = value
            # maintain the initial relaxed std_dev
            self.initialized = True
            return

        diff = value - self.mean
        incr = self.alpha * diff
        self.mean += incr
        self.std_dev = np.sqrt((1 - self.alpha) * self.std_dev**2 + self.alpha * (diff * diff))
        self.std_dev = max(self.std_dev, self.min_std)

    def get_zscore(self, value: float) -> float:
        return (value - self.mean) / self.std_dev

    def recalibrate(self, current_value: float):
        self.mean = current_value
        logger.debug(f"AdaptiveThreshold recalibrated mean to {current_value:.2f}")

class SensorAgent(Agent):
    
    ALERT_COOLDOWN = 1  # Min steps between alerts (allow rapid alerting)
    
    def __init__(
        self,
        agent_id: str,
        node_id: str,
        message_bus: MessageBus,
        reading_type: str = "pressure",
        buffer_size: int = 30,
        neighbor_ids: List[str] = None  # NEW: Know thy neighbors
    ):
        super().__init__(agent_id, message_bus)
        
        self.node_id = node_id
        self.reading_type = reading_type
        self.neighbors = neighbor_ids or []
        
        self._buffer: deque = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size
        
        self._model = AdaptiveThreshold()
        self._learning_enabled = True
        
        self._sampling_mode = SamplingMode.ECO
        self._device_state = DeviceState.IDLE
        self._last_alert_step = -self.ALERT_COOLDOWN
        self._current_step = 0
        self._steps_since_reset = 0  # Warmup counter
        self._anomaly_confidence = 0.0
        
        self._peer_confirmations: Dict[str, bool] = {}
        self._pending_alert = False
        
        self._alerts_sent = 0
        self._readings_collected = 0
        self._false_positives = 0
        
        self._volatility = 0.0
        self._forecast_error = 0.0  # Epistemic uncertainty
        self._last_val = None

        self.subscribe(MessageType.MODE_CHANGE)
        self.subscribe(MessageType.REQUEST_DATA)
        self.subscribe(MessageType.PEER_CONFIRMATION)
        self.subscribe(MessageType.PEER_GOSSIP)
        
        logger.info(f"SensorAgent '{agent_id}' monitoring {reading_type} at node {node_id}")
    
    @property
    def sampling_mode(self) -> SamplingMode:
        return self._sampling_mode
    
    def reset(self):
        super().reset()
        self._buffer.clear()
        self._sampling_mode = SamplingMode.ECO
        self._device_state = DeviceState.IDLE
        self._last_alert_step = -self.ALERT_COOLDOWN
        self._current_step = 0
        self._steps_since_reset = 0
        self._anomaly_confidence = 0.0
        self._volatility = 0.0
        self.last_val = None
        self._readings_collected = 0
        self._model = AdaptiveThreshold() # Reset learned model
        self._peer_confirmations.clear()
        self._gossip_replies = []
        self._pending_alert = False

    def sense(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        reading_val = None
        if "readings" in environment and self.node_id in environment["readings"]:
            reading_val = environment["readings"][self.node_id].get(self.reading_type)
            
        current_time = environment.get("sim_time", 0.0)
        self._current_step += 1
        self._steps_since_reset += 1
        
        if reading_val is not None:
            # Warmup Phase: aggressive learning, no alerts
            if self._steps_since_reset <= 10:
                 self._model.update(reading_val)
                 self._anomaly_confidence = 0.0
                 
                 # Populate buffer for trend analysis
                 reading = LocalReading(
                    timestamp=current_time,
                    value=reading_val,
                    zscore=0.0
                )
                 self._buffer.append(reading)
                 self._readings_collected += 1
                 
                 return {
                    "has_reading": True,
                    "anomaly_detected": False
                 }

            # Normal Phase
            # Calculate Z-score BEFORE updating model to detect deviations from established baseline
            zscore = self._model.get_zscore(reading_val)

            # Only update baseline if the reading is relatively normal (Z < 2.0)
            # This prevents "learning" the leak as normal behavior
            if self._learning_enabled and not self._pending_alert and abs(zscore) < 2.0:
                self._model.update(reading_val)

            predicted = reading_val # Default to persistence
            if len(self._buffer) >= 5:
                recent = list(self._buffer)[-5:]
                y = np.array([r.value for r in recent])
                x = np.arange(len(y))
                slope, intercept = np.polyfit(x, y, 1)
                predicted = slope * 5 + intercept
            
            surprise = abs(reading_val - predicted)
            self._forecast_error = (0.7 * self._forecast_error) + (0.3 * surprise)

            if self._last_val is not None:
                instability = abs(reading_val - self._last_val)
                self._volatility = (0.8 * self._volatility) + (0.2 * instability)
            self._last_val = reading_val

            reading = LocalReading(
                timestamp=current_time,
                value=reading_val,
                zscore=zscore
            )
            self._buffer.append(reading)
            self._readings_collected += 1
            
            # Lowered detection threshold for remote leaks
            if abs(zscore) > 2.5: 
                # Confidence scales: Z=2.5->0.25, Z=3.0->0.5, Z=4.0->1.0
                self._anomaly_confidence = min(1.0, (abs(zscore) - 2.0) / 2.0)
            else:
                self._anomaly_confidence = 0.0
        
        while True:
            msg = self.receive_message()
            if msg is None:
                break
                
            if msg.msg_type == MessageType.MODE_CHANGE:
                self._handle_mode_change(msg.payload)
            elif msg.msg_type == MessageType.PEER_CONFIRMATION:
                self._handle_peer_confirmation(msg)
            elif msg.msg_type == MessageType.PEER_GOSSIP:
                self._handle_peer_gossip(msg)
            elif msg.msg_type == MessageType.SYSTEM_ALERT:
                if msg.payload.get("status") == "FALSE_ALARM":
                     if msg.payload.get("culprit") == self.agent_id:
                         self._penalize_model()
                elif msg.payload.get("action") == "RESET_BASELINE":
                    new_mean = reading_val if reading_val is not None else self._model.mean
                    logger.info(f"Agent {self.agent_id} resetting baseline to {new_mean:.2f} (confirmed leak)")
                    
                    self._model.mean = new_mean
                    self._model.std_dev = 1.0 # Reset to default width
                    self._model.count = 1
                    
                    self._forecast_error = 0.0
                    self._volatility = 0.0
                    
                    self._anomaly_confidence = 0.0
                    self._pending_alert = False
                    self._last_alert_step = -self.ALERT_COOLDOWN 
                    
                    if self._buffer:
                        last_reading = self._buffer[-1]
                        last_reading.zscore = 0.0
                        self._anomaly_confidence = 0.0

        return {
            "has_reading": reading_val is not None,
            "anomaly_detected": self._anomaly_confidence > 0.5
        }
    
    def step(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        observations = self.sense(environment)
        actions = self.decide(observations)
        self.act(actions)
        
    def act(self, actions: Dict[str, Any]) -> None:
        action_list = actions.get("actions", [])
        
        for action in action_list:
            if action["type"] == "SEND_MESSAGE":
                if action["msg_type"] == MessageType.ANOMALY_ALERT:
                    self._alerts_sent += 1
                
                self.send_message(
                    msg_type=action["msg_type"],
                    recipient=action["recipient"],
                    payload=action["payload"]
                )

    def decide(self, observations: Dict[str, Any] = None) -> Dict[str, Any]:
        actions = []
        
        info_value = (self._volatility * 2.0) + (self._forecast_error * 5.0)
        
        COST_HIGH_RES = 0.5  # Energy units
        COST_ECO = 0.1       # Energy units
        
        utility_upgrade = info_value - COST_HIGH_RES
        
        current_utility_threshold = 0.2 if self._sampling_mode == SamplingMode.ECO else -0.2
        
        if utility_upgrade > current_utility_threshold:
             if self._sampling_mode == SamplingMode.ECO:
                self._sampling_mode = SamplingMode.HIGH_RES
                logger.info(f"Agent {self.agent_id} | HIGH_RES | Util:{utility_upgrade:.2f} | Err:{self._forecast_error:.4f}")
        else:
             if self._sampling_mode == SamplingMode.HIGH_RES:
                 self._sampling_mode = SamplingMode.ECO
                 logger.info(f"Agent {self.agent_id} | ECO | Low Value")

        if hasattr(self, "_gossip_replies"):
            for reply in self._gossip_replies:
                actions.append({
                    "type": "SEND_MESSAGE",
                    "msg_type": MessageType.PEER_CONFIRMATION,
                    "recipient": reply["recipient"],
                    "payload": {"confirmed": reply["confirmed"]}
                })
            self._gossip_replies.clear()
        
        # Lowered confidence threshold to trigger actions
        if self._anomaly_confidence > 0.4:
            if self._current_step - self._last_alert_step > self.ALERT_COOLDOWN:
                
                if self.neighbors:
                    if not self._pending_alert:
                        self._pending_alert = True
                        self._peer_confirmations.clear()
                        for neighbor in self.neighbors:
                            actions.append({
                                "type": "SEND_MESSAGE",
                                "msg_type": MessageType.PEER_GOSSIP,
                                "recipient": f"sensor_{neighbor}",
                                "payload": {
                                    "event": "ANOMALY_QUERY",
                                    "zscore": self._buffer[-1].zscore
                                }
                            })
                    else:
                        actions.append(self._create_alert_action())
                        self._pending_alert = False
                else:
                    actions.append(self._create_alert_action())
        
        return {"actions": actions}

    def get_status(self) -> Dict[str, Any]:
        last_val = None
        last_z = None
        if self._buffer:
            last_read = self._buffer[-1]
            last_val = last_read.value
            last_z = last_read.zscore

        return {
            "alerts_sent": self._alerts_sent,
            "confidence": self._anomaly_confidence,
            "anomaly_confidence": self._anomaly_confidence, # Legacy support
            "readings_count": self._readings_collected,
            "mode": self._sampling_mode.name,
            "sampling_mode": self._sampling_mode.name,      # Legacy support
            "last_reading": last_val,
            "last_zscore": last_z
        }

    def _create_alert_action(self):
        self._last_alert_step = self._current_step
        reading = self._buffer[-1]
        return {
            "type": "SEND_MESSAGE",
            "msg_type": MessageType.ANOMALY_ALERT,
            "recipient": "coordinator",
            "payload": {
                "node_id": self.node_id,
                "timestamp": reading.timestamp,
                "value": reading.value,
                "zscore": reading.zscore,
                "confidence": self._anomaly_confidence,
                "learned_stats": {"mean": self._model.mean, "std": self._model.std_dev}
            }
        }

    def _handle_peer_gossip(self, msg: Message):
        sender = msg.sender_id # e.g. "sensor_n1"
        payload = msg.payload
        if payload.get("event") == "ANOMALY_QUERY":
            confirmed = False
            if self._buffer:
                last_z = abs(self._buffer[-1].zscore)
                if last_z > 1.5: # Corroborate if we see something suspicious too (lower threshold)
                    confirmed = True
            
            if not hasattr(self, "_gossip_replies"):
                self._gossip_replies = []
            
            self._gossip_replies.append({
                "recipient": sender,
                "confirmed": confirmed
            })

    def _handle_peer_confirmation(self, msg: Message):
        sender_node = msg.sender_id.replace("sensor_", "")
        self._peer_confirmations[sender_node] = msg.payload.get("confirmed", False)
        
    def _penalize_model(self):
        self._false_positives += 1
        logger.info(f"Agent {self.agent_id} receiving negative feedback. adjusting model.")
        self._model.std_dev *= 1.2
        
    def _handle_mode_change(self, payload: Dict[str, Any]):
        new_mode_str = payload.get("mode")
        if new_mode_str:
            try:
                if isinstance(new_mode_str, str):
                    new_mode = SamplingMode[new_mode_str]
                else:
                    new_mode = new_mode_str
                    
                self._sampling_mode = new_mode
                logger.debug(f"Sensor {self.agent_id} switched to {self._sampling_mode}")
            except (KeyError, ValueError):
                logger.warning(f"Invalid mode requested: {new_mode_str}")

    def recalibrate(self):
        if self._buffer:
            last_reading = self._buffer[-1]
            self._model.recalibrate(last_reading.value)
            
            self._anomaly_confidence = 0.0
            self._forecast_error = 0.0
            self._volatility = 0.0
            self._pending_alert = False
            self._last_alert_step = -self.ALERT_COOLDOWN
            
            for reading in self._buffer:
                 reading.zscore = 0.0
                 
            logger.info(f"Agent {self.agent_id} recalibrated baseline to {last_reading.value:.2f}")
        else:
            self._model = AdaptiveThreshold()
            self._anomaly_confidence = 0.0
