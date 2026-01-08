"""
Coordinator Agent - Central intelligence that aggregates alerts and coordinates response.

The CoordinatorAgent is the "brain" of the multi-agent system:
1. Receives alerts from all SensorAgents
2. Aggregates anomalies into clusters
3. Dispatches localization requests to LocalizerAgent
4. Orchestrates investigation mode switching
"""

import logging
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time

from .base import Agent, MessageBus, MessageType, Message
from ..config import SamplingMode

logger = logging.getLogger(__name__)


@dataclass
class AnomalyRecord:
    """Record of an anomaly alert."""
    node_id: str
    timestamp: float
    zscore: float
    confidence: float
    sensor_agent_id: str


@dataclass  
class Investigation:
    """Active leak investigation."""
    investigation_id: str
    triggered_at: float
    sensor_ids: Set[str] = field(default_factory=set)
    anomalies: List[AnomalyRecord] = field(default_factory=list)
    status: str = "active"  # active, localized, resolved
    localization_result: Optional[Dict] = None


class CoordinatorAgent(Agent):
    """
    Coordinator Agent - Central intelligence for leak detection.
    
    This agent:
    - Aggregates anomaly alerts from sensor agents
    - Clusters related anomalies (spatial/temporal correlation)
    - Decides when to trigger localization
    - Commands sensor mode changes during investigation
    - Maintains system-wide situational awareness
    
    Design Pattern: Blackboard Architecture
    - SensorAgents post to the "blackboard" (message bus)
    - CoordinatorAgent reads, aggregates, and responds
    """
    
    # Configuration
    ANOMALY_AGGREGATION_WINDOW = 600.0  # seconds (simulation uses 300s timesteps)
    MIN_ALERTS_FOR_INVESTIGATION = 2   # Need 2+ sensors alerting
    CONFIDENCE_THRESHOLD = 0.3         # Min average confidence
    
    def __init__(self, agent_id: str, message_bus: MessageBus, sensor_ids: List[str]):
        super().__init__(agent_id, message_bus)
        
        self.sensor_ids = set(sensor_ids)
        
        # Anomaly tracking
        self._recent_anomalies: List[AnomalyRecord] = []
        self._active_investigations: Dict[str, Investigation] = {}
        self._investigation_counter = 0
        
        # Sensor state tracking
        self._sensor_states: Dict[str, Dict] = {}  # Last known state per sensor
        self._sensors_in_high_res: Set[str] = set()
        
        # System state
        self._system_mode = "NORMAL"  # NORMAL, INVESTIGATING, ALERT
        self._current_time = 0.0
        
        # Statistics
        self._total_alerts_received = 0
        self._investigations_opened = 0
        self._leaks_localized = 0
        
        # Subscribe to sensor messages
        self.subscribe(MessageType.ANOMALY_ALERT)
        self.subscribe(MessageType.READING_UPDATE)
        self.subscribe(MessageType.LOCALIZATION_RESULT)
        self.subscribe(MessageType.STATUS_REPORT)
        
        logger.info(f"CoordinatorAgent '{agent_id}' managing {len(sensor_ids)} sensors")
    
    def sense(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perceive the environment - process incoming messages.
        
        The coordinator's "environment" is primarily the message bus.
        """
        self._current_time = environment.get("sim_time", time.time())
        
        # Process all pending messages
        new_anomalies = []
        localization_results = []
        
        while True:
            message = self.receive_message()
            if message is None:
                break
            
            if message.msg_type == MessageType.ANOMALY_ALERT:
                anomaly = self._process_anomaly_alert(message)
                if anomaly:
                    new_anomalies.append(anomaly)
            
            elif message.msg_type == MessageType.READING_UPDATE:
                self._update_sensor_state(message)
            
            elif message.msg_type == MessageType.LOCALIZATION_RESULT:
                localization_results.append(message.payload)
        
        # Clean old anomalies outside the aggregation window
        self._recent_anomalies = [
            a for a in self._recent_anomalies
            if self._current_time - a.timestamp < self.ANOMALY_AGGREGATION_WINDOW
        ]
        
        return {
            "new_anomalies": new_anomalies,
            "recent_anomaly_count": len(self._recent_anomalies),
            "unique_alerting_sensors": len({a.node_id for a in self._recent_anomalies}),
            "active_investigations": len(self._active_investigations),
            "localization_results": localization_results,
            "current_time": self._current_time
        }
    
    def decide(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make coordination decisions based on observations.
        
        Key decisions:
        1. Should we open a new investigation?
        2. Should we request localization?
        3. Should we change sensor modes?
        """
        actions = {
            "open_investigation": False,
            "request_localization": None,
            "mode_changes": [],
            "system_mode": self._system_mode,
            "close_investigations": []
        }
        
        # Process localization results
        for result in observations["localization_results"]:
            inv_id = result.get("investigation_id")
            if inv_id and inv_id in self._active_investigations:
                inv = self._active_investigations[inv_id]
                inv.localization_result = result
                inv.status = "localized"
                self._leaks_localized += 1
                logger.info(f"Coordinator: Investigation {inv_id} - Leak localized!")
        
        # Decision 1: Open new investigation?
        if observations["unique_alerting_sensors"] >= self.MIN_ALERTS_FOR_INVESTIGATION:
            avg_confidence = sum(a.confidence for a in self._recent_anomalies) / len(self._recent_anomalies)
            
            if avg_confidence >= self.CONFIDENCE_THRESHOLD:
                # Check if we already have an active investigation for these sensors
                alerting_nodes = {a.node_id for a in self._recent_anomalies}
                already_investigating = any(
                    len(alerting_nodes & inv.sensor_ids) > 0
                    for inv in self._active_investigations.values()
                    if inv.status == "active"
                )
                
                if not already_investigating:
                    actions["open_investigation"] = True
                    actions["investigation_sensors"] = list(alerting_nodes)
        
        # Decision 2: Request localization for active investigations?
        for inv_id, inv in self._active_investigations.items():
            if inv.status == "active" and len(inv.anomalies) >= self.MIN_ALERTS_FOR_INVESTIGATION:
                # Have enough data to attempt localization
                actions["request_localization"] = inv_id
                break  # One at a time
        
        # Decision 3: Mode changes for sensors near anomalies
        if observations["recent_anomaly_count"] > 0:
            alerting_nodes = {a.node_id for a in self._recent_anomalies}
            for node_id in alerting_nodes:
                if node_id not in self._sensors_in_high_res:
                    actions["mode_changes"].append({
                        "node_id": node_id,
                        "mode": SamplingMode.HIGH_RES
                    })
            actions["system_mode"] = "INVESTIGATING"
        elif self._system_mode == "INVESTIGATING" and not self._active_investigations:
            # No anomalies and no investigations - return to normal
            # Return all sensors to ECO mode
            for node_id in self._sensors_in_high_res.copy():
                actions["mode_changes"].append({
                    "node_id": node_id,
                    "mode": SamplingMode.ECO
                })
            actions["system_mode"] = "NORMAL"
        
        return actions
    
    def act(self, actions: Dict[str, Any]) -> None:
        """Execute decided actions."""
        
        # Action 1: Open new investigation
        if actions["open_investigation"]:
            self._investigation_counter += 1
            inv_id = f"INV-{self._investigation_counter:04d}"
            
            alerting_nodes = set(actions.get("investigation_sensors", []))
            investigation = Investigation(
                investigation_id=inv_id,
                triggered_at=self._current_time,
                sensor_ids=alerting_nodes,
                anomalies=list(self._recent_anomalies)
            )
            
            self._active_investigations[inv_id] = investigation
            self._investigations_opened += 1
            
            logger.info(f"Coordinator: Opened investigation {inv_id} for sensors: {alerting_nodes}")
            
            # Broadcast system alert
            self.broadcast(
                MessageType.SYSTEM_ALERT,
                payload={
                    "alert_type": "investigation_opened",
                    "investigation_id": inv_id,
                    "sensors": list(alerting_nodes),
                    "anomaly_count": len(investigation.anomalies)
                },
                priority=4
            )
        
        # Action 2: Request localization
        if actions["request_localization"]:
            inv_id = actions["request_localization"]
            inv = self._active_investigations[inv_id]
            
            # Send request to localizer agent
            self.send_message(
                MessageType.LOCALIZE_REQUEST,
                "localizer",
                payload={
                    "investigation_id": inv_id,
                    "anomalies": [
                        {
                            "node_id": a.node_id,
                            "zscore": a.zscore,
                            "confidence": a.confidence,
                            "timestamp": a.timestamp
                        }
                        for a in inv.anomalies
                    ]
                },
                priority=5
            )
            logger.debug(f"Coordinator: Requested localization for {inv_id}")
        
        # Action 3: Change sensor modes
        for mode_change in actions["mode_changes"]:
            node_id = mode_change["node_id"]
            new_mode = mode_change["mode"]
            
            # Find the sensor agent ID for this node
            sensor_agent_id = f"sensor_{node_id}"
            
            self.send_message(
                MessageType.MODE_CHANGE,
                sensor_agent_id,
                payload={
                    "mode": new_mode.name,
                    "reason": "investigation"
                },
                priority=3
            )
            
            if new_mode == SamplingMode.HIGH_RES:
                self._sensors_in_high_res.add(node_id)
            else:
                self._sensors_in_high_res.discard(node_id)
        
        # Update system mode
        self._system_mode = actions["system_mode"]
    
    def on_message(self, message: Message):
        """Direct message handler (for when not using sense())."""
        # Most messages handled in sense() during the main loop
        pass
    
    def _process_anomaly_alert(self, message: Message) -> Optional[AnomalyRecord]:
        """Process an anomaly alert from a sensor."""
        payload = message.payload
        
        anomaly = AnomalyRecord(
            node_id=payload["node_id"],
            timestamp=message.timestamp or self._current_time,
            zscore=payload.get("zscore", 0.0),
            confidence=payload.get("confidence", 0.5),
            sensor_agent_id=message.sender_id
        )
        
        self._recent_anomalies.append(anomaly)
        self._total_alerts_received += 1
        
        logger.debug(f"Coordinator: Anomaly from {anomaly.node_id} (Z={anomaly.zscore:.2f})")
        
        return anomaly
    
    def _update_sensor_state(self, message: Message):
        """Update tracking of sensor states."""
        payload = message.payload
        node_id = payload.get("node_id")
        if node_id:
            self._sensor_states[node_id] = {
                "last_value": payload.get("value"),
                "last_zscore": payload.get("zscore"),
                "last_update": self._current_time
            }
    
    def get_active_investigations(self) -> List[Dict]:
        """Get summary of active investigations."""
        return [
            {
                "id": inv.investigation_id,
                "status": inv.status,
                "triggered_at": inv.triggered_at,
                "sensor_count": len(inv.sensor_ids),
                "anomaly_count": len(inv.anomalies),
                "localization": inv.localization_result
            }
            for inv in self._active_investigations.values()
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status for monitoring."""
        return {
            "agent_id": self.agent_id,
            "system_mode": self._system_mode,
            "active_investigations": len(self._active_investigations),
            "sensors_in_high_res": len(self._sensors_in_high_res),
            "recent_anomaly_count": len(self._recent_anomalies),
            "total_alerts_received": self._total_alerts_received,
            "investigations_opened": self._investigations_opened,
            "leaks_localized": self._leaks_localized
        }
    
    def step(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one sense-decide-act cycle.
        
        Overrides base to NOT pre-process messages (we handle them in sense()).
        """
        # Skip base class message processing - we do it in sense()
        observations = self.sense(environment)
        actions = self.decide(observations)
        self.act(actions)
        
        return {
            "agent_id": self.agent_id,
            "observations": observations,
            "actions": actions
        }
