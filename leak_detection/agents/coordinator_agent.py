
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
    node_id: str
    timestamp: float
    zscore: float
    confidence: float
    sensor_agent_id: str

@dataclass  
class Investigation:
    investigation_id: str
    triggered_at: float
    sensor_ids: Set[str] = field(default_factory=set)
    anomalies: List[AnomalyRecord] = field(default_factory=list)
    status: str = "active"
    localization_result: Optional[Dict] = None
    localized_at: Optional[float] = None

class CoordinatorAgent(Agent):
    
    ANOMALY_AGGREGATION_WINDOW = 3600.0
    MIN_ALERTS_FOR_INVESTIGATION = 3
    MIN_SENSORS_FOR_LOCALIZATION = 10
    CONFIDENCE_THRESHOLD = 0.6
    
    def __init__(self, agent_id: str, message_bus: MessageBus, sensor_ids: List[str]):
        super().__init__(agent_id, message_bus)
        
        self.sensor_ids = set(sensor_ids)
        
        self._recent_anomalies: List[AnomalyRecord] = []
        self._active_investigations: Dict[str, Investigation] = {}
        self._investigation_counter = 0
        
        self._sensor_states: Dict[str, Dict] = {}
        self._sensors_in_high_res: Set[str] = set()
        
        self._system_mode = "NORMAL"
        self._current_time = 0.0
        
        self._total_alerts_received = 0
        self._investigations_opened = 0
        self._leaks_localized = 0
        
        self.subscribe(MessageType.ANOMALY_ALERT)
        self.subscribe(MessageType.READING_UPDATE)
        self.subscribe(MessageType.LOCALIZATION_RESULT)
        self.subscribe(MessageType.STATUS_REPORT)
        
        self._pending_messages: List[Message] = []
        
        logger.info(f"CoordinatorAgent '{agent_id}' managing {len(sensor_ids)} sensors")
    
    def reset(self):
        super().reset()
        self._pending_messages.clear()
        self._recent_anomalies.clear()
        self._active_investigations.clear()
        self._investigation_counter = 0
        self._sensor_states.clear()
        self._sensors_in_high_res.clear()
        self._system_mode = "NORMAL"

    def on_message(self, message: Message):
        self._pending_messages.append(message)
    
    def sense(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        self._current_time = environment.get("sim_time", time.time())
        
        incoming = self.receive_messages()
        for msg in incoming:
            self.on_message(msg)
            
        new_anomalies = []
        localization_results = []
        
        messages_to_process = list(self._pending_messages)
        self._pending_messages.clear()
        
        for message in messages_to_process:
            if message.msg_type == MessageType.ANOMALY_ALERT:
                anomaly = self._process_anomaly_alert(message)
                if anomaly:
                    new_anomalies.append(anomaly)
            
            elif message.msg_type == MessageType.READING_UPDATE:
                self._update_sensor_state(message)
            
            elif message.msg_type == MessageType.LOCALIZATION_RESULT:
                localization_results.append(message.payload)
        
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
        actions = {
            "open_investigation": False,
            "request_localization": None,
            "mode_changes": [],
            "system_mode": self._system_mode,
            "close_investigations": []
        }
        
        for result in observations["localization_results"]:
            inv_id = result.get("investigation_id")
            if inv_id and inv_id in self._active_investigations:
                inv = self._active_investigations[inv_id]
                inv.localization_result = result
                inv.status = "localized"
                if inv.localized_at is None:
                    inv.localized_at = self._current_time
                self._leaks_localized += 1
                logger.info(f"Coordinator: Investigation {inv_id} - Leak localized!")
        
        alerting_nodes = set()
        should_investigate = False
        
        # Gather stats per sensor
        sensor_stats = {}
        for a in self._recent_anomalies:
            if a.node_id not in sensor_stats:
                sensor_stats[a.node_id] = {'count': 0, 'max_conf': 0.0}
            sensor_stats[a.node_id]['count'] += 1
            sensor_stats[a.node_id]['max_conf'] = max(sensor_stats[a.node_id]['max_conf'], a.confidence)
            
        unique_sensors = len(sensor_stats)

        if unique_sensors >= self.MIN_ALERTS_FOR_INVESTIGATION:
            avg_max_conf = sum(s['max_conf'] for s in sensor_stats.values()) / unique_sensors
            if avg_max_conf >= self.CONFIDENCE_THRESHOLD:
                alerting_nodes = set(sensor_stats.keys())
                should_investigate = True

        elif unique_sensors == 1:
            node_id = list(sensor_stats.keys())[0]
            stats = sensor_stats[node_id]
            if stats['count'] >= 5 and stats['max_conf'] >= 0.7:
                alerting_nodes = {node_id}
                should_investigate = True

        if should_investigate:
            is_redundant = False
            for inv in self._active_investigations.values():
                if inv.status in ["active", "localized"]:
                    overlap = len(alerting_nodes & inv.sensor_ids)
                    if overlap > 0 and (overlap / len(alerting_nodes) > 0.5):
                        is_redundant = True
                        inv.sensor_ids.update(alerting_nodes)
                        existing_ids = {(a.node_id, a.timestamp) for a in inv.anomalies}
                        for a in self._recent_anomalies:
                            if (a.node_id, a.timestamp) not in existing_ids:
                                inv.anomalies.append(a)
                        break
            
            if not is_redundant:
                actions["open_investigation"] = True
                actions["investigation_sensors"] = list(alerting_nodes)
        
        for inv_id, inv in self._active_investigations.items():
            unique_sensors = len(inv.sensor_ids)
            if inv.status == "active" and unique_sensors >= self.MIN_SENSORS_FOR_LOCALIZATION:
                actions["request_localization"] = inv_id
                break
        
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
            for node_id in self._sensors_in_high_res.copy():
                actions["mode_changes"].append({
                    "node_id": node_id,
                    "mode": SamplingMode.ECO
                })
            actions["system_mode"] = "NORMAL"
        
        return actions
    
    def act(self, actions: Dict[str, Any]) -> None:
        
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
        
        if actions["request_localization"]:
            inv_id = actions["request_localization"]
            inv = self._active_investigations[inv_id]
            
            all_anomalies = list(self._recent_anomalies)
            
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
                        for a in all_anomalies
                    ]
                },
                priority=5
            )
            logger.debug(f"Coordinator: Requested localization for {inv_id}")
        
        for mode_change in actions["mode_changes"]:
            node_id = mode_change["node_id"]
            new_mode = mode_change["mode"]
            
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
        
        self._system_mode = actions["system_mode"]
    
    def _process_anomaly_alert(self, message: Message) -> Optional[AnomalyRecord]:
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
        payload = message.payload
        node_id = payload.get("node_id")
        if node_id:
            self._sensor_states[node_id] = {
                "last_value": payload.get("value"),
                "last_zscore": payload.get("zscore"),
                "last_update": self._current_time
            }
    
    def get_active_investigations(self) -> List[Dict]:
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
        observations = self.sense(environment)
        actions = self.decide(observations)
        self.act(actions)
        
        return {
            "agent_id": self.agent_id,
            "observations": observations,
            "actions": actions
        }