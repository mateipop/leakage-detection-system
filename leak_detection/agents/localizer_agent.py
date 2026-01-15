
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import numpy as np

from .base import Agent, MessageBus, MessageType, Message

logger = logging.getLogger(__name__)

@dataclass
class LocalizationResult:
    investigation_id: str
    probable_location: str  # Most likely node
    confidence: float
    candidate_nodes: List[Dict[str, Any]]  # Ranked candidates
    cluster_info: Dict[str, Any]

class LocalizerAgent(Agent):
    
    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        sensor_nodes: List[str],
        network_distances: Optional[Dict[str, Dict[str, float]]] = None,
        candidate_nodes: Optional[List[str]] = None,
        candidate_distances: Optional[Dict[str, Dict[str, float]]] = None
    ):
        super().__init__(agent_id, message_bus)
        
        self.sensor_nodes = sensor_nodes
        self.candidate_nodes = candidate_nodes or sensor_nodes
        
        self._distances = network_distances or {}
        
        self._candidate_distances = candidate_distances or {}
        
        if not self._candidate_distances and self._distances:
            for s1 in self.sensor_nodes:
                self._candidate_distances[s1] = {}
                for s2 in self.sensor_nodes:
                    if s2 in self._distances.get(s1, {}):
                        self._candidate_distances[s1][s2] = self._distances[s1][s2]
        
        self._cluster_separation_depth = self._compute_separation_depth()
        
        self._localizations_performed = 0
        self._pending_requests: List[Dict] = []
        
        self.subscribe(MessageType.LOCALIZE_REQUEST)
        self.subscribe(MessageType.ANOMALY_CLUSTER)
        
        logger.info(f"LocalizerAgent '{agent_id}' initialized with {len(sensor_nodes)} sensor nodes")
    
    def reset(self):
        super().reset()
        self._pending_requests.clear()
        self._localizations_performed = 0

    def sense(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        new_requests = []
        
        while True:
            message = self.receive_message()
            if message is None:
                break
            
            if message.msg_type == MessageType.LOCALIZE_REQUEST:
                new_requests.append(message.payload)
            elif message.msg_type == MessageType.ANOMALY_CLUSTER:
                new_requests.append(message.payload)
        
        self._pending_requests.extend(new_requests)
        
        return {
            "pending_requests": len(self._pending_requests),
            "new_requests": len(new_requests)
        }
    
    def decide(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        actions = {
            "requests_to_process": [],
            "skip": False
        }
        
        if self._pending_requests:
            actions["requests_to_process"] = [self._pending_requests.pop(0)]
        else:
            actions["skip"] = True
        
        return actions
    
    def act(self, actions: Dict[str, Any]) -> None:
        if actions["skip"]:
            return
        
        for request in actions["requests_to_process"]:
            result = self._perform_localization(request)
            
            if result:
                self.send_message(
                    MessageType.LOCALIZATION_RESULT,
                    "coordinator",
                    payload={
                        "investigation_id": result.investigation_id,
                        "probable_location": result.probable_location,
                        "confidence": result.confidence,
                        "candidates": result.candidate_nodes,
                        "cluster_info": result.cluster_info
                    },
                    priority=5
                )
                
                self._localizations_performed += 1
                logger.info(
                    f"LocalizerAgent: Localized leak at {result.probable_location} "
                    f"(confidence: {result.confidence:.2%})"
                )
    
    def on_message(self, message: Message):
        pass
    
    def _perform_localization(self, request: Dict) -> Optional[LocalizationResult]:
        anomalies = request.get("anomalies", [])
        investigation_id = request.get("investigation_id", "unknown")
        
        if not anomalies:
            logger.warning("LocalizerAgent: No anomalies in localization request")
            return None
        
        anomaly_weights = {}
        for anom in anomalies:
            node_id = anom["node_id"]
            weight = abs(anom.get("zscore", 1.0)) * anom.get("confidence", 1.0)
            anomaly_weights[node_id] = max(anomaly_weights.get(node_id, 0), weight)
        
        if self._distances:
            candidates = self._triangulate_with_distances(anomaly_weights)
        else:
            candidates = [
                {"node_id": node, "score": weight, "confidence": min(1.0, weight / 4.0)}
                for node, weight in sorted(anomaly_weights.items(), key=lambda x: -x[1])
            ]
        
        if not candidates:
            return None
        
        best = candidates[0]
        
        return LocalizationResult(
            investigation_id=investigation_id,
            probable_location=best["node_id"],
            confidence=best.get("confidence", 0.5),
            candidate_nodes=candidates[:5],  # Top 5
            cluster_info={
                "anomaly_count": len(anomalies),
                "unique_sensors": len(anomaly_weights),
                "sensor_nodes": list(anomaly_weights.keys())
            }
        )
    
    def _triangulate_with_distances(self, anomaly_weights: Dict[str, float]) -> List[Dict]:
        candidates = []
        
        search_space = self.candidate_nodes
        alerting_sensors = set(anomaly_weights.keys())
        silent_sensors = [s for s in self.sensor_nodes if s not in alerting_sensors]
        
        for node in search_space:
            
            score = 0.0
            weights_sum = 0.0
            
            dists = self._candidate_distances.get(node, {})
            
            valid_distances_found = False
            
            # Positive evidence from alerting sensors
            for anom_sensor, weight in anomaly_weights.items():
                used_dist = 9999.0
                if node == anom_sensor:
                     used_dist = 0.0
                     valid_distances_found = True
                elif anom_sensor in dists:
                    used_dist = dists[anom_sensor]
                    valid_distances_found = True
                else:
                    used_dist = 9999.0
                    
                if used_dist < 5000: # Only consider reasonably close sensors
                    # Use a larger smoothing factor (200.0) to avoid over-biasing towards sensor nodes
                    # 1 hop is approx 100m. 200m smoothing means 2 hops.
                    score += weight / (200.0 + used_dist) 
                    weights_sum += weight
            
            # Negative evidence from silent sensors
            # If a candidate is very close to a silent sensor, it's unlikely to be the leak
            penalty_score = 0.0
            for silent_s in silent_sensors:
                s_dist = 9999.0
                if node == silent_s:
                    s_dist = 0.0
                elif silent_s in dists:
                    s_dist = dists[silent_s]
                
                # If within ~4 hops (400m) of a silent sensor, penalize heavily
                if s_dist < 400.0:
                    # Decay: At 0m -> 1/50. At 400m -> 1/450.
                    penalty_factor = 1.0 / (50.0 + s_dist)
                    penalty_score += penalty_factor * 20.0 # Weight of negative evidence

            final_score = score - penalty_score
            
            if weights_sum > 0 and valid_distances_found and final_score > 0:
                avg_score = final_score
                confidence = min(1.0, avg_score * 20000) 
                
                candidates.append({
                    "node_id": node,
                    "score": avg_score,
                    "confidence": confidence,
                })
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates
    
    def _compute_separation_depth(self) -> float:
        if not self._distances:
            return 500.0  # Default 500m
        
        all_distances = []
        for node_a, neighbors in self._distances.items():
            for node_b, dist in neighbors.items():
                if dist > 0:
                    all_distances.append(dist)
        
        if all_distances:
            return float(np.percentile(all_distances, 25))
        return 500.0
    
    def set_distances(self, distances: Dict[str, Dict[str, float]]):
        self._distances = distances
        self._cluster_separation_depth = self._compute_separation_depth()
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "sensor_nodes": len(self.sensor_nodes),
            "has_distance_data": bool(self._distances),
            "cluster_separation_depth": self._cluster_separation_depth,
            "localizations_performed": self._localizations_performed,
            "pending_requests": len(self._pending_requests)
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