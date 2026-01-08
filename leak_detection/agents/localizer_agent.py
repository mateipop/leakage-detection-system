"""
Localizer Agent - Specializes in triangulating leak locations.

The LocalizerAgent receives anomaly clusters and uses network topology
to pinpoint the most likely leak location(s).
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import numpy as np

from .base import Agent, MessageBus, MessageType, Message

logger = logging.getLogger(__name__)


@dataclass
class LocalizationResult:
    """Result of leak localization."""
    investigation_id: str
    probable_location: str  # Most likely node
    confidence: float
    candidate_nodes: List[Dict[str, Any]]  # Ranked candidates
    cluster_info: Dict[str, Any]


class LocalizerAgent(Agent):
    """
    Localizer Agent - Expert system for leak triangulation.
    
    This agent specializes in:
    - Receiving anomaly clusters from Coordinator
    - Using sensor distances for triangulation
    - Ranking probable leak locations
    - Returning localization results
    
    Design: Domain Expert Agent
    - Holds specialized knowledge (network topology, sensor distances)
    - Activated on-demand by Coordinator
    - Provides expert opinion back to Coordinator
    """
    
    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        sensor_nodes: List[str],
        network_distances: Optional[Dict[str, Dict[str, float]]] = None
    ):
        super().__init__(agent_id, message_bus)
        
        self.sensor_nodes = sensor_nodes
        
        # Network topology: distances between sensor nodes
        # distances[node_a][node_b] = shortest path distance in meters
        self._distances = network_distances or {}
        
        # Cached computations
        self._cluster_separation_depth = self._compute_separation_depth()
        
        # Statistics
        self._localizations_performed = 0
        self._pending_requests: List[Dict] = []
        
        # Subscribe to localization requests
        self.subscribe(MessageType.LOCALIZE_REQUEST)
        self.subscribe(MessageType.ANOMALY_CLUSTER)
        
        logger.info(f"LocalizerAgent '{agent_id}' initialized with {len(sensor_nodes)} sensor nodes")
    
    def sense(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perceive - check for localization requests.
        
        For the localizer, we mainly listen for requests from coordinator.
        """
        new_requests = []
        
        # Process messages
        while True:
            message = self.receive_message()
            if message is None:
                break
            
            if message.msg_type == MessageType.LOCALIZE_REQUEST:
                new_requests.append(message.payload)
            elif message.msg_type == MessageType.ANOMALY_CLUSTER:
                # Direct cluster data (alternative input)
                new_requests.append(message.payload)
        
        self._pending_requests.extend(new_requests)
        
        return {
            "pending_requests": len(self._pending_requests),
            "new_requests": len(new_requests)
        }
    
    def decide(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide how to process localization requests.
        """
        actions = {
            "requests_to_process": [],
            "skip": False
        }
        
        if self._pending_requests:
            # Process oldest request first (FIFO)
            actions["requests_to_process"] = [self._pending_requests.pop(0)]
        else:
            actions["skip"] = True
        
        return actions
    
    def act(self, actions: Dict[str, Any]) -> None:
        """
        Execute localization and send results.
        """
        if actions["skip"]:
            return
        
        for request in actions["requests_to_process"]:
            result = self._perform_localization(request)
            
            if result:
                # Send result back to coordinator
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
        """Handle direct messages."""
        # Mainly handled in sense()
        pass
    
    def _perform_localization(self, request: Dict) -> Optional[LocalizationResult]:
        """
        Perform leak localization using triangulation.
        
        Algorithm:
        1. Extract anomaly nodes and their severities
        2. Cluster anomalies if multiple distinct locations
        3. For each cluster, find network region with highest anomaly density
        4. Rank candidate nodes by weighted proximity to anomalous sensors
        """
        anomalies = request.get("anomalies", [])
        investigation_id = request.get("investigation_id", "unknown")
        
        if not anomalies:
            logger.warning("LocalizerAgent: No anomalies in localization request")
            return None
        
        # Extract nodes and their anomaly weights
        anomaly_weights = {}
        for anom in anomalies:
            node_id = anom["node_id"]
            # Weight by zscore magnitude (more negative = more weight)
            weight = abs(anom.get("zscore", 1.0)) * anom.get("confidence", 1.0)
            anomaly_weights[node_id] = max(anomaly_weights.get(node_id, 0), weight)
        
        # If we have distance data, perform triangulation
        if self._distances:
            candidates = self._triangulate_with_distances(anomaly_weights)
        else:
            # Fallback: just rank by anomaly weight
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
        """
        Triangulate leak position using network distances.
        
        Uses inverse-distance weighting: nodes closer to more anomalous
        sensors get higher scores.
        """
        candidates = []
        anomaly_nodes = list(anomaly_weights.keys())
        
        # Score each sensor node
        for node in self.sensor_nodes:
            if node not in self._distances:
                continue
            
            # Sum of weighted inverse distances to anomalous nodes
            score = 0.0
            weights_sum = 0.0
            
            for anom_node, weight in anomaly_weights.items():
                if anom_node in self._distances.get(node, {}):
                    dist = self._distances[node][anom_node]
                    if dist > 0:
                        # Inverse distance weighting
                        score += weight / (1 + dist / 100)  # Normalize by 100m
                        weights_sum += weight
                elif anom_node == node:
                    # Same node - highest weight
                    score += weight * 2
                    weights_sum += weight
            
            if weights_sum > 0:
                # Average score
                avg_score = score / weights_sum
                confidence = min(1.0, avg_score / 2.0)
                
                candidates.append({
                    "node_id": node,
                    "score": avg_score,
                    "confidence": confidence,
                    "is_anomaly_node": node in anomaly_nodes
                })
        
        # Sort by score descending
        candidates.sort(key=lambda x: -x["score"])
        
        return candidates
    
    def _compute_separation_depth(self) -> float:
        """Compute optimal cluster separation depth from topology."""
        if not self._distances:
            return 500.0  # Default 500m
        
        all_distances = []
        for node_a, neighbors in self._distances.items():
            for node_b, dist in neighbors.items():
                if dist > 0:
                    all_distances.append(dist)
        
        if all_distances:
            # Use 25th percentile for local clustering
            return float(np.percentile(all_distances, 25))
        return 500.0
    
    def set_distances(self, distances: Dict[str, Dict[str, float]]):
        """Update network distances (for dynamic topology)."""
        self._distances = distances
        self._cluster_separation_depth = self._compute_separation_depth()
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "agent_id": self.agent_id,
            "sensor_nodes": len(self.sensor_nodes),
            "has_distance_data": bool(self._distances),
            "cluster_separation_depth": self._cluster_separation_depth,
            "localizations_performed": self._localizations_performed,
            "pending_requests": len(self._pending_requests)
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
