"""
Leak Localizer - Topology-aware leak location estimation.

Uses pressure sensitivity analysis and network topology to estimate
leak locations even at nodes without sensors.

Supports detection of MULTIPLE simultaneous leaks by clustering
anomalous sensors into separate leak events based on network topology.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque

import numpy as np

from .inference_engine import InferenceResult

logger = logging.getLogger(__name__)


@dataclass
class LocalizationResult:
    """Result of leak localization analysis."""
    estimated_node: str
    confidence: float
    is_sensored: bool  # Whether estimated node has a sensor
    nearby_sensors: List[str]  # Sensors that contributed to estimate
    candidate_region: List[str]  # All candidate nodes in the region
    method: str  # Localization method used
    cluster_id: int = 0  # ID for multi-leak tracking
    contributing_anomalies: List[str] = field(default_factory=list)  # Anomalous sensors in this cluster


@dataclass
class MultiLeakResult:
    """Result containing multiple potential leak locations."""
    leak_count: int
    localizations: List[LocalizationResult]
    total_confidence: float  # Overall detection confidence
    clustering_method: str  # How clusters were identified


class LeakLocalizer:
    """
    Topology-aware leak localizer.

    Uses multiple methods to estimate leak location:
    1. Direct detection: If leak is at a sensored node
    2. Triangulation: Using pressure drops at multiple sensors
    3. Sensitivity analysis: Using pre-computed or estimated sensitivity matrix
    4. Multi-leak clustering: Detecting multiple simultaneous leaks
    """

    def __init__(
        self,
        get_neighbors_fn,
        get_all_nodes_fn,
        sensored_nodes: Set[str],
        cluster_separation_depth: int = None,  # Auto-computed if None
        stabilization_window: int = 5,  # Number of cycles for voting
        confidence_hysteresis: float = 0.15  # Min confidence improvement to switch
    ):
        """
        Initialize the localizer.

        Args:
            get_neighbors_fn: Function(node_id, depth) -> List[str] to get neighbors
            get_all_nodes_fn: Function() -> List[str] to get all network nodes
            sensored_nodes: Set of node IDs that have sensors
            cluster_separation_depth: Min hops between sensors to consider them separate clusters
                                      If None, auto-computed from median sensor distance
            stabilization_window: Number of cycles to consider for voting-based stabilization
            confidence_hysteresis: Minimum confidence improvement required to change prediction
        """
        self._get_neighbors = get_neighbors_fn
        self._get_all_nodes = get_all_nodes_fn
        self._sensored_nodes = sensored_nodes
        
        # Build reverse mapping: for each node, which sensors are nearby
        self._nearby_sensors_cache: Dict[str, List[Tuple[str, int]]] = {}
        # Build sensor-to-sensor distance matrix for clustering
        self._sensor_distances: Dict[Tuple[str, str], int] = {}
        self._build_sensor_proximity_map()
        
        # Auto-compute cluster separation if not specified
        if cluster_separation_depth is None:
            self._cluster_separation_depth = self._compute_optimal_separation()
        else:
            self._cluster_separation_depth = cluster_separation_depth

        # Temporal stabilization state
        self._stabilization_window = stabilization_window
        self._confidence_hysteresis = confidence_hysteresis
        self._prediction_history: List[Tuple[str, float]] = []  # (node, confidence) history
        self._current_stable_prediction: Optional[str] = None
        self._current_stable_confidence: float = 0.0
        self._cycles_at_current: int = 0
        
        # Exclusion zones - nodes to skip during localization (near confirmed leaks)
        self._exclusion_zones: Set[str] = set()

        logger.info(f"LeakLocalizer initialized with {len(sensored_nodes)} sensors, "
                    f"cluster_separation={self._cluster_separation_depth}, "
                    f"stabilization_window={stabilization_window}")

    def _build_sensor_proximity_map(self, max_depth: int = 6):
        """
        Build a map of which sensors are near each node.

        For each node, store list of (sensor_id, distance) tuples.
        Also builds sensor-to-sensor distance matrix for clustering.
        """
        all_nodes = self._get_all_nodes()

        for node in all_nodes:
            nearby = []
            for depth in range(1, max_depth + 1):
                neighbors = self._get_neighbors(node, depth)
                for neighbor in neighbors:
                    if neighbor in self._sensored_nodes:
                        # Check if already added at closer distance
                        if not any(s == neighbor for s, _ in nearby):
                            nearby.append((neighbor, depth))

            # Also check if node itself is sensored (distance 0)
            if node in self._sensored_nodes:
                nearby.insert(0, (node, 0))

            self._nearby_sensors_cache[node] = nearby

        # Build sensor-to-sensor distance matrix
        self._build_sensor_distance_matrix(max_depth * 2)

    def _compute_optimal_separation(self) -> int:
        """
        Compute optimal cluster separation depth based on sensor topology.
        
        Uses the median distance between sensors as a baseline.
        Sensors within this distance are likely affected by the same leak.
        """
        if not self._sensor_distances:
            return 6  # Default fallback
        
        distances = list(self._sensor_distances.values())
        distances.sort()
        
        # Use 25th percentile as separation - sensors closer than this are "nearby"
        # This means ~25% of sensor pairs will be in the same cluster potential
        idx = len(distances) // 4
        percentile_25 = distances[idx] if idx < len(distances) else 6
        
        # Clamp to reasonable range
        optimal = max(4, min(10, percentile_25))
        
        logger.info(f"Auto-computed cluster separation: {optimal} "
                    f"(25th percentile of {len(distances)} sensor pairs)")
        return optimal

    def _build_sensor_distance_matrix(self, max_depth: int = 12):
        """Build distance matrix between all pairs of sensors for clustering."""
        sensors = list(self._sensored_nodes)
        sensors_set = set(sensors)
        
        # Efficient BFS from each sensor to find all other sensors within max_depth
        for start_node in sensors:
            queue = deque([(start_node, 0)])
            visited = {start_node}
            
            while queue:
                current, dist = queue.popleft()
                
                # If we found another sensor, record distance
                if current in sensors_set and current != start_node:
                    self._sensor_distances[(start_node, current)] = dist
                    self._sensor_distances[(current, start_node)] = dist
                
                # Continue BFS if within search depth
                if dist < max_depth:
                    # Get immediate neighbors (depth=1)
                    neighbors = self._get_neighbors(current, 1)
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

    def _find_sensor_distance(self, sensor1: str, sensor2: str, max_depth: int) -> int:
        """Find the network distance (hops) between two sensors."""
        # This is now only used as a fallback or utility, as matrix is pre-filled
        if sensor1 == sensor2:
            return 0
        if (sensor1, sensor2) in self._sensor_distances:
            return self._sensor_distances[(sensor1, sensor2)]
            
        # Fallback to BFS if not in matrix (e.g. dynamic queries)
        queue = deque([(sensor1, 0)])
        visited = {sensor1}
        
        while queue:
            current, dist = queue.popleft()
            if current == sensor2:
                return dist
            
            if dist < max_depth:
                for neighbor in self._get_neighbors(current, 1):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        
        return max_depth + 1  # Not found within max_depth

    def _cluster_anomalies(
        self,
        anomalies: List[Tuple[str, InferenceResult]],
        min_confidence_for_cluster: float = 0.6
    ) -> List[List[Tuple[str, InferenceResult]]]:
        """
        Cluster anomalous sensors by network proximity.
        
        Uses a confidence-weighted agglomerative approach:
        - High-confidence anomalies seed clusters
        - Lower-confidence anomalies are merged into nearby clusters
        - Distant high-confidence anomalies indicate separate leaks
        
        Args:
            anomalies: List of (node_id, InferenceResult) tuples
            min_confidence_for_cluster: Minimum confidence to seed a new cluster
            
        Returns:
            List of clusters, each containing anomalies that are topologically close
        """
        if len(anomalies) <= 1:
            return [anomalies] if anomalies else []
        
        # Initialize: each anomaly in its own cluster
        clusters: List[Set[int]] = [{i} for i in range(len(anomalies))]
        anomaly_nodes = [a[0] for a in anomalies]
        
        # Build adjacency based on sensor distances
        def are_close(idx1: int, idx2: int) -> bool:
            node1, node2 = anomaly_nodes[idx1], anomaly_nodes[idx2]
            distance = self._sensor_distances.get(
                (node1, node2), 
                self._cluster_separation_depth + 1
            )
            return distance <= self._cluster_separation_depth
        
        # Agglomerative clustering
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(clusters):
                j = i + 1
                while j < len(clusters):
                    # Check if any members are close
                    should_merge = False
                    for idx1 in clusters[i]:
                        for idx2 in clusters[j]:
                            if are_close(idx1, idx2):
                                should_merge = True
                                break
                        if should_merge:
                            break
                    
                    if should_merge:
                        clusters[i] = clusters[i] | clusters[j]
                        clusters.pop(j)
                        changed = True
                    else:
                        j += 1
                i += 1
        
        # Convert indices back to anomalies
        result = []
        for cluster in clusters:
            cluster_anomalies = [anomalies[idx] for idx in cluster]
            result.append(cluster_anomalies)
        
        # Sort clusters by max confidence (most confident first)
        result.sort(key=lambda c: max(a[1].confidence for a in c), reverse=True)
        
        # Filter: only keep clusters with at least one high-confidence anomaly
        # This distinguishes real leak clusters from network-wide noise
        filtered = []
        for cluster in result:
            max_conf = max(a[1].confidence for a in cluster)
            if max_conf >= min_confidence_for_cluster:
                filtered.append(cluster)
        
        # If we filtered everything, keep the top cluster
        if not filtered and result:
            filtered = [result[0]]
        
        return filtered

    def _stabilize_prediction(
        self,
        raw_node: str,
        raw_confidence: float
    ) -> Tuple[str, float]:
        """
        Apply temporal stabilization to prevent jittering predictions.
        
        Uses:
        1. Voting window - tracks predictions over time
        2. Confidence hysteresis - only switch if new is significantly better
        3. Persistence - keeps current prediction unless clearly wrong
        
        Args:
            raw_node: The raw (unstabilized) predicted node
            raw_confidence: The raw confidence for this prediction
            
        Returns:
            (stabilized_node, stabilized_confidence) tuple
        """
        # Add to history
        self._prediction_history.append((raw_node, raw_confidence))
        
        # Trim history to window size
        if len(self._prediction_history) > self._stabilization_window:
            self._prediction_history = self._prediction_history[-self._stabilization_window:]
        
        # If no stable prediction yet, use the raw one
        if self._current_stable_prediction is None:
            self._current_stable_prediction = raw_node
            self._current_stable_confidence = raw_confidence
            self._cycles_at_current = 1
            return raw_node, raw_confidence
        
        # Count votes for each node in the window
        vote_counts: Dict[str, int] = defaultdict(int)
        confidence_sums: Dict[str, float] = defaultdict(float)
        
        for node, conf in self._prediction_history:
            vote_counts[node] += 1
            confidence_sums[node] += conf
        
        # Get the winning node (most votes, then highest avg confidence)
        sorted_candidates = sorted(
            vote_counts.keys(),
            key=lambda n: (vote_counts[n], confidence_sums[n] / max(vote_counts[n], 1)),
            reverse=True
        )
        
        winner = sorted_candidates[0] if sorted_candidates else raw_node
        winner_votes = vote_counts[winner]
        winner_avg_conf = confidence_sums[winner] / max(winner_votes, 1)
        
        # Check if we should switch predictions
        current_votes = vote_counts.get(self._current_stable_prediction, 0)
        current_avg_conf = confidence_sums.get(self._current_stable_prediction, 0) / max(current_votes, 1)
        
        should_switch = False
        
        # Switch if winner has significantly more votes
        if winner_votes > current_votes + 1:
            should_switch = True
        # Or if same votes but significantly higher confidence
        elif winner_votes >= current_votes and winner_avg_conf > current_avg_conf + self._confidence_hysteresis:
            should_switch = True
        # Or if current prediction has dropped out of top results
        elif current_votes == 0:
            should_switch = True
        
        if should_switch and winner != self._current_stable_prediction:
            logger.info(f"Stabilized prediction switch: {self._current_stable_prediction} -> {winner} "
                       f"(votes: {current_votes} -> {winner_votes}, conf: {current_avg_conf:.2f} -> {winner_avg_conf:.2f})")
            self._current_stable_prediction = winner
            self._current_stable_confidence = winner_avg_conf
            self._cycles_at_current = 1
        else:
            self._cycles_at_current += 1
            # Update confidence with rolling average
            self._current_stable_confidence = (
                0.7 * self._current_stable_confidence + 0.3 * current_avg_conf
            )
        
        return self._current_stable_prediction, self._current_stable_confidence

    def reset_stabilization(self):
        """Reset the stabilization state (call when leaks are cleared)."""
        self._prediction_history.clear()
        self._current_stable_prediction = None
        self._current_stable_confidence = 0.0
        self._cycles_at_current = 0
        self._exclusion_zones.clear()
        logger.info("Leak localizer stabilization state reset")

    def set_exclusion_zones(self, zones: Set[str]):
        """
        Set nodes to exclude from localization.
        
        These are typically nodes near confirmed leaks that should not
        be considered for new leak detection.
        
        Args:
            zones: Set of node IDs to exclude
        """
        self._exclusion_zones = zones
        logger.info(f"Exclusion zones set: {len(zones)} nodes excluded from localization")

    def add_exclusion_zone(self, node_id: str, depth: int = 2):
        """
        Add a node and its neighbors to exclusion zones.
        
        Args:
            node_id: Center node of exclusion zone
            depth: How many hops around the node to exclude
        """
        self._exclusion_zones.add(node_id)
        neighbors = self._get_neighbors(node_id, depth)
        self._exclusion_zones.update(neighbors)
        logger.info(f"Added exclusion zone around {node_id}: {len(neighbors) + 1} nodes")

    def localize(
        self,
        inference_results: Dict[str, InferenceResult],
        anomaly_threshold: float = 0.5
    ) -> Optional[LocalizationResult]:
        """
        Estimate leak location from inference results (single leak mode).

        Returns the highest-confidence detection, excluding nodes in exclusion zones.
        For multiple leaks, use localize_multiple() instead.

        Args:
            inference_results: Inference results from all sensors
            anomaly_threshold: Minimum confidence to consider as anomaly

        Returns:
            LocalizationResult or None if no leak detected
        """
        # Get anomalous sensors sorted by confidence
        # Filter out nodes in exclusion zones (near confirmed leaks)
        anomalies = [
            (node_id, result)
            for node_id, result in inference_results.items()
            if result.confidence >= anomaly_threshold
            and node_id not in self._exclusion_zones
        ]

        if not anomalies:
            return None

        anomalies.sort(key=lambda x: x[1].confidence, reverse=True)

        # Get raw localization - return the highest confidence result directly
        top_node, top_result = anomalies[0]

        if len(anomalies) == 1:
            return LocalizationResult(
                estimated_node=top_node,
                confidence=top_result.confidence,
                is_sensored=True,
                nearby_sensors=[top_node],
                candidate_region=self._get_neighbors(top_node, depth=2) + [top_node],
                method="direct_detection",
                cluster_id=0,
                contributing_anomalies=[top_node]
            )
        
        # Triangulation for multiple anomalies
        return self._triangulate(anomalies, anomaly_threshold)

    def localize_multiple(
        self,
        inference_results: Dict[str, InferenceResult],
        anomaly_threshold: float = 0.5,
        max_leaks: int = 5
    ) -> Optional[MultiLeakResult]:
        """
        Detect and localize MULTIPLE simultaneous leaks.
        
        Clusters anomalous sensors by network topology, then localizes
        each cluster independently. Uses confidence-weighted scoring
        to distinguish real leak clusters from network-wide pressure effects.

        Args:
            inference_results: Inference results from all sensors
            anomaly_threshold: Minimum confidence to consider as anomaly
            max_leaks: Maximum number of leaks to report (prevents over-detection)

        Returns:
            MultiLeakResult containing all detected leaks, or None if no leaks
        """
        # Get anomalous sensors
        anomalies = [
            (node_id, result)
            for node_id, result in inference_results.items()
            if result.confidence >= anomaly_threshold
        ]

        if not anomalies:
            return None

        # Cluster anomalies by network proximity
        # Use higher confidence threshold for cluster seeds
        high_conf_threshold = anomaly_threshold + 0.1
        clusters = self._cluster_anomalies(anomalies, min_confidence_for_cluster=high_conf_threshold)
        
        logger.info(f"Found {len(anomalies)} anomalies in {len(clusters)} cluster(s)")

        localizations = []
        for cluster_id, cluster in enumerate(clusters[:max_leaks]):
            # Localize each cluster independently
            loc = self._localize_cluster(cluster, cluster_id, anomaly_threshold)
            if loc:
                localizations.append(loc)

        if not localizations:
            return None

        # Calculate total confidence
        total_confidence = max(loc.confidence for loc in localizations)

        return MultiLeakResult(
            leak_count=len(localizations),
            localizations=localizations,
            total_confidence=total_confidence,
            clustering_method="topological_agglomerative"
        )

    def _localize_cluster(
        self,
        cluster_anomalies: List[Tuple[str, InferenceResult]],
        cluster_id: int,
        threshold: float
    ) -> Optional[LocalizationResult]:
        """Localize a single cluster of anomalies."""
        if not cluster_anomalies:
            return None

        cluster_anomalies.sort(key=lambda x: x[1].confidence, reverse=True)
        top_node, top_result = cluster_anomalies[0]
        contributing = [node for node, _ in cluster_anomalies]

        if len(cluster_anomalies) == 1:
            return LocalizationResult(
                estimated_node=top_node,
                confidence=top_result.confidence,
                is_sensored=True,
                nearby_sensors=[top_node],
                candidate_region=self._get_neighbors(top_node, depth=2) + [top_node],
                method="direct_detection",
                cluster_id=cluster_id,
                contributing_anomalies=contributing
            )

        # Triangulate within this cluster
        return self._triangulate(cluster_anomalies, threshold, cluster_id)

    def _triangulate(
        self,
        anomalies: List[Tuple[str, InferenceResult]],
        threshold: float,
        cluster_id: int = 0
    ) -> LocalizationResult:
        """
        Triangulate leak location from multiple sensor readings.

        The principle: a leak causes pressure drops that propagate through
        the network. Sensors closer to the leak see larger drops.
        
        NEW APPROACH: Find the node that is the CENTER of the anomaly pattern.
        - For each candidate node, compute weighted distance to all anomalous sensors
        - The weighted center should be where distance * confidence is minimized
        - Also consider: the sensor with highest confidence is likely NEAR the leak
        """
        contributing_anomalies = [node for node, _ in anomalies]
        
        # Sort anomalies by confidence - highest first
        sorted_anomalies = sorted(anomalies, key=lambda x: x[1].confidence, reverse=True)
        top_node, top_result = sorted_anomalies[0]
        
        # Weight each anomalous sensor by its confidence and z-score magnitude
        anomaly_weights = {}
        for node_id, result in anomalies:
            # Use z-score magnitude if available, else just confidence
            z_mag = abs(result.z_score) if hasattr(result, 'z_score') and result.z_score is not None else 0
            # Combined weight: confidence indicates certainty, z-score indicates magnitude
            weight = result.confidence * (1.0 + min(z_mag, 10.0) / 10.0)
            anomaly_weights[node_id] = weight
        
        # Strategy 1: The highest-confidence sensor is probably very close to the leak
        # Use it as an anchor - search only in its neighborhood
        anchor_depth = 4  # Search within 4 hops of top sensor
        
        # Get candidates near the anchor (top-confidence sensor)
        anchor_neighbors = set(self._get_neighbors(top_node, anchor_depth))
        anchor_neighbors.add(top_node)
        
        # Also add candidates near other high-confidence sensors
        for node_id, result in sorted_anomalies[1:3]:  # Top 3 sensors
            if result.confidence > 0.6:
                anchor_neighbors.update(self._get_neighbors(node_id, 3))
                anchor_neighbors.add(node_id)
        
        all_nodes = self._get_all_nodes()
        node_scores: Dict[str, float] = {}
        
        # Score candidates in the anchor region
        candidates = anchor_neighbors if anchor_neighbors else all_nodes
        
        for candidate in candidates:
            nearby = self._nearby_sensors_cache.get(candidate, [])
            if not nearby:
                continue
            
            # Compute weighted inverse distance score
            # High score = close to high-confidence anomalies
            score = 0.0
            total_weight = 0.0
            min_anomaly_dist = float('inf')
            
            for sensor_id, distance in nearby:
                if sensor_id in anomaly_weights:
                    weight = anomaly_weights[sensor_id]
                    # Sharp distance penalty: 1/(1+d)^2 instead of 1/(1+0.3d)
                    distance_factor = 1.0 / ((1.0 + distance) ** 1.5)
                    score += weight * distance_factor
                    total_weight += weight
                    min_anomaly_dist = min(min_anomaly_dist, distance)
            
            # Also check if candidate IS an anomalous sensor (distance 0)
            if candidate in anomaly_weights:
                weight = anomaly_weights[candidate]
                score += weight * 1.0  # Max distance factor for self
                total_weight += weight
                min_anomaly_dist = 0
            
            if total_weight > 0:
                # DON'T normalize by sensor count - reward being close to STRONG anomalies
                # Add bonus for being very close to any anomaly
                proximity_bonus = 1.0 / (1.0 + min_anomaly_dist) if min_anomaly_dist < float('inf') else 0
                node_scores[candidate] = score + proximity_bonus * max(anomaly_weights.values())

        if not node_scores:
            # Fallback to top anomaly
            return LocalizationResult(
                estimated_node=top_node,
                confidence=top_result.confidence,
                is_sensored=True,
                nearby_sensors=[top_node],
                candidate_region=[top_node],
                method="fallback_top_anomaly",
                cluster_id=cluster_id,
                contributing_anomalies=contributing_anomalies
            )

        # Find best candidate
        best_node = max(node_scores, key=node_scores.get)
        best_score = node_scores[best_node]

        # Get nearby sensors that contributed
        nearby = self._nearby_sensors_cache.get(best_node, [])
        contributing = [s for s, _ in nearby if s in anomaly_weights]
        
        # If best_node IS an anomalous sensor, include it
        if best_node in anomaly_weights and best_node not in contributing:
            contributing.insert(0, best_node)

        # Get candidate region (nodes with similar high scores)
        score_threshold = best_score * 0.85
        candidate_region = [
            n for n, s in node_scores.items()
            if s >= score_threshold
        ]

        # Confidence: use the top contributing sensor's confidence
        # boosted slightly if we have multiple confirming sensors
        base_conf = top_result.confidence
        multi_sensor_boost = min(0.1, 0.02 * len(contributing))
        confidence = min(1.0, base_conf + multi_sensor_boost)

        return LocalizationResult(
            estimated_node=best_node,
            confidence=confidence,
            is_sensored=best_node in self._sensored_nodes,
            nearby_sensors=contributing,
            candidate_region=candidate_region[:20],
            method="triangulation",
            cluster_id=cluster_id,
            contributing_anomalies=contributing_anomalies
        )

    def get_sensor_coverage_for_node(self, node_id: str) -> List[Tuple[str, int]]:
        """
        Get sensors that would detect a leak at the given node.

        Args:
            node_id: Node to check

        Returns:
            List of (sensor_id, distance) tuples
        """
        return self._nearby_sensors_cache.get(node_id, [])

    def estimate_detectability(self, node_id: str) -> float:
        """
        Estimate how detectable a leak at this node would be.

        Returns a score from 0 (undetectable) to 1 (highly detectable).
        """
        nearby = self._nearby_sensors_cache.get(node_id, [])

        if not nearby:
            return 0.0

        # Score based on:
        # 1. Number of nearby sensors
        # 2. Distance to nearest sensor
        # 3. Whether node itself has a sensor

        if nearby[0][1] == 0:  # Has sensor
            return 1.0

        min_distance = min(d for _, d in nearby)
        num_sensors = len(nearby)

        # Detectability decreases with distance, increases with sensor count
        distance_factor = 1.0 / (1.0 + min_distance * 0.5)
        coverage_factor = min(1.0, num_sensors / 3.0)  # Saturates at 3 sensors

        return 0.3 + 0.7 * distance_factor * coverage_factor
