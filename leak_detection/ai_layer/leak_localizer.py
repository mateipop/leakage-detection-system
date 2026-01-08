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
from collections import defaultdict

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
        cluster_separation_depth: int = None  # Auto-computed if None
    ):
        """
        Initialize the localizer.

        Args:
            get_neighbors_fn: Function(node_id, depth) -> List[str] to get neighbors
            get_all_nodes_fn: Function() -> List[str] to get all network nodes
            sensored_nodes: Set of node IDs that have sensors
            cluster_separation_depth: Min hops between sensors to consider them separate clusters
                                      If None, auto-computed from median sensor distance
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

        logger.info(f"LeakLocalizer initialized with {len(sensored_nodes)} sensors, "
                    f"cluster_separation={self._cluster_separation_depth}")

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
        
        for i, sensor1 in enumerate(sensors):
            for sensor2 in sensors[i+1:]:
                # Find shortest path distance
                distance = self._find_sensor_distance(sensor1, sensor2, max_depth)
                self._sensor_distances[(sensor1, sensor2)] = distance
                self._sensor_distances[(sensor2, sensor1)] = distance

    def _find_sensor_distance(self, sensor1: str, sensor2: str, max_depth: int) -> int:
        """Find the network distance (hops) between two sensors."""
        if sensor1 == sensor2:
            return 0
        
        for depth in range(1, max_depth + 1):
            neighbors = self._get_neighbors(sensor1, depth)
            if sensor2 in neighbors:
                return depth
        
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

    def localize(
        self,
        inference_results: Dict[str, InferenceResult],
        anomaly_threshold: float = 0.5
    ) -> Optional[LocalizationResult]:
        """
        Estimate leak location from inference results (single leak mode).

        For multiple leaks, use localize_multiple() instead.

        Args:
            inference_results: Inference results from all sensors
            anomaly_threshold: Minimum confidence to consider as anomaly

        Returns:
            LocalizationResult or None if no leak detected
        """
        # Get anomalous sensors sorted by confidence
        anomalies = [
            (node_id, result)
            for node_id, result in inference_results.items()
            if result.confidence >= anomaly_threshold
        ]

        if not anomalies:
            return None

        anomalies.sort(key=lambda x: x[1].confidence, reverse=True)

        # Method 1: Direct detection - highest confidence sensor
        top_node, top_result = anomalies[0]

        if len(anomalies) == 1:
            # Single anomaly - likely at or near that sensor
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

        # Method 2: Triangulation - multiple anomalous sensors
        # Find the region where anomalies cluster
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
        We find the point that best explains the observed pattern.
        """
        # Score each node based on how well it explains the anomaly pattern
        all_nodes = self._get_all_nodes()
        node_scores: Dict[str, float] = defaultdict(float)
        contributing_anomalies = [node for node, _ in anomalies]

        # Weight each anomalous sensor by its confidence
        anomaly_weights = {
            node_id: result.confidence
            for node_id, result in anomalies
        }

        for candidate in all_nodes:
            nearby = self._nearby_sensors_cache.get(candidate, [])

            if not nearby:
                continue

            score = 0.0
            contributing_sensors = []

            for sensor_id, distance in nearby:
                if sensor_id in anomaly_weights:
                    # Closer sensors with higher confidence = higher score
                    # Weight decreases with distance (inverse relationship)
                    weight = anomaly_weights[sensor_id]
                    distance_factor = 1.0 / (1.0 + distance * 0.3)
                    score += weight * distance_factor
                    contributing_sensors.append(sensor_id)

            if contributing_sensors:
                # Normalize by number of contributing sensors
                node_scores[candidate] = score / len(contributing_sensors)

        if not node_scores:
            # Fallback to top anomaly
            top_node, top_result = anomalies[0]
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

        # Get candidate region (nodes with similar scores)
        score_threshold = best_score * 0.8
        candidate_region = [
            n for n, s in node_scores.items()
            if s >= score_threshold
        ]

        # Confidence based on score relative to max possible
        max_possible = sum(anomaly_weights.values())
        confidence = min(1.0, best_score / max_possible) if max_possible > 0 else 0.5

        return LocalizationResult(
            estimated_node=best_node,
            confidence=confidence,
            is_sensored=best_node in self._sensored_nodes,
            nearby_sensors=contributing,
            candidate_region=candidate_region[:20],  # Limit size
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
