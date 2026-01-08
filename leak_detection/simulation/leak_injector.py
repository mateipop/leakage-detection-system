"""
Leak Injector - Controlled leak injection for testing the detection system.
"""

import logging
import random
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

from .network_simulator import NetworkSimulator

logger = logging.getLogger(__name__)


@dataclass
class LeakEvent:
    """Record of a leak injection event."""
    leak_id: str
    node_id: str
    start_time: float
    leak_rate: float  # L/s
    end_time: Optional[float] = None
    detected: bool = False
    detection_time: Optional[float] = None
    estimated_location: Optional[str] = None
    distance_hops: Optional[int] = None  # Network distance between actual and estimated
    confirmed: bool = False  # Has this leak been confirmed and should be masked?
    confirmation_time: Optional[float] = None
    pressure_signature: Optional[Dict[str, float]] = None  # Pressure drop caused at each sensor


class LeakInjector:
    """
    Controls leak injection for testing the detection system.

    Provides methods to:
    - Inject leaks at specific nodes
    - Inject random leaks
    - Track leak history for evaluation
    - Compare ground truth with AI predictions
    - Manage confirmed leaks for masking
    """

    def __init__(self, network: NetworkSimulator):
        """
        Initialize the leak injector.

        Args:
            network: The network simulator instance
        """
        self.network = network
        self._leak_history: List[LeakEvent] = []
        self._active_leaks: Dict[str, LeakEvent] = {}
        self._confirmed_leaks: Dict[str, LeakEvent] = {}  # Leaks that are confirmed and masked
        self._leak_counter = 0

    def inject_leak(
        self,
        node_id: str,
        leak_rate: float = 5.0,
        sim_time: float = 0.0
    ) -> Optional[LeakEvent]:
        """
        Inject a leak at a specific node.

        Args:
            node_id: Junction node ID where leak occurs
            leak_rate: Leak rate in L/s
            sim_time: Current simulation time

        Returns:
            LeakEvent if successful, None otherwise
        """
        if node_id in self._active_leaks:
            logger.warning(f"Leak already active at {node_id}")
            return None

        success = self.network.inject_leak(node_id, leak_rate)
        if not success:
            return None

        self._leak_counter += 1
        leak_id = f"LEAK_{self._leak_counter:04d}"

        event = LeakEvent(
            leak_id=leak_id,
            node_id=node_id,
            start_time=sim_time,
            leak_rate=leak_rate
        )

        self._active_leaks[node_id] = event
        self._leak_history.append(event)

        logger.info(f"Injected {leak_id} at node {node_id} ({leak_rate} L/s)")
        return event

    def inject_random_leak(
        self,
        leak_rate_range: Tuple[float, float] = (2.0, 10.0),
        sim_time: float = 0.0,
        exclude_nodes: List[str] = None
    ) -> Optional[LeakEvent]:
        """
        Inject a leak at a random junction node.

        Args:
            leak_rate_range: (min, max) leak rate in L/s
            sim_time: Current simulation time
            exclude_nodes: Nodes to exclude from selection

        Returns:
            LeakEvent if successful, None otherwise
        """
        exclude_nodes = exclude_nodes or []
        available_nodes = [
            n for n in self.network.junction_names
            if n not in self._active_leaks and n not in exclude_nodes
        ]

        if not available_nodes:
            logger.warning("No available nodes for leak injection")
            return None

        node_id = random.choice(available_nodes)
        leak_rate = random.uniform(*leak_rate_range)

        return self.inject_leak(node_id, leak_rate, sim_time)

    def remove_leak(self, node_id: str, sim_time: float = 0.0) -> bool:
        """
        Remove a leak from a node.

        Args:
            node_id: Node to remove leak from
            sim_time: Current simulation time

        Returns:
            True if leak was removed
        """
        if node_id not in self._active_leaks:
            return False

        event = self._active_leaks.pop(node_id)
        event.end_time = sim_time

        success = self.network.remove_leak(node_id)
        logger.info(f"Removed {event.leak_id} from node {node_id}")

        return success

    def remove_all_leaks(self, sim_time: float = 0.0):
        """Remove all active leaks."""
        for node_id in list(self._active_leaks.keys()):
            self.remove_leak(node_id, sim_time)

    def record_detection(
        self,
        actual_node: str,
        estimated_node: str,
        detection_time: float
    ) -> Optional[LeakEvent]:
        """
        Record that the AI detected a leak.

        Args:
            actual_node: The actual leak location (from active leaks)
            estimated_node: The AI's estimated location
            detection_time: Time of detection

        Returns:
            Updated LeakEvent if found, None if already detected or not found
        """
        if actual_node in self._active_leaks:
            event = self._active_leaks[actual_node]
            
            # Don't re-record if already detected
            if event.detected:
                return None
            
            event.detected = True
            event.detection_time = detection_time
            event.estimated_location = estimated_node
            
            # Calculate network distance
            event.distance_hops = self._calculate_distance(actual_node, estimated_node)
            
            logger.info(f"Recorded detection: actual={actual_node}, estimated={estimated_node}, distance={event.distance_hops} hops")
            return event

        # Try to find the nearest undetected active leak
        for node_id, event in self._active_leaks.items():
            if not event.detected:
                event.detected = True
                event.detection_time = detection_time
                event.estimated_location = estimated_node
                event.distance_hops = self._calculate_distance(node_id, estimated_node)
                return event

        return None

    def confirm_leak(
        self,
        node_id: str,
        pressure_signature: Dict[str, float],
        confirmation_time: float
    ) -> Optional[LeakEvent]:
        """
        Confirm a detected leak and add it to the masked list.
        
        Once confirmed, the leak's pressure signature will be subtracted
        from sensor readings so new leaks can be detected.
        
        Args:
            node_id: The actual or estimated leak location
            pressure_signature: Pressure drop at each sensor caused by this leak
            confirmation_time: Time of confirmation
            
        Returns:
            The confirmed LeakEvent or None if not found
        """
        # Find the leak event (check active leaks)
        event = self._active_leaks.get(node_id)
        
        if event is None:
            # Try to find by estimated location
            for actual_node, evt in self._active_leaks.items():
                if evt.estimated_location == node_id and evt.detected and not evt.confirmed:
                    event = evt
                    break
        
        if event is None:
            logger.warning(f"No active leak found to confirm at {node_id}")
            return None
        
        event.confirmed = True
        event.confirmation_time = confirmation_time
        event.pressure_signature = pressure_signature
        
        # Add to confirmed leaks (for masking)
        self._confirmed_leaks[event.node_id] = event
        
        logger.info(f"Confirmed leak {event.leak_id} at {event.node_id} - will be masked from detection")
        return event

    def get_confirmed_leaks(self) -> Dict[str, LeakEvent]:
        """Get all confirmed (masked) leaks."""
        return dict(self._confirmed_leaks)

    def get_unconfirmed_detected_leaks(self) -> List[LeakEvent]:
        """Get leaks that are detected but not yet confirmed."""
        return [e for e in self._active_leaks.values() if e.detected and not e.confirmed]

    def get_total_pressure_signature(self) -> Dict[str, float]:
        """
        Get combined pressure signature of all confirmed leaks.
        
        This should be subtracted from sensor readings to mask known leaks.
        """
        combined = {}
        for event in self._confirmed_leaks.values():
            if event.pressure_signature:
                for node_id, pressure_drop in event.pressure_signature.items():
                    combined[node_id] = combined.get(node_id, 0.0) + pressure_drop
        return combined

    def get_exclusion_zones(self, depth: int = 2) -> set:
        """
        Get nodes that should be excluded from new leak detection.
        
        Returns nodes within 'depth' hops of confirmed leaks.
        """
        exclusion = set()
        for node_id in self._confirmed_leaks.keys():
            exclusion.add(node_id)
            neighbors = self.network.get_node_neighbors(node_id, depth=depth)
            exclusion.update(neighbors)
        return exclusion
    
    def _calculate_distance(self, actual_node: str, estimated_node: str) -> int:
        """Calculate network distance (hops) between two nodes."""
        if actual_node == estimated_node:
            return 0
        
        for depth in range(1, 10):
            neighbors = self.network.get_node_neighbors(actual_node, depth=depth)
            if estimated_node in neighbors:
                return depth
        
        return 99  # Far away / not found

    def get_active_leaks(self) -> Dict[str, LeakEvent]:
        """Get all currently active leaks."""
        return dict(self._active_leaks)

    def get_ground_truth(self) -> List[str]:
        """Get list of nodes with active leaks (ground truth)."""
        return list(self._active_leaks.keys())

    def get_leak_history(self) -> List[LeakEvent]:
        """Get full leak history."""
        return list(self._leak_history)
    
    def get_last_detections(self, n: int = 5) -> List[Dict]:
        """
        Get the last N detection results with accuracy info.
        
        Args:
            n: Number of recent detections to return
            
        Returns:
            List of detection result dictionaries
        """
        detected_events = [e for e in self._leak_history if e.detected]
        recent = detected_events[-n:] if len(detected_events) >= n else detected_events
        
        results = []
        for event in reversed(recent):  # Most recent first
            distance = event.distance_hops if event.distance_hops is not None else 99
            
            if distance == 0:
                accuracy = "EXACT!"
                accuracy_color = "green"
            elif distance == 1:
                accuracy = "1 hop"
                accuracy_color = "cyan"
            elif distance <= 2:
                accuracy = f"{distance} hops"
                accuracy_color = "yellow"
            elif distance <= 4:
                accuracy = f"{distance} hops"
                accuracy_color = "orange3"
            else:
                accuracy = "FAR"
                accuracy_color = "red"
            
            delay = (event.detection_time - event.start_time) if event.detection_time else 0
            
            results.append({
                "leak_id": event.leak_id,
                "actual": event.node_id,
                "estimated": event.estimated_location,
                "distance": distance,
                "accuracy": accuracy,
                "accuracy_color": accuracy_color,
                "delay_seconds": delay,
                "leak_rate": event.leak_rate
            })
        
        return results

    def calculate_detection_accuracy(self) -> Dict:
        """
        Calculate detection accuracy metrics.

        Returns:
            Dictionary with accuracy metrics
        """
        if not self._leak_history:
            return {
                'total_leaks': 0,
                'detected': 0,
                'detection_rate': 0.0,
                'correct_location': 0,
                'location_accuracy': 0.0,
                'avg_detection_delay': 0.0,
                'details': []
            }

        detected = sum(1 for e in self._leak_history if e.detected)
        correct = sum(
            1 for e in self._leak_history
            if e.detected and e.estimated_location == e.node_id
        )

        # Calculate average detection delay
        delays = [
            e.detection_time - e.start_time
            for e in self._leak_history
            if e.detected and e.detection_time is not None
        ]
        avg_delay = sum(delays) / len(delays) if delays else 0.0

        # Calculate location error (using topology distance)
        details = []
        for event in self._leak_history:
            detail = {
                'leak_id': event.leak_id,
                'actual_node': event.node_id,
                'estimated_node': event.estimated_location,
                'detected': event.detected,
                'correct_location': event.detected and event.estimated_location == event.node_id,
                'detection_delay': (
                    event.detection_time - event.start_time
                    if event.detected and event.detection_time
                    else None
                )
            }

            # Check if estimate is a neighbor
            if event.detected and event.estimated_location:
                neighbors = self.network.get_node_neighbors(event.node_id, depth=2)
                detail['is_neighbor'] = event.estimated_location in neighbors

            details.append(detail)

        return {
            'total_leaks': len(self._leak_history),
            'detected': detected,
            'detection_rate': detected / len(self._leak_history) if self._leak_history else 0.0,
            'correct_location': correct,
            'location_accuracy': correct / detected if detected > 0 else 0.0,
            'avg_detection_delay': avg_delay,
            'details': details
        }

    def get_detection_summary(self) -> str:
        """Get a human-readable detection summary."""
        metrics = self.calculate_detection_accuracy()

        lines = [
            "=" * 50,
            "LEAK DETECTION SUMMARY",
            "=" * 50,
            f"Total Leaks Injected: {metrics['total_leaks']}",
            f"Leaks Detected: {metrics['detected']} ({metrics['detection_rate']*100:.1f}%)",
            f"Correct Locations: {metrics['correct_location']} ({metrics['location_accuracy']*100:.1f}%)",
            f"Avg Detection Delay: {metrics['avg_detection_delay']:.1f} seconds",
            "",
            "DETAILED RESULTS:",
            "-" * 50
        ]

        for detail in metrics['details']:
            status = "[OK] DETECTED" if detail['detected'] else "[X] MISSED"
            loc_status = ""
            if detail['detected']:
                if detail['correct_location']:
                    loc_status = "[EXACT MATCH]"
                elif detail.get('is_neighbor'):
                    loc_status = "[NEAR MISS - neighbor node]"
                else:
                    loc_status = "[WRONG LOCATION]"

            lines.append(f"  {detail['leak_id']}: {status} {loc_status}")
            lines.append(f"    Actual: {detail['actual_node']}")
            if detail['estimated_node']:
                lines.append(f"    Estimated: {detail['estimated_node']}")
            if detail['detection_delay'] is not None:
                lines.append(f"    Delay: {detail['detection_delay']:.1f}s")
            lines.append("")

        lines.append("=" * 50)
        return "\n".join(lines)

    def reset(self):
        """Reset all leak history and active leaks."""
        self.remove_all_leaks()
        self._leak_history.clear()
        self._confirmed_leaks.clear()
        self._leak_counter = 0
