"""
Orchestrator - Ties together all system components.
"""

import logging
import random
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from .config import SystemConfig, SamplingMode, DEFAULT_CONFIG
from .simulation import NetworkSimulator, DeviceSimulator, LeakInjector
from .simulation.device_simulator import DeviceFleet
from .data_layer import DataPipeline
from .ai_layer import AgentController, AgentState
from .agents import MultiAgentSystem, AgentSystemConfig

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Result of one orchestration cycle."""
    sim_time: float
    metrics: Dict[str, Dict]
    ground_truth: List[str]
    estimated_location: Optional[str]
    status: 'AgentStatus'
    anomaly: Optional['InferenceResult']
    # Multi-agent data
    agent_summary: Optional[Dict] = None
    detected_leaks: Optional[List[Dict]] = None


class SystemOrchestrator:
    """
    Main orchestrator that ties together all system components.

    Manages:
    - Network simulation
    - Device fleet
    - Leak injection
    - AI agent (single-agent OR multi-agent system)
    - Data flow between components
    """

    def __init__(
        self,
        config: SystemConfig = None,
        event_callback: Callable[[str], None] = None,
        use_multi_agent: bool = True  # NEW: Enable multi-agent by default
    ):
        """
        Initialize the system orchestrator.

        Args:
            config: System configuration
            event_callback: Callback for log events
            use_multi_agent: If True, use multi-agent system (SensorAgents + Coordinator + Localizer)
        """
        self.config = config or DEFAULT_CONFIG
        self._event_callback = event_callback
        self._use_multi_agent = use_multi_agent

        # Initialize components
        logger.info("Initializing system components...")

        # 1. Network Simulator (Physics Layer)
        self._network = NetworkSimulator(self.config.simulation)

        # 2. Device Fleet (IoT Layer)
        self._fleet = DeviceFleet(self.config.simulation)
        self._setup_devices()

        # 3. Leak Injector (Test Controller)
        self._leak_injector = LeakInjector(self._network)

        # 4. AI Agent Controller (original system - always created for compatibility)
        self._agent = AgentController(
            self._network,
            self._fleet,
            self.config.ai,
            event_callback=event_callback
        )
        
        # 5. Multi-Agent System (NEW)
        self._multi_agent_system: Optional[MultiAgentSystem] = None
        if use_multi_agent:
            self._setup_multi_agent_system()

        # Simulation state
        self._sim_time = 0.0
        self._time_step = self.config.simulation.hydraulic_timestep_seconds
        self._is_running = False

        # Run initial simulation if using WNTR
        if not self._network.is_mock:
            self._network.run_simulation()

        logger.info(f"System initialized with {len(self._fleet.device_ids)} devices")
        logger.info(f"Network: {'MOCK' if self._network.is_mock else 'WNTR'}")
        if use_multi_agent:
            logger.info(f"Multi-Agent System: ENABLED ({self._multi_agent_system.sensor_agents.__len__()} sensors + coordinator + localizer)")
    
    def _setup_multi_agent_system(self):
        """Initialize the multi-agent system with sensor nodes."""
        # Get sensor node IDs from the fleet
        sensor_nodes = list(self._fleet.monitored_nodes)
        
        # Compute network distances for localization (simplified)
        # In production, this would use actual network topology
        distances = self._compute_sensor_distances()
        
        # Create multi-agent system
        mas_config = AgentSystemConfig(
            sensor_buffer_size=30,
            coordinator_aggregation_window=10.0,
            min_alerts_for_investigation=2
        )
        
        self._multi_agent_system = MultiAgentSystem(
            sensor_nodes=sensor_nodes,
            network_distances=distances,
            config=mas_config
        )
        self._multi_agent_system.initialize()
        
        self._emit_event(f"[cyan]Multi-Agent System initialized: {len(sensor_nodes)} sensor agents[/cyan]")
    
    def _compute_sensor_distances(self) -> Dict[str, Dict[str, float]]:
        """Compute approximate distances between sensor nodes."""
        distances = {}
        nodes = list(self._fleet.monitored_nodes)
        
        # Use network topology if available
        if hasattr(self._network, '_wn') and self._network._wn is not None:
            try:
                import networkx as nx
                G = self._network._wn.to_graph()
                
                for i, node_a in enumerate(nodes):
                    distances[node_a] = {}
                    for node_b in nodes:
                        if node_a != node_b:
                            try:
                                # Shortest path length as distance proxy
                                path_len = nx.shortest_path_length(G.to_undirected(), node_a, node_b)
                                distances[node_a][node_b] = float(path_len * 100)  # Scale to meters
                            except (nx.NetworkXNoPath, nx.NodeNotFound):
                                distances[node_a][node_b] = 1000.0  # Default far distance
            except Exception as e:
                logger.warning(f"Could not compute network distances: {e}")
        
        return distances

    def _setup_devices(self):
        """Set up sensor devices based on defined locations in the .inp file."""
        # Get sensor locations from the .inp file comments
        pressure_sensor_nodes, amr_nodes = self._network.get_sensor_locations()

        if not pressure_sensor_nodes:
            # Fallback: use a subset of junctions if no sensors defined
            junctions = self._network.junction_names
            pressure_sensor_nodes = junctions[:20]
            amr_nodes = junctions[::5]
            logger.warning("No sensor locations found in .inp file, using fallback")

        # Create pressure sensors at defined locations
        for i, node_id in enumerate(pressure_sensor_nodes):
            self._fleet.add_device(
                device_id=f"PRESSURE_{i:04d}",
                node_id=node_id,
                reading_type='pressure'
            )

        # Create flow sensors (AMR) at defined locations
        for i, node_id in enumerate(amr_nodes):
            self._fleet.add_device(
                device_id=f"FLOW_{i:04d}",
                node_id=node_id,
                reading_type='flow'
            )

        logger.info(f"Created {len(self._fleet.device_ids)} sensor devices")
        logger.info(f"  - {len(pressure_sensor_nodes)} pressure sensors")
        logger.info(f"  - {len(amr_nodes)} AMR flow meters")

    def _emit_event(self, message: str):
        """Emit a log event."""
        if self._event_callback:
            self._event_callback(message)

    def inject_random_leak(self, at_sensor: bool = True) -> Optional[str]:
        """
        Inject a leak at a random node.

        Args:
            at_sensor: If True, inject at a node with a PRESSURE sensor (detectable).
                       If False, inject at any junction (may be undetectable).

        Returns:
            Node ID where leak was injected, or None
        """
        # Lock baseline before first leak injection to prevent contamination
        if not self._agent.pipeline._baseline_locked:
            self._agent.pipeline.lock_baseline()
            self._emit_event("[cyan]Baseline locked[/cyan]")
        
        if at_sensor:
            # Get nodes with PRESSURE sensors (not just any monitored node)
            # Pressure sensors are the key to detecting leaks
            pressure_sensor_nodes, _ = self._network.get_sensor_locations()
            candidates = list(pressure_sensor_nodes)
            if not candidates:
                # Fallback to any monitored node
                candidates = self._fleet.monitored_nodes
            if not candidates:
                logger.warning("No sensor nodes available for leak injection")
                return None
        else:
            # Any junction in the network
            candidates = self._network.junction_names

        # Pick a random node
        node_id = random.choice(candidates)

        # Random leak rate between 3-8 L/s
        leak_rate = random.uniform(3.0, 8.0)

        event = self._leak_injector.inject_leak(node_id, leak_rate, self._sim_time)

        if event:
            has_pressure = node_id in self._network.get_sensor_locations()[0]
            sensor_status = "[PRESSURE SENSOR]" if has_pressure else "[NO PRESSURE SENSOR]"
            self._emit_event(f"[red]LEAK INJECTED at {node_id} ({leak_rate:.1f} L/s) {sensor_status}[/red]")

            # Re-run simulation with leak
            if not self._network.is_mock:
                self._network.run_simulation()

            return node_id

        return None

    def clear_all_leaks(self):
        """Remove all active leaks."""
        self._leak_injector.remove_all_leaks(self._sim_time)
        self._emit_event("[green]All leaks cleared[/green]")
        
        # FULL Reset of legacy Single-Agent to prevent ghost detections
        self._agent.reset()
        
        # Completely re-initialize Multi-Agent System to ensure clean state
        # This prevents any lingering state (messages, active investigations) from persisting
        if self._use_multi_agent:
            # Explicitly shutdown old system if needed
            if self._multi_agent_system:
                # Optional: self._multi_agent_system.shutdown() 
                pass
            self._setup_multi_agent_system()
        
        # Clear all confirmed leak exclusion zones
        self._agent.clear_confirmed_leaks()

        # Re-run simulation without leaks
        if not self._network.is_mock:
            self._network.run_simulation()

    def get_ground_truth(self) -> List[str]:
        """Get list of actual leak locations (ground truth)."""
        return self._leak_injector.get_ground_truth()

    def get_detection_summary(self) -> str:
        """Get leak detection accuracy summary."""
        return self._leak_injector.get_detection_summary()

    def record_detection(self, estimated_node: str):
        """Record that the AI detected a leak.
        
        Matches the detection to the closest undetected leak.
        """
        ground_truth = self.get_ground_truth()
        if not ground_truth:
            return
        
        # Check if system is in cooldown
        if self._agent.is_in_cooldown(self._sim_time):
            logger.debug(f"Detection skipped - system in cooldown")
            return
        
        # Find the closest active leak to the estimated location
        best_leak = None
        best_distance = 999
        
        for actual_node in ground_truth:
            # Check if this leak was already detected
            active_leaks = self._leak_injector.get_active_leaks()
            if actual_node in active_leaks and active_leaks[actual_node].detected:
                continue  # Skip already-detected leaks
            
            # Calculate distance
            dist = self._calculate_detection_distance(actual_node, estimated_node)
            if dist < best_distance:
                best_distance = dist
                best_leak = actual_node
        
        if best_leak:
            self._leak_injector.record_detection(
                best_leak,
                estimated_node,
                self._sim_time
            )
            # Start cooldown after successful detection
            self._agent.start_cooldown(self._sim_time)
    
    def confirm_leak(self, node_id: str) -> bool:
        """
        Confirm a detected leak and add it to exclusion zones.
        
        This prevents the confirmed leak from interfering with
        detection of new leaks.
        
        Args:
            node_id: Node where the leak is confirmed
            
        Returns:
            True if leak was confirmed, False if not found
        """
        # Get pressure signature (empty for now - we don't track it yet)
        pressure_signature = {}
        
        # Confirm in leak injector
        confirmed = self._leak_injector.confirm_leak(
            node_id, 
            pressure_signature, 
            self._sim_time
        )
        
        if confirmed:
            # Add to agent controller exclusion zones
            self._agent.confirm_leak(node_id, depth=2)
            self._emit_event(f"[cyan]✓ Leak at {node_id} confirmed and masked[/cyan]")
            return True
        
        return False
    
    def auto_confirm_detections(self, min_cycles: int = 5):
        """
        Automatically confirm leaks that have been detected for multiple cycles.
        
        Args:
            min_cycles: Minimum stable detection cycles before auto-confirm
        """
        active_leaks = self._leak_injector.get_active_leaks()
        
        for node_id, event in active_leaks.items():
            if event.detected and not event.confirmed:
                # For now, confirm immediately after detection
                # In a real system, you'd check detection_time vs current time
                self.confirm_leak(node_id)
    
    def get_confirmed_leaks(self) -> List[str]:
        """Get list of confirmed leak locations."""
        return self._leak_injector.get_confirmed_leaks()
    
    def get_unconfirmed_detections(self) -> List[str]:
        """Get detected but not yet confirmed leaks."""
        active_leaks = self._leak_injector.get_active_leaks()
        return [
            node_id for node_id, event in active_leaks.items()
            if event.detected and not event.confirmed
        ]
    
    def _calculate_detection_distance(self, actual_node: str, estimated_node: str) -> int:
        """Calculate network distance between two nodes."""
        if actual_node == estimated_node:
            return 0
        
        for depth in range(1, 10):
            neighbors = self._network.get_node_neighbors(actual_node, depth=depth)
            if estimated_node in neighbors:
                return depth
        
        return 99

    def step(self) -> OrchestrationResult:
        """
        Run one simulation step.

        Returns:
            OrchestrationResult with all current state
        """
        # Advance simulation time
        self._sim_time += self._time_step

        # Run AI monitoring cycle (original single-agent)
        cycle_result = self._agent.run_cycle(self._sim_time)

        # Get current network state for metrics display
        state = self._network.get_state_at_time(self._sim_time)

        # Build metrics dictionary for display
        metrics = {}
        for node_id in self._fleet.monitored_nodes:
            pressure = state.pressures.get(node_id, 0)
            # Use node_flows for junction-level flow (not pipe flows)
            flow = state.node_flows.get(node_id, 0) if state.node_flows else state.demands.get(node_id, 0)

            # Get Z-scores from features
            features = cycle_result.features.get(node_id)
            inference = cycle_result.inference_results.get(node_id)

            metrics[node_id] = {
                'pressure': pressure,
                'flow': flow,
                'pressure_zscore': features.pressure_zscore if features else None,
                'flow_zscore': features.flow_zscore if features else None,
                'confidence': inference.confidence if inference else 0
            }

        # === MULTI-AGENT SYSTEM STEP ===
        agent_summary = None
        detected_leaks = None
        
        if self._multi_agent_system:
            # Build environment for multi-agent system
            readings = {
                node_id: {"pressure": state.pressures.get(node_id, 0)}
                for node_id in self._fleet.monitored_nodes
            }
            environment = {
                "readings": readings,
                "sim_time": self._sim_time
            }
            
            # Run multi-agent step
            agent_summary = self._multi_agent_system.step(environment)
            detected_leaks = self._multi_agent_system.get_detected_leaks()
            
            # Log coordinator alerts
            coord_status = agent_summary.get("coordinator", {})
            if coord_status.get("system_mode") == "INVESTIGATING":
                active_invs = agent_summary.get("active_investigations", [])
                for inv in active_invs:
                    if inv["status"] == "localized" and inv.get("localization"):
                        loc = inv["localization"]
                        node = loc.get("probable_location", "unknown")
                        conf = loc.get("confidence", 0)
                        self._emit_event(f"[yellow]🔍 Multi-Agent: Leak localized at {node} ({conf:.0%} confidence)[/yellow]")

        # Get AI's estimate (original system)
        estimated_location = self._agent.get_estimated_leak_location()
        
        # === HYBRID DETECTION STRATEGY ===
        # Use Multi-Agent System results if available (Priority 1)
        if self._multi_agent_system and detected_leaks:
            # Find high-confidence leak
            best_leak = max(detected_leaks, key=lambda x: x.get('confidence', 0))
            if best_leak.get('confidence', 0) > 0.5:
                # Override single-agent estimate with MAS result
                estimated_location = best_leak['location']
                # Record detection directly from MAS
                self.record_detection(estimated_location)

        # Fallback to single-agent logic if MAS didn't trigger
        elif (estimated_location and
            cycle_result.agent_status.system_status.name == 'LEAK_CONFIRMED'):
            self.record_detection(estimated_location)

        # Get top anomaly for UI (from original system)
        anomalies = self._agent.inference_engine.get_anomalies(
            cycle_result.inference_results
        )
        top_anomaly = anomalies[0] if anomalies else None

        return OrchestrationResult(
            sim_time=self._sim_time,
            metrics=metrics,
            ground_truth=self.get_ground_truth(),
            estimated_location=estimated_location,
            status=cycle_result.agent_status,
            anomaly=top_anomaly,
            agent_summary=agent_summary,
            detected_leaks=detected_leaks
        )

    def reset(self):
        """Reset the entire system."""
        self._sim_time = 0.0
        self._leak_injector.reset()
        self._agent.reset()
        self._network.reset()
        
        # Reset multi-agent system
        if self._use_multi_agent:
            self._setup_multi_agent_system()

        if not self._network.is_mock:
            self._network.run_simulation()

        self._emit_event("[yellow]System reset complete[/yellow]")

    @property
    def sim_time(self) -> float:
        """Current simulation time in seconds."""
        return self._sim_time

    @property
    def network(self) -> NetworkSimulator:
        """Access to network simulator."""
        return self._network

    @property
    def agent(self) -> AgentController:
        """Access to AI agent."""
        return self._agent
    
    @property
    def multi_agent_system(self) -> Optional[MultiAgentSystem]:
        """Access to multi-agent system."""
        return self._multi_agent_system

    @property
    def leak_injector(self) -> LeakInjector:
        """Access to leak injector."""
        return self._leak_injector
