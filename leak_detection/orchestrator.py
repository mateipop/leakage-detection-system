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
            at_sensor: If True, inject at a node with a sensor (detectable).
                       If False, inject at any junction (may be undetectable).

        Returns:
            Node ID where leak was injected, or None
        """
        if at_sensor:
            # Get monitored nodes (where we have sensors)
            candidates = self._fleet.monitored_nodes
            if not candidates:
                logger.warning("No monitored nodes available for leak injection")
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
            is_monitored = node_id in self._fleet.monitored_nodes
            sensor_status = "[SENSORED]" if is_monitored else "[UNSENSORED]"
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
        """Record that the AI detected a leak."""
        ground_truth = self.get_ground_truth()
        if ground_truth:
            self._leak_injector.record_detection(
                ground_truth[0],  # First active leak
                estimated_node,
                self._sim_time
            )

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
            flow = state.flows.get(node_id, 0)

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

        # Get top anomaly (original system)
        anomalies = self._agent.inference_engine.get_anomalies(
            cycle_result.inference_results
        )
        top_anomaly = anomalies[0] if anomalies else None

        # Get AI's estimate (original system)
        estimated_location = self._agent.get_estimated_leak_location()

        # Record detection if leak confirmed
        if (cycle_result.decision.estimated_leak_location and
            cycle_result.agent_status.system_status.name == 'LEAK_CONFIRMED'):
            self.record_detection(estimated_location)

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
