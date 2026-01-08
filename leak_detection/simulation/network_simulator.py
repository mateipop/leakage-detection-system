"""
Network Simulator - WNTR-based hydraulic simulation.
"""

import os
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

try:
    import wntr
    WNTR_AVAILABLE = True
except ImportError:
    WNTR_AVAILABLE = False

from ..config import SimulationConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class SimulationState:
    """Current state of the hydraulic simulation."""
    time_seconds: float
    pressures: Dict[str, float]  # node_id -> pressure (psi)
    flows: Dict[str, float]      # link_id -> flow (L/s)
    demands: Dict[str, float]    # node_id -> demand (L/s)
    node_flows: Dict[str, float] = None  # node_id -> flow at node (L/s)


class MockNetwork:
    """
    Mock network for when WNTR or L-Town.inp is unavailable.
    Creates a simple test network for demonstration.
    """

    def __init__(self):
        self.junction_names = [f"J{i}" for i in range(1, 21)]
        self.pipe_names = [f"P{i}" for i in range(1, 25)]
        self.reservoir_names = ["R1"]
        self.tank_names = ["T1"]

        # Simple graph structure for topology
        self._adjacency = self._build_mock_topology()

        # Base values for simulation
        self._base_pressures = {j: 50.0 + np.random.uniform(-5, 5) for j in self.junction_names}
        self._base_flows = {p: 10.0 + np.random.uniform(-2, 2) for p in self.pipe_names}

        logger.warning("Using MOCK network - L-Town.inp not available")

    def _build_mock_topology(self) -> Dict[str, List[str]]:
        """Build a simple mock network topology."""
        adj = {}
        # Create a simple branching network
        for i, j in enumerate(self.junction_names):
            neighbors = []
            if i > 0:
                neighbors.append(self.junction_names[i-1])
            if i < len(self.junction_names) - 1:
                neighbors.append(self.junction_names[i+1])
            # Add some cross-connections
            if i % 5 == 0 and i + 5 < len(self.junction_names):
                neighbors.append(self.junction_names[i+5])
            adj[j] = neighbors
        return adj

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get neighboring nodes."""
        return self._adjacency.get(node_id, [])

    def simulate_step(self, time_seconds: float, leak_nodes: Dict[str, float]) -> SimulationState:
        """Simulate one timestep."""
        pressures = {}
        flows = {}
        demands = {}

        # Time-varying demand pattern (diurnal)
        hour = (time_seconds / 3600) % 24
        demand_multiplier = 1.0 + 0.3 * np.sin(2 * np.pi * (hour - 6) / 24)

        for j in self.junction_names:
            base_p = self._base_pressures[j]
            # Apply demand effect
            pressures[j] = base_p * (1.0 - 0.1 * (demand_multiplier - 1.0))
            demands[j] = 0.5 * demand_multiplier

            # Apply leak effect
            if j in leak_nodes:
                leak_rate = leak_nodes[j]
                # Pressure drop proportional to leak rate
                pressures[j] -= leak_rate * 2.0
                demands[j] += leak_rate

        for p in self.pipe_names:
            base_f = self._base_flows[p]
            flows[p] = base_f * demand_multiplier

        # Node flows = demands at nodes
        node_flows = dict(demands)

        return SimulationState(
            time_seconds=time_seconds,
            pressures=pressures,
            flows=flows,
            demands=demands,
            node_flows=node_flows
        )


class NetworkSimulator:
    """
    WNTR-based water network simulator.
    Acts as the physics engine for the digital twin.
    """

    def __init__(self, config: SimulationConfig = None):
        self.config = config or DEFAULT_CONFIG.simulation
        self.wn = None
        self.sim = None
        self._mock_network: Optional[MockNetwork] = None
        self._current_time: float = 0.0
        self._leak_nodes: Dict[str, float] = {}  # node_id -> leak rate
        self._results_cache: Optional[object] = None
        self._is_mock = False
        self._undirected_graph = None  # Cached undirected graph for topology

        self._load_network()

    def _load_network(self):
        """Load the water network model."""
        if not WNTR_AVAILABLE:
            logger.warning("WNTR not installed. Using mock network.")
            self._mock_network = MockNetwork()
            self._is_mock = True
            return

        inp_path = self.config.inp_file

        # Try multiple paths
        search_paths = [
            inp_path,
            os.path.join(os.path.dirname(__file__), "..", "..", inp_path),
            os.path.join(os.path.dirname(__file__), "..", "..", "data", inp_path),
            os.path.join(os.getcwd(), inp_path),
            os.path.join(os.getcwd(), "data", inp_path),
        ]

        for path in search_paths:
            if os.path.exists(path):
                try:
                    self.wn = wntr.network.WaterNetworkModel(path)
                    logger.info(f"Loaded network from: {path}")
                    self._setup_simulation()
                    return
                except Exception as e:
                    logger.error(f"Error loading {path}: {e}")

        # Create mock network if file not found
        logger.warning(f"L-Town.inp not found. Creating mock network for demonstration.")
        self._mock_network = MockNetwork()
        self._is_mock = True

    def _setup_simulation(self):
        """Configure the WNTR simulation."""
        if self.wn is None:
            return

        # Set simulation options
        self.wn.options.time.duration = int(self.config.simulation_duration_hours * 3600)
        self.wn.options.time.hydraulic_timestep = self.config.hydraulic_timestep_seconds
        self.wn.options.time.pattern_timestep = self.config.pattern_timestep_seconds
        self.wn.options.time.report_timestep = self.config.hydraulic_timestep_seconds

    @property
    def is_mock(self) -> bool:
        """Check if using mock network."""
        return self._is_mock

    @property
    def junction_names(self) -> List[str]:
        """Get list of junction node names."""
        if self._is_mock:
            return self._mock_network.junction_names
        return list(self.wn.junction_name_list)

    @property
    def pipe_names(self) -> List[str]:
        """Get list of pipe names."""
        if self._is_mock:
            return self._mock_network.pipe_names
        return list(self.wn.pipe_name_list)

    @property
    def reservoir_names(self) -> List[str]:
        """Get list of reservoir names."""
        if self._is_mock:
            return self._mock_network.reservoir_names
        return list(self.wn.reservoir_name_list)

    @property
    def tank_names(self) -> List[str]:
        """Get list of tank names."""
        if self._is_mock:
            return self._mock_network.tank_names
        return list(self.wn.tank_name_list)

    def get_node_neighbors(self, node_id: str, depth: int = 1) -> List[str]:
        """Get neighboring nodes up to specified depth using network topology."""
        if self._is_mock:
            neighbors = set()
            current_level = {node_id}
            for _ in range(depth):
                next_level = set()
                for n in current_level:
                    for neighbor in self._mock_network.get_neighbors(n):
                        if neighbor != node_id:
                            neighbors.add(neighbor)
                            next_level.add(neighbor)
                current_level = next_level
            return list(neighbors)

        # Use WNTR's graph representation - convert to undirected for topology traversal
        # (to_graph() returns a DiGraph based on pipe flow direction, but we need
        # undirected connectivity for neighbor discovery)
        # Cache the undirected graph for performance
        if self._undirected_graph is None:
            self._undirected_graph = self.wn.to_graph().to_undirected()
        G = self._undirected_graph
        neighbors = set()
        current_level = {node_id}

        for _ in range(depth):
            next_level = set()
            for n in current_level:
                if n in G:
                    for neighbor in G.neighbors(n):
                        if neighbor != node_id:
                            neighbors.add(neighbor)
                            next_level.add(neighbor)
            current_level = next_level

        return list(neighbors)

    def inject_leak(self, node_id: str, leak_rate: float = 5.0) -> bool:
        """
        Inject a leak at the specified node.

        Args:
            node_id: Junction node ID where leak occurs
            leak_rate: Leak rate in L/s

        Returns:
            True if leak injection successful
        """
        if self._is_mock:
            if node_id in self._mock_network.junction_names:
                self._leak_nodes[node_id] = leak_rate
                logger.info(f"MOCK: Injected leak at {node_id} with rate {leak_rate} L/s")
                return True
            return False

        if node_id not in self.wn.junction_name_list:
            logger.error(f"Node {node_id} not found in network")
            return False

        try:
            # Add leak as an emitter (models orifice-type leak)
            junction = self.wn.get_node(node_id)
            # Emitter coefficient: Q = C * P^0.5, solve for C given desired Q at nominal pressure
            nominal_pressure = 50.0  # psi
            emitter_coeff = leak_rate / (nominal_pressure ** 0.5)
            junction.emitter_coefficient = emitter_coeff

            self._leak_nodes[node_id] = leak_rate
            logger.info(f"Injected leak at {node_id} with rate ~{leak_rate} L/s")
            return True

        except Exception as e:
            logger.error(f"Failed to inject leak: {e}")
            return False

    def remove_leak(self, node_id: str) -> bool:
        """Remove a leak from the specified node."""
        if node_id in self._leak_nodes:
            del self._leak_nodes[node_id]

            if not self._is_mock and self.wn:
                try:
                    junction = self.wn.get_node(node_id)
                    junction.emitter_coefficient = 0.0
                except Exception as e:
                    logger.error(f"Failed to remove leak: {e}")
                    return False

            logger.info(f"Removed leak from {node_id}")
            return True
        return False

    def clear_all_leaks(self):
        """Remove all injected leaks."""
        for node_id in list(self._leak_nodes.keys()):
            self.remove_leak(node_id)

    def get_active_leaks(self) -> Dict[str, float]:
        """Get dictionary of active leaks."""
        return dict(self._leak_nodes)

    def run_simulation(self) -> bool:
        """Run the full hydraulic simulation."""
        if self._is_mock:
            logger.info("MOCK: Simulation ready")
            return True

        try:
            sim = wntr.sim.EpanetSimulator(self.wn)
            self._results_cache = sim.run_sim()
            logger.info("Hydraulic simulation completed")
            return True
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return False

    def get_state_at_time(self, time_seconds: float) -> SimulationState:
        """
        Get the simulation state at a specific time.

        Args:
            time_seconds: Time in seconds from simulation start

        Returns:
            SimulationState with pressures, flows, and demands
        """
        if self._is_mock:
            return self._mock_network.simulate_step(time_seconds, self._leak_nodes)

        if self._results_cache is None:
            self.run_simulation()

        results = self._results_cache

        # Find nearest timestep
        times = results.node['pressure'].index
        idx = np.abs(times - time_seconds).argmin()
        actual_time = times[idx]

        pressures = {}
        flows = {}
        demands = {}

        # Extract pressures at junctions
        for node_id in self.junction_names:
            try:
                # Convert from m to psi (1 m H2O ≈ 1.422 psi)
                pressure_m = results.node['pressure'].loc[actual_time, node_id]
                pressures[node_id] = pressure_m * 1.422
            except KeyError:
                pressures[node_id] = 0.0

        # Extract flows
        for link_id in self.pipe_names:
            try:
                # Flow in L/s
                flows[link_id] = results.link['flowrate'].loc[actual_time, link_id] * 1000
            except KeyError:
                flows[link_id] = 0.0

        # Extract demands
        for node_id in self.junction_names:
            try:
                demands[node_id] = results.node['demand'].loc[actual_time, node_id] * 1000
            except KeyError:
                demands[node_id] = 0.0

        # Also add node-level flows (demand at node = flow through node)
        # For flow sensors at junctions, we use the demand value
        node_flows = dict(demands)  # Flow at junction = demand at junction

        return SimulationState(
            time_seconds=actual_time,
            pressures=pressures,
            flows=flows,
            demands=demands,
            node_flows=node_flows
        )

    def step(self, dt_seconds: float = None) -> SimulationState:
        """
        Advance simulation by one timestep.

        Args:
            dt_seconds: Time step in seconds (default: hydraulic timestep)

        Returns:
            Current simulation state
        """
        if dt_seconds is None:
            dt_seconds = self.config.hydraulic_timestep_seconds

        self._current_time += dt_seconds
        return self.get_state_at_time(self._current_time)

    @property
    def current_time(self) -> float:
        """Current simulation time in seconds."""
        return self._current_time

    def reset(self):
        """Reset simulation to initial state."""
        self._current_time = 0.0
        self._results_cache = None
        if not self._is_mock:
            self._load_network()

    def get_sensor_locations(self) -> Tuple[List[str], List[str]]:
        """
        Parse the .inp file to extract defined sensor locations.

        Returns:
            Tuple of (pressure_sensor_nodes, amr_flow_nodes)
        """
        if self._is_mock:
            # Mock network: return subset as "sensors"
            return (
                self._mock_network.junction_names[:10],  # Pressure sensors
                self._mock_network.junction_names[::2]    # AMR (every other)
            )

        pressure_sensors = []
        amr_nodes = []

        # Find the .inp file path
        inp_path = None
        search_paths = [
            self.config.inp_file,
            os.path.join(os.path.dirname(__file__), "..", "..", self.config.inp_file),
            os.path.join(os.path.dirname(__file__), "..", "..", "data", self.config.inp_file),
            os.path.join(os.getcwd(), self.config.inp_file),
            os.path.join(os.getcwd(), "data", self.config.inp_file),
        ]

        for path in search_paths:
            if os.path.exists(path):
                inp_path = path
                break

        if not inp_path:
            logger.warning("Could not find .inp file for sensor parsing")
            return ([], [])

        try:
            with open(inp_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('['):
                        continue

                    # Check for sensor markers in comments
                    if ';PRESSURE SENSOR' in line or ';AMR & PRESSURE SENSOR' in line:
                        # Extract node ID (first column)
                        parts = line.split()
                        if parts:
                            node_id = parts[0]
                            pressure_sensors.append(node_id)

                    if ';AMR' in line:
                        parts = line.split()
                        if parts:
                            node_id = parts[0]
                            amr_nodes.append(node_id)

            logger.info(f"Found {len(pressure_sensors)} pressure sensors, {len(amr_nodes)} AMR nodes")
            return (pressure_sensors, amr_nodes)

        except Exception as e:
            logger.error(f"Error parsing .inp file for sensors: {e}")
            return ([], [])
