
import os
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import networkx as nx

try:
    import wntr
    WNTR_AVAILABLE = True
except ImportError:
    WNTR_AVAILABLE = False

from ..config import SimulationConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class SimulationState:
    time_seconds: float
    pressures: Dict[str, float]  # node_id -> pressure (psi)
    flows: Dict[str, float]      # link_id -> flow (L/s)
    demands: Dict[str, float]    # node_id -> demand (L/s)
    node_flows: Dict[str, float] = None  # node_id -> flow at node (L/s)

class MockNetwork:

    def __init__(self):
        self.junction_names = [f"J{i}" for i in range(1, 21)]
        self.pipe_names = [f"P{i}" for i in range(1, 25)]
        self.reservoir_names = ["R1"]
        self.tank_names = ["T1"]

        self._adjacency = self._build_mock_topology()

        self._base_pressures = {j: 50.0 + np.random.uniform(-5, 5) for j in self.junction_names}
        self._base_flows = {p: 10.0 + np.random.uniform(-2, 2) for p in self.pipe_names}

        logger.warning("Using MOCK network - L-Town.inp not available")

    def _build_mock_topology(self) -> Dict[str, List[str]]:
        adj = {}
        for i, j in enumerate(self.junction_names):
            neighbors = []
            if i > 0:
                neighbors.append(self.junction_names[i-1])
            if i < len(self.junction_names) - 1:
                neighbors.append(self.junction_names[i+1])
            if i % 5 == 0 and i + 5 < len(self.junction_names):
                neighbors.append(self.junction_names[i+5])
            adj[j] = neighbors
        return adj

    def get_neighbors(self, node_id: str) -> List[str]:
        return self._adjacency.get(node_id, [])

    def simulate_step(self, time_seconds: float, leak_nodes: Dict[str, float]) -> SimulationState:
        pressures = {}
        flows = {}
        demands = {}

        hour = (time_seconds / 3600) % 24
        demand_multiplier = 1.0 + 0.3 * np.sin(2 * np.pi * (hour - 6) / 24)

        for j in self.junction_names:
            base_p = self._base_pressures[j]
            pressures[j] = base_p * (1.0 - 0.1 * (demand_multiplier - 1.0))
            demands[j] = 0.5 * demand_multiplier

            if j in leak_nodes:
                leak_rate = leak_nodes[j]
                pressures[j] -= leak_rate * 2.0
                demands[j] += leak_rate

        for p in self.pipe_names:
            base_f = self._base_flows[p]
            flows[p] = base_f * demand_multiplier

        node_flows = dict(demands)

        return SimulationState(
            time_seconds=time_seconds,
            pressures=pressures,
            flows=flows,
            demands=demands,
            node_flows=node_flows
        )

class NetworkSimulator:

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
        self._weighted_graph = None # Cached weighted graph for distance

        self._load_network()

    def _load_network(self):
        if not WNTR_AVAILABLE:
            logger.warning("WNTR not installed. Using mock network.")
            self._mock_network = MockNetwork()
            self._is_mock = True
            return

        inp_path = self.config.inp_file

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

        logger.warning(f"L-Town.inp not found. Creating mock network for demonstration.")
        self._mock_network = MockNetwork()
        self._is_mock = True

    def _setup_simulation(self):
        if self.wn is None:
            return

        self.wn.options.time.duration = int(self.config.simulation_duration_hours * 3600)
        self.wn.options.time.hydraulic_timestep = self.config.hydraulic_timestep_seconds
        self.wn.options.time.pattern_timestep = self.config.pattern_timestep_seconds
        self.wn.options.time.report_timestep = self.config.hydraulic_timestep_seconds

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    @property
    def junction_names(self) -> List[str]:
        if self._is_mock:
            return self._mock_network.junction_names
        return list(self.wn.junction_name_list)

    @property
    def pipe_names(self) -> List[str]:
        if self._is_mock:
            return self._mock_network.pipe_names
        return list(self.wn.pipe_name_list)

    @property
    def reservoir_names(self) -> List[str]:
        if self._is_mock:
            return self._mock_network.reservoir_names
        return list(self.wn.reservoir_name_list)

    @property
    def tank_names(self) -> List[str]:
        if self._is_mock:
            return self._mock_network.tank_names
        return list(self.wn.tank_name_list)

    def get_node_neighbors(self, node_id: str, depth: int = 1) -> List[str]:
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

    def calculate_shortest_path_distance(self, node_start: str, node_end: str) -> int:
        if node_start == node_end:
            return 0
            
        if self._is_mock:
            visited = {node_start}
            queue = [(node_start, 0)]
            while queue:
                current, dist = queue.pop(0)
                if current == node_end:
                    return dist
                
                for neighbor in self._mock_network.get_neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            return 99 # Not found/disconnected

        if self._undirected_graph is None:
            self._undirected_graph = self.wn.to_graph().to_undirected()
        
        if node_start not in self._undirected_graph:
            logger.warning(f"Start node {node_start} NOT in graph")
        if node_end not in self._undirected_graph:
            logger.warning(f"End node {node_end} NOT in graph")

        try:
            return nx.shortest_path_length(self._undirected_graph, source=node_start, target=node_end)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            logger.warning(f"Path not found or node missing between {node_start} and {node_end}")
            return 99
        except Exception as e:
            logger.error(f"Unexpected error calculating path {node_start}->{node_end}: {e}")
            return 99

    def calculate_topological_distance(self, node_start: str, node_end: str) -> float:
        """Calculates physical distance in meters between two nodes."""
        if node_start == node_end:
            return 0.0
            
        if self._is_mock:
            # Mock has no lengths, assume 100m per hop
            hops = self.calculate_shortest_path_distance(node_start, node_end)
            return float(hops * 100.0)

        if self._weighted_graph is None:
            G = self.wn.to_graph()
            # Add weights based on pipe length
            for u, v, key, data in G.edges(keys=True, data=True):
                try:
                    link = self.wn.get_link(key)
                    # Use length or default 50m
                    data['weight'] = max(getattr(link, 'length', 50.0), 0.1)
                except:
                    data['weight'] = 50.0
            self._weighted_graph = G.to_undirected()
        
        try:
            return nx.shortest_path_length(self._weighted_graph, source=node_start, target=node_end, weight='weight')
        except:
             return 9999.0

    def inject_leak(self, node_id: str, leak_rate: float = 5.0) -> bool:
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
            junction = self.wn.get_node(node_id)
            nominal_pressure_psi = 50.0
            
            leak_rate_cms = leak_rate / 1000.0
            nominal_pressure_m = nominal_pressure_psi * 0.70307
            
            emitter_coeff = leak_rate_cms / (nominal_pressure_m ** 0.5)
            
            junction.emitter_coefficient = emitter_coeff

            self._leak_nodes[node_id] = leak_rate
            logger.info(f"Injected leak at {node_id} with rate ~{leak_rate} L/s (C={emitter_coeff:.6f})")
            return True

        except Exception as e:
            logger.error(f"Failed to inject leak: {e}")
            return False

    def remove_leak(self, node_id: str) -> bool:
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
        for node_id in list(self._leak_nodes.keys()):
            self.remove_leak(node_id)

    def get_active_leaks(self) -> Dict[str, float]:
        return dict(self._leak_nodes)

    def run_simulation(self) -> bool:
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
        if self._is_mock:
            return self._mock_network.simulate_step(time_seconds, self._leak_nodes)

        if self._results_cache is None:
            self.run_simulation()

        results = self._results_cache

        times = results.node['pressure'].index
        idx = np.abs(times - time_seconds).argmin()
        actual_time = times[idx]

        pressures = {}
        flows = {}
        demands = {}

        node_flows = {}

        for node_id in self.junction_names:
            pressure_m = 0.0
            try:
                pressure_m = results.node['pressure'].loc[actual_time, node_id]
                pressures[node_id] = pressure_m * 1.422
            except KeyError:
                pressures[node_id] = 0.0

            demand_lps = 0.0
            try:
                demand_lps = results.node['demand'].loc[actual_time, node_id] * 1000
            except KeyError:
                demand_lps = 0.0
            demands[node_id] = demand_lps
            
            # Add leakage to node_flow if present
            # WNTR 'demand' does not include emitter leakage
            leak_flow = 0.0
            if node_id in self._leak_nodes and pressure_m > 0 and self.wn:
                try:
                    node = self.wn.get_node(node_id)
                    if hasattr(node, 'emitter_coefficient') and node.emitter_coefficient:
                        # Q = C * P^0.5 (SI units: m3/s, m)
                        q_cms = node.emitter_coefficient * (pressure_m ** 0.5)
                        leak_flow = q_cms * 1000.0
                except Exception:
                    pass
            
            node_flows[node_id] = demand_lps + leak_flow

        return SimulationState(
            time_seconds=actual_time,
            pressures=pressures,
            flows=flows,
            demands=demands,
            node_flows=node_flows
        )

    def step(self, dt_seconds: float = None) -> SimulationState:
        if dt_seconds is None:
            dt_seconds = self.config.hydraulic_timestep_seconds

        self._current_time += dt_seconds
        return self.get_state_at_time(self._current_time)

    @property
    def current_time(self) -> float:
        return self._current_time

    def reset(self):
        self._current_time = 0.0
        self._results_cache = None
        if not self._is_mock:
            self._load_network()

    def get_sensor_locations(self) -> Tuple[List[str], List[str]]:
        if self._is_mock:
            return (
                self._mock_network.junction_names[:10],  # Pressure sensors
                self._mock_network.junction_names[::2]    # AMR (every other)
            )

        pressure_sensors = []
        amr_nodes = []

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

                    if ';PRESSURE SENSOR' in line or ';AMR & PRESSURE SENSOR' in line:
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