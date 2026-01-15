
import logging
import random
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

from .config import SystemConfig, SamplingMode, DEFAULT_CONFIG
from .simulation import NetworkSimulator, DeviceSimulator, LeakInjector
from .simulation.device_simulator import DeviceFleet
from .data_layer import DataPipeline
from .agents import MultiAgentSystem, AgentSystemConfig

logger = logging.getLogger(__name__)

@dataclass
class OrchestrationResult:
    sim_time: float
    metrics: Dict[str, Dict]
    ground_truth: List[str]
    estimated_location: Optional[str]
    status: Any
    anomaly: Any
    agent_summary: Optional[Dict] = None
    detected_leaks: Optional[List[Dict]] = None

class SystemOrchestrator:

    def __init__(
        self,
        config: SystemConfig = None,
        event_callback: Callable[[str], None] = None,
        use_multi_agent: bool = True  # Kept for compatibility, but ignored (always True)
    ):
        self.config = config or DEFAULT_CONFIG
        self._event_callback = event_callback
        
        logger.info("Initializing system components...")

        self._network = NetworkSimulator(self.config.simulation)

        self._fleet = DeviceFleet(self.config.simulation)
        self._setup_devices()

        self._data_pipeline = DataPipeline(self.config.data_layer)

        self._leak_injector = LeakInjector(self._network)

        self._multi_agent_system: Optional[MultiAgentSystem] = None
        self._setup_multi_agent_system()
            
        self._masking_offsets: Dict[str, Dict[str, float]] = {}
        
        self._sim_time = 0.0
        self._time_step = self.config.simulation.hydraulic_timestep_seconds
        self._is_running = False

        if not self._network.is_mock:
            self._network.run_simulation()

        logger.info(f"System initialized with {len(self._fleet.device_ids)} devices")
        logger.info(f"Network: {'MOCK' if self._network.is_mock else 'WNTR'}")
        logger.info(f"Multi-Agent System: ENABLED ({self._multi_agent_system.sensor_agents.__len__()} sensors + coordinator + localizer)")
    
    def _compute_candidate_distances(self, sensors: List[str]) -> Dict[str, Dict[str, float]]:
        distances = {} # candidate -> {sensor -> dist}
        
        if hasattr(self._network, 'junction_names'):
            all_nodes = self._network.junction_names
        else:
            all_nodes = self._fleet.monitored_nodes # Fallback

        if hasattr(self._network, 'wn') and self._network.wn is not None:
            try:
                import networkx as nx
                G = self._network.wn.to_graph()
                G_un = G.to_undirected()
                
                # Check coverage
                logger.info(f"Computing distances from {len(sensors)} sensors to {len(all_nodes)} candidates...")
                
                # Initialize dicts
                for node in all_nodes:
                    distances[node] = {}

                for sensor in sensors:
                    if sensor in G_un:
                        # Get hop count to all nodes
                        lengths = nx.single_source_shortest_path_length(G_un, sensor)
                        for target, hops in lengths.items():
                            if target in distances:
                                distances[target][sensor] = float(hops * 100.0) # Approx 100m per pipe
            except Exception as e:
                logger.warning(f"Could not compute candidate distances: {e}")
        
        return distances

    def _setup_multi_agent_system(self):
        # Determine sensor types
        sensor_config = {}
        for node in self._fleet.monitored_nodes:
            devices = self._fleet.get_devices_at_node(node)
            # Prefer pressure if available, else flow
            if any(d.reading_type == 'pressure' for d in devices):
                sensor_config[node] = 'pressure'
            elif any(d.reading_type == 'flow' for d in devices):
                sensor_config[node] = 'flow'

        sensor_nodes = list(sensor_config.keys())
        
        # FAIRNESS UPDATE: Allow searching all nodes
        candidate_nodes = self._network.junction_names
        
        distances = self._compute_sensor_distances()
        candidate_dists = self._compute_candidate_distances(sensor_nodes)
        
        mas_config = AgentSystemConfig(
            sensor_buffer_size=30,
            coordinator_aggregation_window=10.0,
            min_alerts_for_investigation=2
        )
        
        self._multi_agent_system = MultiAgentSystem(
            sensor_nodes=sensor_nodes,
            network_distances=distances,
            config=mas_config,
            sensor_types=sensor_config,
            candidate_nodes=candidate_nodes,
            candidate_distances=candidate_dists
        )
        self._multi_agent_system.initialize()
        
        self._emit_event(f"[cyan]Multi-Agent System initialized: {len(sensor_nodes)} sensors, {len(candidate_nodes)} candidates[/cyan]")
    
    def _compute_sensor_distances(self) -> Dict[str, Dict[str, float]]:
        distances = {}
        nodes = list(self._fleet.monitored_nodes)
        
        if hasattr(self._network, '_wn') and self._network._wn is not None:
            try:
                import networkx as nx
                G = self._network._wn.to_graph()
                
                for i, node_a in enumerate(nodes):
                    distances[node_a] = {}
                    for node_b in nodes:
                        if node_a != node_b:
                            try:
                                path_len = nx.shortest_path_length(G.to_undirected(), node_a, node_b)
                                distances[node_a][node_b] = float(path_len * 100)  # Scale to meters
                            except (nx.NetworkXNoPath, nx.NodeNotFound):
                                distances[node_a][node_b] = 1000.0  # Default far distance
            except Exception as e:
                logger.warning(f"Could not compute network distances: {e}")
        
        return distances

    def _setup_devices(self):
        pressure_sensor_nodes, amr_nodes = self._network.get_sensor_locations()

        if not pressure_sensor_nodes:
            junctions = self._network.junction_names
            pressure_sensor_nodes = junctions[:20]
            amr_nodes = junctions[::5]
            logger.warning("No sensor locations found in .inp file, using fallback")

        for i, node_id in enumerate(pressure_sensor_nodes):
            self._fleet.add_device(
                device_id=f"PRESSURE_{i:04d}",
                node_id=node_id,
                reading_type='pressure'
            )

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
        if self._event_callback:
            self._event_callback(message)

    def inject_random_leak(self, at_sensor: bool = False) -> Optional[str]:
        
        if at_sensor:
            pressure_sensor_nodes, _ = self._network.get_sensor_locations()
            candidates = list(pressure_sensor_nodes)
            if not candidates:
                candidates = self._fleet.monitored_nodes
            if not candidates:
                logger.warning("No sensor nodes available for leak injection")
                return None
        else:
            candidates = self._network.junction_names

        node_id = random.choice(candidates)

        leak_rate = random.uniform(3.0, 8.0)

        event = self._leak_injector.inject_leak(node_id, leak_rate, self._sim_time)

        if event:
            has_pressure = node_id in self._network.get_sensor_locations()[0]
            sensor_status = "[PRESSURE SENSOR]" if has_pressure else "[NO PRESSURE SENSOR]"
            self._emit_event(f"[red]LEAK INJECTED at {node_id} ({leak_rate:.1f} L/s) {sensor_status}[/red]")

            if not self._network.is_mock:
                self._network.run_simulation()

            return node_id

        return None

    def clear_all_leaks(self):
        self._leak_injector.remove_all_leaks(self._sim_time)
        self._emit_event("[green]All leaks cleared[/green]")
        
        if self._multi_agent_system:
             self._multi_agent_system.reset()
        
        if not self._network.is_mock:
            self._network.run_simulation()

    def get_ground_truth(self) -> List[str]:
        return self._leak_injector.get_ground_truth()

    def get_detection_summary(self) -> str:
        return self._leak_injector.get_detection_summary()

    def record_detection(self, estimated_node: str):
        ground_truth = self.get_ground_truth()
        if not ground_truth:
            return
        
        best_leak = None
        best_distance = 999
        
        for actual_node in ground_truth:
            active_leaks = self._leak_injector.get_active_leaks()
            if actual_node in active_leaks and active_leaks[actual_node].detected:
                continue  # Skip already-detected leaks
            
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
    
    def confirm_leak(self, node_id: str) -> bool:
        pressure_signature = {}
        
        if self._multi_agent_system:
            self._multi_agent_system.recalibrate()

        confirmed = self._leak_injector.confirm_leak(
            node_id, 
            pressure_signature, 
            self._sim_time
        )
        
        if confirmed:
            self._emit_event(f"[cyan][CONFIRMED] Leak at {node_id} confirmed and masked (Matched Ground Truth)[/cyan]")
            return True
        else:
            self._emit_event(f"[yellow][MISSED] Leak at {node_id} masked (Missed Ground Truth)[/yellow]")
            return False
    
    def auto_confirm_detections(self, min_cycles: int = 5):
        active_leaks = self._leak_injector.get_active_leaks()
        
        for node_id, event in active_leaks.items():
            if event.detected and not event.confirmed:
                self.confirm_leak(node_id)
    
    def get_confirmed_leaks(self) -> List[str]:
        return self._leak_injector.get_confirmed_leaks()
    
    def get_unconfirmed_detections(self) -> List[str]:
        active_leaks = self._leak_injector.get_active_leaks()
        return [
            node_id for node_id, event in active_leaks.items()
            if event.detected and not event.confirmed
        ]
    
    def _calculate_detection_distance(self, actual_node: str, estimated_node: str) -> int:
        if actual_node == estimated_node:
            return 0
        
        for depth in range(1, 10):
            neighbors = self._network.get_node_neighbors(actual_node, depth=depth)
            if estimated_node in neighbors:
                return depth
        
        return 99

    def step(self) -> OrchestrationResult:
        self._sim_time += self._time_step

        state = self._network.get_state_at_time(self._sim_time)

        # Raw readings from "physical" simulation
        raw_readings = {}
        for node_id in self._fleet.monitored_nodes:
            node_data = {}
            # Retrieve devices to know what to measure
            devices = self._fleet.get_devices_at_node(node_id)
            for d in devices:
                if d.reading_type == 'pressure':
                    node_data['pressure'] = state.pressures.get(node_id, 0)
                elif d.reading_type == 'flow':
                    node_data['flow'] = state.node_flows.get(node_id, 0)
            
            if node_data:
                raw_readings[node_id] = node_data
        
        # Ingest and process through Pipeline/Redis/Buffer
        # This transforms Sim Data -> Telemetry Data
        environment = self._process_via_pipeline(raw_readings, self._sim_time)
        
        if self._multi_agent_system:
            agent_summary = self._multi_agent_system.step(environment)
            detected_leaks = self._multi_agent_system.get_detected_leaks()
            
            coord_status = agent_summary.get("coordinator", {})
            if coord_status.get("system_mode") == "INVESTIGATING":
                active_invs = agent_summary.get("active_investigations", [])
                for inv in active_invs:
                    if inv["status"] == "localized" and inv.get("localization"):
                        loc = inv["localization"]
                        node = loc.get("probable_location", "unknown")
                        conf = loc.get("confidence", 0)
                        if conf > 0.8:
                            self._emit_event(f"[yellow][LOCALIZED] Multi-Agent: Leak localized at {node} ({conf:.0%} confidence)[/yellow]")

        metrics = {}
        for node_id in self._fleet.monitored_nodes:
            pressure = state.pressures.get(node_id, 0)
            flow = state.node_flows.get(node_id, 0) if state.node_flows else state.demands.get(node_id, 0)

            zscore = None
            confidence = 0.0
            if agent_summary and "sensors" in agent_summary:
                sensor_data = agent_summary["sensors"].get(node_id)
                if sensor_data:
                    zscore = sensor_data.get("zscore")
                    confidence = sensor_data.get("confidence", 0.0)

            metrics[node_id] = {
                'pressure': pressure,
                'flow': flow,
                'pressure_zscore': None,
                'flow_zscore': None,
                'confidence': confidence
            }
            
            if zscore is not None:
                # Determine if this is a pressure or flow z-score based on what the agent monitors
                # Logic: If node has pressure sensor, agent monitors pressure. Else flow.
                devices = self._fleet.get_devices_at_node(node_id)
                if any(d.reading_type == 'pressure' for d in devices):
                     metrics[node_id]['pressure_zscore'] = zscore
                else:
                     metrics[node_id]['flow_zscore'] = zscore

        estimated_location = None
        if detected_leaks:
            best_leak = max(detected_leaks, key=lambda x: x.get('confidence', 0))
            if best_leak.get('confidence', 0) > 0.5:
                estimated_location = best_leak['location']
                self.record_detection(estimated_location)
        
        # Inject Pipeline Stats into agent_summary so dashboard can see them
        if agent_summary is None:
            agent_summary = {}
        
        pipeline_stats = self._data_pipeline.statistics
        # Merge stats directly or add as sub-dict. 
        # Dashboard looks for samples_processed in self._status which comes from ... STATUS?
        # Wait, the dashboard code says: samples_processed = self._status.samples_processed
        # self._status is OrchestrationResult.status which is None in current step()
        
        # Let's populate status in OrchestrationResult with a dummy object or dictionary that has these attributes
        @dataclass
        class StepStatus:
            system_status: Any
            sampling_mode: Any
            monitored_nodes: int
            samples_processed: int
            alerts_triggered: int

        from .config import SystemStatus as SysStatus, SamplingMode as SampMode
        
        current_sys_status = SysStatus.NORMAL
        if agent_summary.get("coordinator", {}).get("system_mode") == "INVESTIGATING":
            current_sys_status = SysStatus.INVESTIGATING
            
        current_mode = SampMode.ECO # Default
        if current_sys_status == SysStatus.INVESTIGATING:
            current_mode = SampMode.HIGH_RES
            
        status_obj = StepStatus(
            system_status=current_sys_status,
            sampling_mode=current_mode,
            monitored_nodes=len(self._fleet.monitored_nodes),
            samples_processed=pipeline_stats.get('processed', 0),
            alerts_triggered=agent_summary.get("coordinator", {}).get("total_alerts", 0)
        )

        return OrchestrationResult(
            sim_time=self._sim_time,
            metrics=metrics,
            ground_truth=self.get_ground_truth(),
            estimated_location=estimated_location,
            status=status_obj, 
            anomaly=None, 
            agent_summary=agent_summary,
            detected_leaks=detected_leaks
        )

    def _process_via_pipeline(self, raw_readings: Dict[str, Dict[str, float]], sim_time: float) -> Dict[str, Any]:
        """
        Pass readings through DataPipeline -> Redis -> Buffer -> return filtered env
        """
        from .simulation.device_simulator import SensorReading
        
        sensor_readings = []
        for node_id, data in raw_readings.items():
            if 'pressure' in data:
                sensor_readings.append(SensorReading(
                    device_id=f"P_{node_id}",
                    node_id=node_id,
                    reading_type="pressure",
                    value=data['pressure'],
                    timestamp=sim_time,
                    unit="m"
                ))
            if 'flow' in data:
                sensor_readings.append(SensorReading(
                    device_id=f"F_{node_id}",
                    node_id=node_id,
                    reading_type="flow",
                    value=data['flow'],
                    timestamp=sim_time,
                    unit="L/s"
                ))
        
        # 1. Ingest into Pipeline (validates, cleans, pushes to Redis)
        processed_records = self._data_pipeline.process_batch(sensor_readings)
        
        # 2. Re-construct environment from Pipeline output (this mimics agents reading from queue)
        pipeline_readings = {}
        for record in processed_records:
             # Use filtered value if available (it came from buffer/pipeline)
             val = record.filtered_value if record.filtered_value is not None else record.raw_value
             if record.node_id not in pipeline_readings:
                 pipeline_readings[record.node_id] = {}
             pipeline_readings[record.node_id][record.reading_type] = val

        return {
            "readings": pipeline_readings,
            "sim_time": sim_time
        }

    def reset(self):
        self._sim_time = 0.0
        self._leak_injector.reset()
        self._network.reset()
        
        if self._multi_agent_system:
            self._multi_agent_system.reset()

        if not self._network.is_mock:
            self._network.run_simulation()

        self._emit_event("[yellow]System reset complete[/yellow]")

    @property
    def sim_time(self) -> float:
        return self._sim_time

    @property
    def network(self) -> NetworkSimulator:
        return self._network

    @property
    def multi_agent_system(self) -> Optional[MultiAgentSystem]:
        return self._multi_agent_system

    @property
    def leak_injector(self) -> LeakInjector:
        return self._leak_injector