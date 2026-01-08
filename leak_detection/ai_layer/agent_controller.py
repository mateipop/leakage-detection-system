"""
Agent Controller - Orchestrates the AI monitoring loop and executes decisions.
"""

import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from ..config import AIConfig, SamplingMode, SystemStatus, DEFAULT_CONFIG
from ..simulation.network_simulator import NetworkSimulator
from ..simulation.device_simulator import DeviceFleet
from ..data_layer.data_pipeline import DataPipeline, PipelineEvent
from ..data_layer.feature_extractor import FeatureExtractor, FeatureVector
from .inference_engine import InferenceEngine, InferenceResult
from .decision_engine import DecisionEngine, Decision, ActionType
from .leak_localizer import LeakLocalizer, LocalizationResult, MultiLeakResult

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent operational state."""
    IDLE = auto()
    MONITORING = auto()
    ALERT = auto()
    INVESTIGATING = auto()
    CONFIRMED = auto()


@dataclass
class AgentStatus:
    """Current agent status for monitoring."""
    state: AgentState
    system_status: SystemStatus
    sampling_mode: SamplingMode
    current_decision: Optional[Decision]
    top_anomaly: Optional[InferenceResult]
    monitored_nodes: int
    alerts_triggered: int
    leaks_detected: int
    samples_processed: int


@dataclass
class MonitoringCycleResult:
    """Result of one monitoring cycle."""
    timestamp: float
    features: Dict[str, FeatureVector]
    inference_results: Dict[str, InferenceResult]
    decision: Decision
    agent_status: AgentStatus


class AgentController:
    """
    Central controller for the AI monitoring agent.

    Orchestrates the full monitoring loop:
    1. Collect sensor data
    2. Process through data pipeline
    3. Extract features
    4. Run inference
    5. Make decisions
    6. Execute actions (change sampling rates, etc.)
    """

    def __init__(
        self,
        network: NetworkSimulator,
        fleet: DeviceFleet,
        config: AIConfig = None,
        event_callback: Callable[[str], None] = None
    ):
        """
        Initialize the agent controller.

        Args:
            network: Network simulator
            fleet: Device fleet
            config: AI configuration
            event_callback: Function called with status messages
        """
        self.config = config or DEFAULT_CONFIG.ai
        self._network = network
        self._fleet = fleet
        self._event_callback = event_callback

        # Initialize components
        self._pipeline = DataPipeline(
            event_callback=self._on_pipeline_event
        )
        self._feature_extractor = FeatureExtractor(self._pipeline)
        self._inference_engine = InferenceEngine(self.config)
        self._decision_engine = DecisionEngine(self.config)

        # Initialize leak localizer with network topology
        sensored_nodes = set(fleet.monitored_nodes)
        self._leak_localizer = LeakLocalizer(
            get_neighbors_fn=network.get_node_neighbors,
            get_all_nodes_fn=lambda: network.junction_names,
            sensored_nodes=sensored_nodes
        )

        # State
        self._state = AgentState.IDLE
        self._current_mode = SamplingMode.ECO
        self._last_decision: Optional[Decision] = None
        self._last_inference: Dict[str, InferenceResult] = {}

        # Statistics
        self._alerts_triggered = 0
        self._leaks_detected = 0
        self._samples_processed = 0

        logger.info("AgentController initialized")

    def _emit_event(self, message: str):
        """Emit an event message."""
        logger.info(f"Agent: {message}")
        if self._event_callback:
            self._event_callback(message)

    def _on_pipeline_event(self, event: PipelineEvent):
        """Handle pipeline events."""
        if event.severity == 'warning' or event.severity == 'error':
            self._emit_event(f"[Pipeline] {event.message}")

    def _topology_fn(self, node_id: str) -> List[str]:
        """Get neighboring nodes for a given node."""
        return self._network.get_node_neighbors(
            node_id,
            depth=self.config.topology_search_depth
        )

    def _collect_readings(self, sim_time: float) -> int:
        """
        Collect sensor readings from all devices that should sample.

        Returns:
            Number of readings collected
        """
        readings_count = 0
        state = self._network.get_state_at_time(sim_time)

        for device_id in self._fleet.device_ids:
            device = self._fleet.get_device(device_id)
            if not device:
                continue

            if not device.should_sample(sim_time):
                continue

            # Get true value from simulation
            if device.reading_type == 'pressure':
                true_value = state.pressures.get(device.node_id, 0.0)
            else:
                true_value = state.flows.get(device.node_id, 0.0)

            # Take noisy reading
            reading = device.sample(true_value, sim_time, add_noise=True)

            # Process through pipeline
            record = self._pipeline.process(reading)
            if record:
                readings_count += 1

        self._samples_processed += readings_count
        return readings_count

    def _execute_decision(self, decision: Decision):
        """Execute the action from a decision."""
        if decision.action == ActionType.MAINTAIN_ECO:
            pass  # No change

        elif decision.action == ActionType.SWITCH_ECO:
            self._fleet.set_all_sampling_mode(SamplingMode.ECO)
            self._current_mode = SamplingMode.ECO
            self._state = AgentState.MONITORING
            self._emit_event("Switched all sensors to ECO mode")

        elif decision.action == ActionType.SWITCH_HIGH_RES:
            self._fleet.set_node_sampling_mode(
                decision.target_nodes,
                SamplingMode.HIGH_RES
            )
            self._current_mode = SamplingMode.HIGH_RES
            self._state = AgentState.INVESTIGATING
            self._emit_event(
                f"Switched {len(decision.target_nodes)} sensors to HIGH-RES mode"
            )

        elif decision.action == ActionType.SWITCH_ALL_HIGH_RES:
            self._fleet.set_all_sampling_mode(SamplingMode.HIGH_RES)
            self._current_mode = SamplingMode.HIGH_RES
            self._state = AgentState.ALERT
            self._emit_event("ALERT: Switched ALL sensors to HIGH-RES mode")

        elif decision.action == ActionType.ALERT:
            self._alerts_triggered += 1
            self._state = AgentState.ALERT
            self._emit_event(f"ALERT: {decision.reason}")

        elif decision.action == ActionType.CONFIRM_LEAK:
            self._leaks_detected += 1
            self._state = AgentState.CONFIRMED
            self._emit_event(
                f"LEAK CONFIRMED at {decision.estimated_leak_location} "
                f"(confidence: {decision.confidence:.1%})"
            )

    def run_cycle(self, sim_time: float) -> MonitoringCycleResult:
        """
        Run one complete monitoring cycle.

        Args:
            sim_time: Current simulation time

        Returns:
            MonitoringCycleResult with all intermediate data
        """
        # Update state to monitoring if idle
        if self._state == AgentState.IDLE:
            self._state = AgentState.MONITORING

        # Step 1: Collect readings
        readings_count = self._collect_readings(sim_time)

        # Step 2: Extract features for all monitored nodes
        monitored_nodes = self._fleet.monitored_nodes
        features = self._feature_extractor.extract_all_features(
            monitored_nodes,
            sim_time,
            topology_fn=self._topology_fn
        )

        # Step 3: Run inference
        inference_results = self._inference_engine.infer_batch(features)
        self._last_inference = inference_results

        # Step 4: Make decision
        decision = self._decision_engine.make_decision(
            inference_results,
            sim_time,
            topology_fn=self._topology_fn
        )
        self._last_decision = decision

        # Step 5: Execute decision
        self._execute_decision(decision)

        # Track alerts
        if decision.action in [ActionType.ALERT, ActionType.SWITCH_HIGH_RES]:
            if self._decision_engine.status == SystemStatus.ALERT:
                self._alerts_triggered += 1

        # Build status
        anomalies = self._inference_engine.get_anomalies(inference_results)
        top_anomaly = anomalies[0] if anomalies else None

        status = AgentStatus(
            state=self._state,
            system_status=self._decision_engine.status,
            sampling_mode=self._current_mode,
            current_decision=decision,
            top_anomaly=top_anomaly,
            monitored_nodes=len(monitored_nodes),
            alerts_triggered=self._alerts_triggered,
            leaks_detected=self._leaks_detected,
            samples_processed=self._samples_processed
        )

        return MonitoringCycleResult(
            timestamp=sim_time,
            features=features,
            inference_results=inference_results,
            decision=decision,
            agent_status=status
        )

    def get_status(self) -> AgentStatus:
        """Get current agent status."""
        anomalies = self._inference_engine.get_anomalies(self._last_inference)

        return AgentStatus(
            state=self._state,
            system_status=self._decision_engine.status,
            sampling_mode=self._current_mode,
            current_decision=self._last_decision,
            top_anomaly=anomalies[0] if anomalies else None,
            monitored_nodes=len(self._fleet.monitored_nodes),
            alerts_triggered=self._alerts_triggered,
            leaks_detected=self._leaks_detected,
            samples_processed=self._samples_processed
        )

    def get_latest_inference(self) -> Dict[str, InferenceResult]:
        """Get the latest inference results."""
        return dict(self._last_inference)

    def get_estimated_leak_location(self) -> Optional[str]:
        """Get the AI's current estimate of leak location."""
        if not self._last_inference:
            return None

        # Use the leak localizer for better topology-aware estimation
        localization = self._leak_localizer.localize(
            self._last_inference,
            self.config.anomaly_threshold
        )

        if localization:
            return localization.estimated_node

        # Fall back to decision engine's estimate
        if self._last_decision and self._last_decision.estimated_leak_location:
            return self._last_decision.estimated_leak_location

        return None

    def get_localization_result(self) -> Optional[LocalizationResult]:
        """Get detailed localization result including confidence and method."""
        if not self._last_inference:
            return None

        return self._leak_localizer.localize(
            self._last_inference,
            self.config.anomaly_threshold
        )

    def get_multi_leak_result(self, max_leaks: int = 5) -> Optional[MultiLeakResult]:
        """
        Get multi-leak detection result.
        
        Returns localization results for all detected leaks,
        clustered by network topology.
        
        Args:
            max_leaks: Maximum number of leaks to report (default: 5)
        """
        if not self._last_inference:
            return None

        return self._leak_localizer.localize_multiple(
            self._last_inference,
            self.config.anomaly_threshold,
            max_leaks=max_leaks
        )

    def get_all_leak_locations(self) -> List[str]:
        """
        Get estimated locations for all detected leaks.
        
        Returns:
            List of node IDs where leaks are estimated
        """
        result = self.get_multi_leak_result()
        if not result:
            return []
        return [loc.estimated_node for loc in result.localizations]

    @property
    def pipeline(self) -> DataPipeline:
        """Access to the data pipeline."""
        return self._pipeline

    @property
    def inference_engine(self) -> InferenceEngine:
        """Access to the inference engine."""
        return self._inference_engine

    @property
    def decision_engine(self) -> DecisionEngine:
        """Access to the decision engine."""
        return self._decision_engine

    def reset(self):
        """Reset the agent to initial state."""
        self._state = AgentState.IDLE
        self._current_mode = SamplingMode.ECO
        self._last_decision = None
        self._last_inference.clear()
        self._alerts_triggered = 0
        self._leaks_detected = 0
        self._samples_processed = 0

        self._pipeline.reset()
        self._inference_engine.reset_history()
        self._decision_engine.reset()
        self._fleet.reset_all()
