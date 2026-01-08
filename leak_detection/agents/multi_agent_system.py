"""
Multi-Agent System Orchestrator.

This module provides the MultiAgentSystem class that creates and
coordinates all agents in the leak detection system.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base import MessageBus
from .sensor_agent import SensorAgent
from .coordinator_agent import CoordinatorAgent
from .localizer_agent import LocalizerAgent

logger = logging.getLogger(__name__)


@dataclass
class AgentSystemConfig:
    """Configuration for the multi-agent system."""
    sensor_buffer_size: int = 30
    coordinator_aggregation_window: float = 10.0
    min_alerts_for_investigation: int = 2


class MultiAgentSystem:
    """
    Multi-Agent System for Leak Detection.
    
    This class instantiates and coordinates:
    - N SensorAgents (one per monitored node)
    - 1 CoordinatorAgent (central intelligence)
    - 1 LocalizerAgent (leak triangulation expert)
    
    Architecture:
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                      MessageBus                              │
    │  (Pub/Sub + Direct Messaging)                               │
    └──────┬──────────┬──────────┬──────────┬──────────┬─────────┘
           │          │          │          │          │
    ┌──────▼───┐ ┌────▼────┐ ┌───▼────┐ ┌───▼────┐ ┌──▼───────┐
    │ Sensor   │ │ Sensor  │ │ Sensor │ │Coordin-│ │Localizer │
    │ Agent 1  │ │ Agent 2 │ │ Agent N│ │ator    │ │Agent     │
    │(node_1)  │ │(node_2) │ │(node_N)│ │        │ │          │
    └──────────┘ └─────────┘ └────────┘ └────────┘ └──────────┘
    
    Sense→Decide→Act   Sense→Decide→Act   Aggregates    Triangulates
    Local anomaly      Local anomaly      alerts        positions
    detection          detection          Opens invs    Returns results
    ```
    
    Usage:
        mas = MultiAgentSystem(sensor_nodes=["n1", "n2", "n3"])
        mas.initialize()
        
        # Each simulation step:
        environment = {"readings": {...}, "sim_time": t}
        mas.step(environment)
        
        # Get results:
        status = mas.get_system_status()
    """
    
    def __init__(
        self,
        sensor_nodes: List[str],
        network_distances: Optional[Dict[str, Dict[str, float]]] = None,
        config: Optional[AgentSystemConfig] = None
    ):
        self.config = config or AgentSystemConfig()
        self.sensor_nodes = sensor_nodes
        self.network_distances = network_distances or {}
        
        # Create shared message bus
        self.message_bus = MessageBus()
        
        # Agents will be created in initialize()
        self.sensor_agents: Dict[str, SensorAgent] = {}
        self.coordinator: Optional[CoordinatorAgent] = None
        self.localizer: Optional[LocalizerAgent] = None
        
        self._initialized = False
        self._step_count = 0
        
        logger.info(f"MultiAgentSystem created for {len(sensor_nodes)} sensor nodes")
    
    def initialize(self) -> None:
        """Initialize all agents in the system."""
        if self._initialized:
            logger.warning("MultiAgentSystem already initialized")
            return
        
        # Create sensor agents (one per node)
        for node_id in self.sensor_nodes:
            agent_id = f"sensor_{node_id}"
            agent = SensorAgent(
                agent_id=agent_id,
                node_id=node_id,
                message_bus=self.message_bus,
                reading_type="pressure",
                buffer_size=self.config.sensor_buffer_size
            )
            self.sensor_agents[node_id] = agent
        
        # Create coordinator agent
        self.coordinator = CoordinatorAgent(
            agent_id="coordinator",
            message_bus=self.message_bus,
            sensor_ids=list(self.sensor_agents.keys())
        )
        
        # Create localizer agent
        self.localizer = LocalizerAgent(
            agent_id="localizer",
            message_bus=self.message_bus,
            sensor_nodes=self.sensor_nodes,
            network_distances=self.network_distances
        )
        
        self._initialized = True
        logger.info(
            f"MultiAgentSystem initialized: {len(self.sensor_agents)} sensors, "
            f"1 coordinator, 1 localizer"
        )
    
    def reset(self) -> None:
        """Reset all agents and system state."""
        if not self._initialized:
            return
            
        # Reset agents
        for agent in self.sensor_agents.values():
            agent.reset()
        
        if self.coordinator:
            self.coordinator.reset()
            
        if self.localizer:
            self.localizer.reset()
            
        # Clear all pending messages from the bus
        self.message_bus.reset_queues()
        self.message_bus.clear_log()
        
        logger.info("MultiAgentSystem reset complete")

    def step(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one step of the multi-agent system.
        
        This runs the sense→decide→act cycle for all agents.
        
        Args:
            environment: Dict containing:
                - readings: Dict[node_id, Dict[str, float]] with sensor values
                - sim_time: Current simulation time
                
        Returns:
            System state summary
        """
        if not self._initialized:
            raise RuntimeError("MultiAgentSystem not initialized. Call initialize() first.")
        
        self._step_count += 1
        
        # 1. Run all sensor agents (parallel in concept, sequential here)
        for node_id, agent in self.sensor_agents.items():
            agent.step(environment)
        
        # 2. Run coordinator agent (processes messages from sensors)
        self.coordinator.step(environment)
        
        # 3. Run localizer agent (processes requests from coordinator)
        self.localizer.step(environment)
        
        # 4. Let coordinator process localization results
        self.coordinator.step(environment)
        
        return self.get_system_status()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self._initialized:
            return {"error": "Not initialized"}
        
        # Aggregate sensor statuses
        sensor_statuses = {
            node_id: agent.get_status()
            for node_id, agent in self.sensor_agents.items()
        }
        
        # Count alerts
        total_alerts = sum(s["alerts_sent"] for s in sensor_statuses.values())
        sensors_alerting = sum(1 for s in sensor_statuses.values() if s["anomaly_confidence"] > 0.5)
        
        return {
            "step_count": self._step_count,
            "agent_count": len(self.sensor_agents) + 2,  # +2 for coordinator and localizer
            "sensor_count": len(self.sensor_agents),
            "total_alerts": total_alerts,
            "sensors_alerting": sensors_alerting,
            "coordinator": self.coordinator.get_status(),
            "localizer": self.localizer.get_status(),
            "active_investigations": self.coordinator.get_active_investigations(),
            "sensors": sensor_statuses
        }
    
    def get_detected_leaks(self) -> List[Dict[str, Any]]:
        """Get list of detected/localized leaks."""
        if not self._initialized:
            return []
        
        leaks = []
        for inv in self.coordinator.get_active_investigations():
            if inv["status"] == "localized" and inv["localization"]:
                loc = inv["localization"]
                leaks.append({
                    "investigation_id": inv["id"],
                    "location": loc.get("probable_location"),
                    "confidence": loc.get("confidence", 0.0),
                    "candidates": loc.get("candidates", [])
                })
        
        return leaks
    
    def get_agent_summary(self) -> str:
        """Get human-readable summary of agents."""
        if not self._initialized:
            return "System not initialized"
        
        lines = [
            "=== Multi-Agent System Summary ===",
            f"Total Agents: {len(self.sensor_agents) + 2}",
            "",
            "SENSOR AGENTS:",
        ]
        
        for node_id, agent in list(self.sensor_agents.items())[:5]:
            status = agent.get_status()
            lines.append(
                f"  • {agent.agent_id}: node={node_id}, "
                f"alerts={status['alerts_sent']}, "
                f"confidence={status['anomaly_confidence']:.2f}"
            )
        
        if len(self.sensor_agents) > 5:
            lines.append(f"  ... and {len(self.sensor_agents) - 5} more sensors")
        
        lines.extend([
            "",
            "COORDINATOR AGENT:",
            f"  • {self.coordinator.agent_id}: "
            f"mode={self.coordinator.get_status()['system_mode']}, "
            f"investigations={self.coordinator.get_status()['active_investigations']}",
            "",
            "LOCALIZER AGENT:",
            f"  • {self.localizer.agent_id}: "
            f"localizations={self.localizer.get_status()['localizations_performed']}",
        ])
        
        return "\n".join(lines)
