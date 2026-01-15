
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
    sensor_buffer_size: int = 30
    coordinator_aggregation_window: float = 10.0
    min_alerts_for_investigation: int = 2

class MultiAgentSystem:
    
    def __init__(
        self,
        sensor_nodes: List[str],
        network_distances: Optional[Dict[str, Dict[str, float]]] = None,
        config: Optional[AgentSystemConfig] = None,
        sensor_neighbors: Dict[str, List[str]] = None, # NEW: Neighbor map
        candidate_nodes: List[str] = None,
        candidate_distances: Dict[str, Dict[str, float]] = None,
        sensor_types: Dict[str, str] = None
    ):
        self.config = config or AgentSystemConfig()
        self.sensor_nodes = sensor_nodes
        self.network_distances = network_distances or {}
        self.sensor_neighbors = sensor_neighbors or {} # Store neighbor map
        self.candidate_nodes = candidate_nodes or sensor_nodes
        self.candidate_distances = candidate_distances or {}
        self.sensor_types = sensor_types or {}
        
        self.message_bus = MessageBus()
        
        self.sensor_agents: Dict[str, SensorAgent] = {}
        self.coordinator: Optional[CoordinatorAgent] = None
        self.localizer: Optional[LocalizerAgent] = None
        
        self._initialized = False
        self._step_count = 0
        
        logger.info(f"MultiAgentSystem created for {len(sensor_nodes)} sensor nodes")
    
    def initialize(self) -> None:
        if self._initialized:
            logger.warning("MultiAgentSystem already initialized")
            return
        
        for node_id in self.sensor_nodes:
            agent_id = f"sensor_{node_id}"
            
            my_neighbors = self.sensor_neighbors.get(node_id, [])
            reading_type = self.sensor_types.get(node_id, "pressure")
            
            agent = SensorAgent(
                agent_id=agent_id,
                node_id=node_id,
                message_bus=self.message_bus,
                reading_type=reading_type,
                buffer_size=self.config.sensor_buffer_size,
                neighbor_ids=my_neighbors
            )
            self.sensor_agents[node_id] = agent
        
        self.coordinator = CoordinatorAgent(
            agent_id="coordinator",
            message_bus=self.message_bus,
            sensor_ids=list(self.sensor_agents.keys())
        )
        
        self.localizer = LocalizerAgent(
            agent_id="localizer",
            message_bus=self.message_bus,
            sensor_nodes=self.sensor_nodes,
            network_distances=self.network_distances,
            candidate_nodes=self.candidate_nodes,
            candidate_distances=self.candidate_distances
        )
        
        self._initialized = True
        logger.info(
            f"MultiAgentSystem initialized: {len(self.sensor_agents)} sensors, "
            f"1 coordinator, 1 localizer"
        )
    
    def reset(self) -> None:
        if not self._initialized:
            return
            
        for agent in self.sensor_agents.values():
            agent.reset()
        
        if self.coordinator:
            self.coordinator.reset()
            
        if self.localizer:
            self.localizer.reset()
            
        self.message_bus.reset_queues()
        self.message_bus.clear_log()
        
        logger.info("MultiAgentSystem reset complete")

    def step(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("MultiAgentSystem not initialized. Call initialize() first.")
        
        self._step_count += 1
        
        for node_id, agent in self.sensor_agents.items():
            agent.step(environment)
        
        self.coordinator.step(environment)
        
        self.localizer.step(environment)
        
        self.coordinator.step(environment)
        
        return self.get_system_status()
    
    def get_system_status(self) -> Dict[str, Any]:
        if not self._initialized:
            return {"error": "Not initialized"}
            
        sensor_status = {}
        for node_id, agent in self.sensor_agents.items():
            st = agent.get_status()
            # Flatten or extract key metrics for the summary
            sensor_status[node_id] = {
                "reading": st.get("last_reading"),
                "zscore": st.get("last_zscore"),
                "confidence": st.get("confidence", 0.0),
                "mode": st.get("mode")
            }
            
        coord_status = {}
        active_invs_list = []
        if self.coordinator:
            coord_status = self.coordinator.get_status()
            active_invs_list = [
                {
                    "id": k, 
                    "status": v.status,
                    "localization": v.localization_result
                }
                for k, v in self.coordinator._active_investigations.items()
            ]

        return {
            "step": self._step_count, 
            "agent_count": len(self.sensor_agents) + (2 if self.coordinator else 0),
            "sensor_count": len(self.sensor_agents),
            "sensors": sensor_status,
            "coordinator": coord_status,
            "active_investigations": active_invs_list
        }

    def get_detected_leaks(self) -> List[Dict[str, Any]]:
        if not self.localizer:
            return []
            
        return self.localizer.get_results()

    def recalibrate(self):
        if not self._initialized:
            return
            
        for agent in self.sensor_agents.values():
            agent.recalibrate()
            
        self.coordinator._recent_anomalies.clear()
        self.coordinator._active_investigations.clear()
        self.coordinator._system_mode = "NORMAL"
        
        logger.info("Multi-Agent System recalibrated to new baseline")
        
        sensor_statuses = {
            node_id: agent.get_status()
            for node_id, agent in self.sensor_agents.items()
        }
        
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