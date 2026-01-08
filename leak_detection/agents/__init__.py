"""
Multi-Agent System for Leak Detection.

This module implements a distributed multi-agent architecture where:
- SensorAgents: Autonomous IoT devices with local anomaly detection
- CoordinatorAgent: Aggregates alerts and coordinates investigation
- LocalizerAgent: Specializes in triangulating leak locations
- MultiAgentSystem: Orchestrates all agents together

Agents communicate via an asynchronous message-passing system.
"""

from .base import Agent, Message, MessageBus, MessageType
from .sensor_agent import SensorAgent
from .coordinator_agent import CoordinatorAgent
from .localizer_agent import LocalizerAgent
from .multi_agent_system import MultiAgentSystem, AgentSystemConfig

__all__ = [
    "Agent",
    "Message",
    "MessageBus",
    "MessageType",
    "SensorAgent",
    "CoordinatorAgent",
    "LocalizerAgent",
    "MultiAgentSystem",
    "AgentSystemConfig",
]
