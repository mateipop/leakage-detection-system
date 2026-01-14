
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
import queue
import threading

logger = logging.getLogger(__name__)

class MessageType(Enum):
    ANOMALY_ALERT = auto()          # Sensor detected anomaly
    READING_UPDATE = auto()         # Regular sensor reading
    STATUS_REPORT = auto()          # Agent status update
    
    MODE_CHANGE = auto()            # Change sampling mode
    REQUEST_DATA = auto()           # Request immediate reading
    
    LOCALIZE_REQUEST = auto()       # Request leak localization
    ANOMALY_CLUSTER = auto()        # Cluster of anomalies to analyze
    
    LOCALIZATION_RESULT = auto()    # Leak location estimate
    
    PEER_GOSSIP = auto()            # Exchange data with neighbors
    PEER_CONFIRMATION = auto()      # Request confirmation from neighbor
    
    SYSTEM_ALERT = auto()           # System-wide alert
    HEARTBEAT = auto()              # Agent alive signal

@dataclass
class Message:
    msg_type: MessageType
    sender_id: str
    recipient_id: str  # Use "*" for broadcast
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    priority: int = 0  # Higher = more urgent
    
    def __lt__(self, other):
        return self.priority > other.priority  # Higher priority first

class MessageBus:
    
    def __init__(self):
        self._queues: Dict[str, queue.PriorityQueue] = {}
        self._subscribers: Dict[MessageType, List[str]] = defaultdict(list)
        self._lock = threading.Lock()
        self._message_log: List[Message] = []
        self._log_enabled = True
        
    def register_agent(self, agent_id: str):
        with self._lock:
            if agent_id not in self._queues:
                self._queues[agent_id] = queue.PriorityQueue()
                logger.debug(f"MessageBus: Registered agent '{agent_id}'")
    
    def unregister_agent(self, agent_id: str):
        with self._lock:
            if agent_id in self._queues:
                del self._queues[agent_id]
                for subscribers in self._subscribers.values():
                    if agent_id in subscribers:
                        subscribers.remove(agent_id)
    
    def subscribe(self, agent_id: str, msg_type: MessageType):
        with self._lock:
            if agent_id not in self._subscribers[msg_type]:
                self._subscribers[msg_type].append(agent_id)
    
    def publish(self, message: Message):
        with self._lock:
            if self._log_enabled:
                self._message_log.append(message)
            
            if message.recipient_id == "*":
                recipients = self._subscribers.get(message.msg_type, [])
                for recipient in recipients:
                    if recipient in self._queues and recipient != message.sender_id:
                        self._queues[recipient].put(message)
            else:
                if message.recipient_id in self._queues:
                    self._queues[message.recipient_id].put(message)
                else:
                    logger.warning(f"MessageBus: Unknown recipient '{message.recipient_id}'")
    
    def get_messages(self, agent_id: str, max_messages: int = 100) -> List[Message]:
        messages = []
        if agent_id in self._queues:
            q = self._queues[agent_id]
            while not q.empty() and len(messages) < max_messages:
                try:
                    messages.append(q.get_nowait())
                except queue.Empty:
                    break
        return messages
    
    def get_message_count(self, agent_id: str) -> int:
        if agent_id in self._queues:
            return self._queues[agent_id].qsize()
        return 0
    
    def get_message_log(self, limit: int = 100) -> List[Message]:
        with self._lock:
            return list(self._message_log[-limit:])
    
    def clear_log(self):
        with self._lock:
            self._message_log.clear()

    def reset_queues(self):
        with self._lock:
            for q in self._queues.values():
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

class Agent(ABC):
    
    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self._bus = message_bus
        self._bus.register_agent(agent_id)
        self._running = False
        self._state: Dict[str, Any] = {}
        
        logger.info(f"Agent '{agent_id}' initialized")
    
    def send_message(
        self,
        msg_type: MessageType,
        recipient: str,
        payload: Dict[str, Any] = None,
        timestamp: float = 0.0,
        priority: int = 0
    ):
        message = Message(
            msg_type=msg_type,
            sender_id=self.agent_id,
            recipient_id=recipient,
            payload=payload or {},
            timestamp=timestamp,
            priority=priority
        )
        self._bus.publish(message)
    
    def broadcast(
        self,
        msg_type: MessageType,
        payload: Dict[str, Any] = None,
        timestamp: float = 0.0,
        priority: int = 0
    ):
        self.send_message(msg_type, "*", payload, timestamp, priority)
    
    def receive_messages(self) -> List[Message]:
        return self._bus.get_messages(self.agent_id)
    
    def receive_message(self) -> Optional[Message]:
        messages = self._bus.get_messages(self.agent_id)
        return messages[0] if messages else None
    
    def subscribe(self, msg_type: MessageType):
        self._bus.subscribe(self.agent_id, msg_type)
    
    @abstractmethod
    def sense(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def decide(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def act(self, actions: Dict[str, Any]) -> None:
        pass
    
    def step(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        messages = self.receive_messages()
        self._handle_messages(messages)
        
        observations = self.sense(environment)
        actions = self.decide(observations)
        self.act(actions)
        
        return {
            "agent_id": self.agent_id,
            "observations": observations,
            "actions": actions,
            "messages_processed": len(messages)
        }
    
    def _handle_messages(self, messages: List[Message]):
        for message in messages:
            self.on_message(message)
    
    def on_message(self, message: Message):
        logger.debug(f"Agent '{self.agent_id}' received {message.msg_type.name} from '{message.sender_id}'")
    
    def shutdown(self):
        self._bus.unregister_agent(self.agent_id)
        logger.info(f"Agent '{self.agent_id}' shut down")

    def reset(self):
        self._state.clear()
        