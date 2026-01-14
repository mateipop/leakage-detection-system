
from dataclasses import dataclass, field
from typing import List
from enum import Enum, auto

class SamplingMode(Enum):
    ECO = auto()       # Low power, low frequency
    HIGH_RES = auto()  # High power, high frequency

class DeviceState(Enum):
    IDLE = auto()           # Low power standby
    CALCULATING = auto()    # Physics calculation
    SAMPLING = auto()       # High power sampling

class SystemStatus(Enum):
    NORMAL = auto()
    ALERT = auto()
    INVESTIGATING = auto()
    LEAK_CONFIRMED = auto()

@dataclass
class SimulationConfig:
    inp_file: str = "L-TOWN.inp"
    simulation_duration_hours: float = 24.0
    hydraulic_timestep_seconds: int = 300  # 5 minutes
    pattern_timestep_seconds: int = 300
    noise_std_pressure: float = 0.5  # psi
    noise_std_flow: float = 0.02  # L/s
    eco_sampling_interval: float = 60.0  # seconds
    highres_sampling_interval: float = 1.0  # seconds

@dataclass
class DataLayerConfig:
    buffer_size: int = 1000
    moving_average_window: int = 3  # Smaller window for faster response
    zscore_window: int = 30  # Reduced for faster baseline establishment
    validation_min_pressure: float = -5000.0  # Very permissive
    validation_max_pressure: float = 10000.0  # Very permissive
    validation_min_flow: float = -50000.0  # L/s (negative = reverse flow)
    validation_max_flow: float = 50000.0  # L/s

@dataclass
class AIConfig:
    anomaly_threshold: float = 0.8  # Raised to reduce false positives (requiring higher confidence)
    confidence_decay: float = 0.9  # Faster response to changes
    min_samples_for_detection: int = 5  # Faster confirmation
    pressure_drop_threshold: float = 2.0  # Standard deviations (Raised from 1.5)
    topology_search_depth: int = 3

@dataclass
class TUIConfig:
    refresh_rate_ms: int = 500
    log_buffer_size: int = 100
    chart_history_points: int = 60

@dataclass
class SystemConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    data_layer: DataLayerConfig = field(default_factory=DataLayerConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    tui: TUIConfig = field(default_factory=TUIConfig)

    monitored_junctions: List[str] = field(default_factory=list)
    monitored_pipes: List[str] = field(default_factory=list)

DEFAULT_CONFIG = SystemConfig()