
import logging
import time
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..config import DeviceState, SamplingMode, SimulationConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    device_id: str
    node_id: str
    timestamp: float
    reading_type: str  # 'pressure' or 'flow'
    value: float
    unit: str
    is_corrupted: bool = False

@dataclass
class DeviceStatus:
    device_id: str
    node_id: str
    state: DeviceState
    sampling_mode: SamplingMode
    battery_level: float  # 0.0 to 1.0
    power_consumption_mw: float
    samples_collected: int
    last_sample_time: Optional[float]

class DeviceSimulator:

    POWER_IDLE = 0.5
    POWER_CALCULATING = 10.0
    POWER_SAMPLING = 50.0

    BATTERY_CAPACITY = 5000.0

    def __init__(
        self,
        device_id: str,
        node_id: str,
        reading_type: str = 'pressure',
        config: SimulationConfig = None
    ):
        self.device_id = device_id
        self.node_id = node_id
        self.reading_type = reading_type
        self.config = config or DEFAULT_CONFIG.simulation

        self._state = DeviceState.IDLE
        self._sampling_mode = SamplingMode.ECO
        self._battery_mwh = self.BATTERY_CAPACITY
        self._samples_collected = 0
        self._last_sample_time: Optional[float] = None
        self._last_state_change: float = 0.0

        if reading_type == 'pressure':
            self._noise_std = self.config.noise_std_pressure
            self._unit = 'psi'
        else:
            self._noise_std = self.config.noise_std_flow
            self._unit = 'L/s'

        logger.debug(f"Created device {device_id} monitoring {reading_type} at {node_id}")

    @property
    def state(self) -> DeviceState:
        return self._state

    @property
    def sampling_mode(self) -> SamplingMode:
        return self._sampling_mode

    @property
    def battery_level(self) -> float:
        return self._battery_mwh / self.BATTERY_CAPACITY

    @property
    def sampling_interval(self) -> float:
        if self._sampling_mode == SamplingMode.ECO:
            return self.config.eco_sampling_interval
        return self.config.highres_sampling_interval

    @property
    def is_dead(self) -> bool:
        return self._battery_mwh <= 0

    def get_status(self) -> DeviceStatus:
        power = {
            DeviceState.IDLE: self.POWER_IDLE,
            DeviceState.CALCULATING: self.POWER_CALCULATING,
            DeviceState.SAMPLING: self.POWER_SAMPLING
        }[self._state]

        return DeviceStatus(
            device_id=self.device_id,
            node_id=self.node_id,
            state=self._state,
            sampling_mode=self._sampling_mode,
            battery_level=self.battery_level,
            power_consumption_mw=power,
            samples_collected=self._samples_collected,
            last_sample_time=self._last_sample_time
        )

    def set_sampling_mode(self, mode: SamplingMode):
        if mode != self._sampling_mode:
            logger.info(f"Device {self.device_id}: Switching from {self._sampling_mode.name} to {mode.name}")
            self._sampling_mode = mode

    def _consume_power(self, duration_seconds: float):
        power_mw = {
            DeviceState.IDLE: self.POWER_IDLE,
            DeviceState.CALCULATING: self.POWER_CALCULATING,
            DeviceState.SAMPLING: self.POWER_SAMPLING
        }[self._state]

        consumed = power_mw * (duration_seconds / 3600.0)
        self._battery_mwh = max(0.0, self._battery_mwh - consumed)

    def _transition_state(self, new_state: DeviceState, current_time: float):
        duration = current_time - self._last_state_change
        if duration < 0: duration = 0
        
        self._consume_power(duration)

        self._state = new_state
        self._last_state_change = current_time

    def sample(
        self,
        true_value: float,
        sim_time: float,
        add_noise: bool = True,
        corrupt_probability: float = 0.0  # Disabled - set > 0 to test error handling
    ) -> Optional[SensorReading]:
        if self.is_dead:
            return None

        # Account for IDLE time since last sample/change
        if self._state == DeviceState.IDLE:
            self._transition_state(DeviceState.SAMPLING, sim_time)

        # Explicitly consume energy for the sampling/transmission action
        # Simulating ~2 seconds of high-power activity (sensor warm-up + measurement + radio TX)
        # 500mW * 2s
        active_energy_mwh = 500.0 * (2.0 / 3600.0)
        self._battery_mwh = max(0.0, self._battery_mwh - active_energy_mwh)

        is_corrupted = corrupt_probability > 0 and np.random.random() < corrupt_probability

        if is_corrupted:
            corruption_type = np.random.choice(['nan', 'extreme', 'stuck'])
            if corruption_type == 'nan':
                value = float('nan')
            elif corruption_type == 'extreme':
                value = true_value * np.random.uniform(10, 100) * np.random.choice([-1, 1])
            else:  # stuck
                value = 0.0
        else:
            if add_noise:
                relative_noise = abs(true_value) * 0.005
                noise_std = max(self._noise_std, relative_noise)
                value = true_value + np.random.normal(0, noise_std)
            else:
                value = true_value

        reading = SensorReading(
            device_id=self.device_id,
            node_id=self.node_id,
            timestamp=sim_time,
            reading_type=self.reading_type,
            value=value,
            unit=self._unit,
            is_corrupted=is_corrupted
        )

        self._samples_collected += 1
        self._last_sample_time = sim_time

        # Return to IDLE state
        self._transition_state(DeviceState.IDLE, sim_time)

        return reading

    def should_sample(self, current_time: float) -> bool:
        if self._last_sample_time is None:
            return True

        elapsed = current_time - self._last_sample_time
        return elapsed >= self.sampling_interval

    def reset(self):
        self._state = DeviceState.IDLE
        self._sampling_mode = SamplingMode.ECO
        self._battery_mwh = self.BATTERY_CAPACITY
        self._samples_collected = 0
        self._last_sample_time = None
        self._last_state_change = 0.0

class DeviceFleet:

    def __init__(self, config: SimulationConfig = None):
        self.config = config or DEFAULT_CONFIG.simulation
        self._devices: Dict[str, DeviceSimulator] = {}
        self._node_to_devices: Dict[str, List[str]] = {}

    def add_device(
        self,
        device_id: str,
        node_id: str,
        reading_type: str = 'pressure'
    ) -> DeviceSimulator:
        device = DeviceSimulator(device_id, node_id, reading_type, self.config)
        self._devices[device_id] = device

        if node_id not in self._node_to_devices:
            self._node_to_devices[node_id] = []
        self._node_to_devices[node_id].append(device_id)

        return device

    def get_device(self, device_id: str) -> Optional[DeviceSimulator]:
        return self._devices.get(device_id)

    def get_devices_at_node(self, node_id: str) -> List[DeviceSimulator]:
        device_ids = self._node_to_devices.get(node_id, [])
        return [self._devices[did] for did in device_ids]

    def set_all_sampling_mode(self, mode: SamplingMode):
        for device in self._devices.values():
            device.set_sampling_mode(mode)

    def set_node_sampling_mode(self, node_ids: List[str], mode: SamplingMode):
        for node_id in node_ids:
            for device in self.get_devices_at_node(node_id):
                device.set_sampling_mode(mode)

    def get_all_statuses(self) -> Dict[str, DeviceStatus]:
        return {did: dev.get_status() for did, dev in self._devices.items()}

    @property
    def device_ids(self) -> List[str]:
        return list(self._devices.keys())

    @property
    def monitored_nodes(self) -> List[str]:
        return list(self._node_to_devices.keys())

    def reset_all(self):
        for device in self._devices.values():
            device.reset()