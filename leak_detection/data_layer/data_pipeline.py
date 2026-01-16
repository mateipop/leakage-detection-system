
import logging
import math
from typing import Optional, Callable, List, Dict
from dataclasses import dataclass, asdict
from collections import deque

import numpy as np

from ..config import DataLayerConfig, DEFAULT_CONFIG
from ..simulation.device_simulator import SensorReading
from .telemetry_buffer import TelemetryBuffer, TelemetryRecord
from .redis_manager import RedisManager

logger = logging.getLogger(__name__)

@dataclass
class PipelineEvent:
    event_type: str  # 'validation_error', 'filter_applied', 'anomaly_flagged'
    node_id: str
    timestamp: float
    message: str
    severity: str = 'info'  # 'info', 'warning', 'error'

class DataPipeline:

    def __init__(
        self,
        config: DataLayerConfig = None,
        event_callback: Callable[[PipelineEvent], None] = None
    ):
        self.config = config or DEFAULT_CONFIG.data_layer
        self._event_callback = event_callback

        self._buffer = TelemetryBuffer(self.config)

        self._filter_windows: Dict[str, Dict[str, deque]] = {}

        self._processed_count = 0
        self._valid_count = 0
        self._invalid_count = 0
        
        self._baseline_snapshot: Dict[tuple, Dict[str, float]] = {}
        self._baseline_locked = False
        self._baseline_samples_needed = 20  # Samples before locking baseline
        
        # Differential memory to store offsets from original baseline
        # Key: (node_id, reading_type) -> offset_value
        self._baseline_offsets: Dict[tuple, float] = {}

        self._redis = RedisManager()
        self._redis.connect()

        logger.info("DataPipeline initialized")

    def _emit_event(
        self,
        event_type: str,
        node_id: str,
        timestamp: float,
        message: str,
        severity: str = 'info'
    ):
        event = PipelineEvent(
            event_type=event_type,
            node_id=node_id,
            timestamp=timestamp,
            message=message,
            severity=severity
        )
        logger.log(
            logging.WARNING if severity == 'error' else logging.DEBUG,
            f"Pipeline [{event_type}] {node_id}: {message}"
        )
        if self._event_callback:
            self._event_callback(event)

    def _validate(self, reading: SensorReading) -> tuple[bool, Optional[str]]:
        if reading.value is None or math.isnan(reading.value):
            return False, "NaN value detected"

        if math.isinf(reading.value):
            return False, "Infinite value detected"

        if reading.reading_type == 'pressure':
            if reading.value < self.config.validation_min_pressure:
                return False, f"Pressure below minimum ({reading.value:.2f} < {self.config.validation_min_pressure})"
            if reading.value > self.config.validation_max_pressure:
                return False, f"Pressure above maximum ({reading.value:.2f} > {self.config.validation_max_pressure})"
        elif reading.reading_type == 'flow':
            if reading.value < self.config.validation_min_flow:
                return False, f"Flow below minimum ({reading.value:.2f} < {self.config.validation_min_flow})"
            if reading.value > self.config.validation_max_flow:
                return False, f"Flow above maximum ({reading.value:.2f} > {self.config.validation_max_flow})"

        if reading.is_corrupted:
            return False, "Device flagged reading as corrupted"

        return True, None

    def _apply_moving_average(
        self,
        node_id: str,
        reading_type: str,
        value: float
    ) -> float:
        if node_id not in self._filter_windows:
            self._filter_windows[node_id] = {}
        if reading_type not in self._filter_windows[node_id]:
            self._filter_windows[node_id][reading_type] = deque(
                maxlen=self.config.moving_average_window
            )

        window = self._filter_windows[node_id][reading_type]
        window.append(value)

        return sum(window) / len(window)

    def lock_baseline(self):
        self._baseline_locked = True
        logger.info(f"Baseline statistics locked ({len(self._baseline_snapshot)} sensors captured)")

    def recalibrate_baseline(self):
        recalibrated_count = 0
        
        current_state = {}
        for node_id in self._filter_windows:
            if node_id not in current_state: current_state[node_id] = {}
            for reading_type in self._filter_windows[node_id]:
                stats = self._buffer.get_windowed_statistics(node_id, reading_type, window_size=5)
                
                if stats and stats['count'] > 0:
                    current_state[node_id][reading_type] = {
                        'mean': stats['mean'],
                        'std': max(stats['std'], 0.5) # Preserve existing volatility or min 0.5
                    }

        self._filter_windows.clear()
        self._buffer.clear()
        
        # Differential Memory: Retain offsets
        if self._baseline_locked and self._baseline_snapshot:
             for node_id, readings in current_state.items():
                for reading_type, state in readings.items():
                    key = (node_id, reading_type)
                    if key in self._baseline_snapshot:
                        original_mean = self._baseline_snapshot[key]['mean']
                        new_mean = state['mean']
                        # Calculate drift/offset from original baseline
                        current_offset = self._baseline_offsets.get(key, 0.0)
                        additional_offset = original_mean - new_mean
                        self._baseline_offsets[key] = current_offset + additional_offset

        seed_count = self.config.moving_average_window
        
        for node_id, readings in current_state.items():
            for reading_type, state in readings.items():
                mean = state['mean']
                std = state['std']
                
                ts = 0.0 
                for i in range(seed_count):
                    sign = 1 if i % 2 == 0 else -1
                    noisy_val = mean + (sign * std)
                    
                    rec = TelemetryRecord(
                        node_id=node_id,
                        timestamp=ts,
                        reading_type=reading_type,
                        raw_value=noisy_val,
                        filtered_value=noisy_val,
                        z_score=0.0,
                        is_valid=True
                    )
                    self._buffer.store(rec)
                    self._apply_moving_average(node_id, reading_type, noisy_val)

        self._baseline_locked = False
        self._baseline_snapshot.clear() # Clear snapshot to force rolling window usage
        
        logger.info(f"System Reset: Baselines unlocked, buffers cleared. Adapting to new state.")

        self._baseline_locked = True # Force lock usage of new snapshot
        logger.info(f"Baseline recalibrated for {recalibrated_count} sensors. System re-zeroed.")

    def get_baseline_offset(self, node_id: str, reading_type: str) -> float:
        """Returns the cumulative baseline offset (drift) for this sensor due to recalibrations."""
        return self._baseline_offsets.get((node_id, reading_type), 0.0)

    def _calculate_zscore(
        self,
        node_id: str,
        reading_type: str,
        value: float
    ) -> Optional[float]:
        key = (node_id, reading_type)
        
        if self._baseline_locked and key in self._baseline_snapshot:
            baseline = self._baseline_snapshot[key]
            if baseline['std'] < 1e-6:
                return 0.0
            z = (value - baseline['mean']) / baseline['std']
            return z
        
        stats = self._buffer.get_windowed_statistics(
            node_id,
            reading_type,
            self.config.zscore_window
        )

        if stats is None or stats['count'] < self._baseline_samples_needed:
            return None
        
        if not self._baseline_locked and stats['count'] >= self._baseline_samples_needed:
            if key not in self._baseline_snapshot:
                self._baseline_snapshot[key] = {
                    'mean': stats['mean'],
                    'std': max(stats['std'], 0.5)  # Enforce min std of 0.5 to prevent noise amplification
                }
                logger.debug(f"Baseline captured for {node_id}/{reading_type}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

        if stats['std'] < 1e-6:
            return 0.0

        z_score = (value - stats['mean']) / stats['std']
        return z_score

    def process(self, reading: SensorReading) -> Optional[TelemetryRecord]:
        self._processed_count += 1

        is_valid, error_msg = self._validate(reading)

        record = TelemetryRecord(
            node_id=reading.node_id,
            timestamp=reading.timestamp,
            reading_type=reading.reading_type,
            raw_value=reading.value if is_valid else None,
            is_valid=is_valid,
            validation_error=error_msg
        )

        if not is_valid:
            self._invalid_count += 1
            self._emit_event(
                'validation_error',
                reading.node_id,
                reading.timestamp,
                f"Invalid data discarded: {error_msg}",
                'warning'
            )
            self._buffer.store(record)
            return None

        self._valid_count += 1

        filtered_value = self._apply_moving_average(
            reading.node_id,
            reading.reading_type,
            reading.value
        )
        record.filtered_value = filtered_value

        value_for_zscore = reading.value if self._baseline_locked else filtered_value
        z_score = self._calculate_zscore(
            reading.node_id,
            reading.reading_type,
            value_for_zscore
        )
        record.z_score = z_score

        self._buffer.store(record)
        
        # Store in Redis (or fallback)
        self._redis.store_reading(record.node_id, asdict(record))

        self._emit_event(
            'data_processed',
            reading.node_id,
            reading.timestamp,
            f"{reading.reading_type}: raw={reading.value:.2f}, filtered={filtered_value:.2f}, z={z_score:.2f}" if z_score else f"{reading.reading_type}: raw={reading.value:.2f}, filtered={filtered_value:.2f}",
            'info'
        )

        return record

    def process_batch(
        self,
        readings: List[SensorReading]
    ) -> List[TelemetryRecord]:
        results = []
        for reading in readings:
            record = self.process(reading)
            if record is not None:
                results.append(record)
        return results

    def get_latest_record(
        self,
        node_id: str,
        reading_type: str
    ) -> Optional[TelemetryRecord]:
        records = self._buffer.get_recent(node_id, reading_type, 1)
        return records[0] if records else None

    def get_node_statistics(
        self,
        node_id: str,
        reading_type: str
    ) -> Optional[Dict]:
        return self._buffer.get_windowed_statistics(node_id, reading_type)

    def get_all_latest_zscores(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for node_id in self._buffer.get_all_nodes():
            result[node_id] = {}
            for reading_type in self._buffer.get_node_types(node_id):
                records = self._buffer.get_recent(node_id, reading_type, 1)
                if records and records[0].z_score is not None:
                    result[node_id][reading_type] = records[0].z_score
        return result

    @property
    def buffer(self) -> TelemetryBuffer:
        return self._buffer

    @property
    def statistics(self) -> Dict:
        return {
            'processed': self._processed_count,
            'valid': self._valid_count,
            'invalid': self._invalid_count,
            'validity_rate': (
                self._valid_count / self._processed_count
                if self._processed_count > 0 else 0.0
            )
        }

    def lock_baseline(self):
        if self._baseline_snapshot:
            self._baseline_locked = True
            logger.info(f"Baseline locked with {len(self._baseline_snapshot)} sensor baselines")
        else:
            logger.warning("Cannot lock baseline - no baseline data captured yet")
    
    def unlock_baseline(self):
        self._baseline_locked = False
        logger.info("Baseline unlocked")
    
    def clear_baseline(self):
        self._baseline_snapshot.clear()
        self._baseline_locked = False
        logger.info("Baseline cleared")

    def reset(self):
        self._buffer.clear()
        self._filter_windows.clear()
        self._processed_count = 0
        self._valid_count = 0
        self._invalid_count = 0
        self._baseline_snapshot.clear()
        self._baseline_locked = False