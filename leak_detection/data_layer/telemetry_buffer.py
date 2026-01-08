"""
Telemetry Buffer - Time-series data storage (mock database).
"""

import logging
from typing import Dict, List, Optional, Deque
from collections import deque
from dataclasses import dataclass

from ..config import DataLayerConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class TelemetryRecord:
    """A single telemetry record."""
    node_id: str
    timestamp: float
    reading_type: str  # 'pressure' or 'flow'
    raw_value: float
    filtered_value: Optional[float] = None
    z_score: Optional[float] = None
    is_valid: bool = True
    validation_error: Optional[str] = None


class TelemetryBuffer:
    """
    In-memory time-series buffer for telemetry data.
    Acts as a mock time-series database.
    """

    def __init__(self, config: DataLayerConfig = None):
        """
        Initialize the telemetry buffer.

        Args:
            config: Data layer configuration
        """
        self.config = config or DEFAULT_CONFIG.data_layer
        self._buffer_size = self.config.buffer_size

        # Separate buffers per node and reading type
        # Structure: {node_id: {reading_type: deque}}
        self._buffers: Dict[str, Dict[str, Deque[TelemetryRecord]]] = {}

        # Statistics for each node/type combination
        self._stats: Dict[str, Dict[str, Dict]] = {}

        logger.debug(f"Initialized TelemetryBuffer with size {self._buffer_size}")

    def _ensure_buffer(self, node_id: str, reading_type: str):
        """Ensure buffer exists for node/type combination."""
        if node_id not in self._buffers:
            self._buffers[node_id] = {}
            self._stats[node_id] = {}

        if reading_type not in self._buffers[node_id]:
            self._buffers[node_id][reading_type] = deque(maxlen=self._buffer_size)
            self._stats[node_id][reading_type] = {
                'count': 0,
                'sum': 0.0,
                'sum_sq': 0.0,
                'min': float('inf'),
                'max': float('-inf')
            }

    def store(self, record: TelemetryRecord):
        """
        Store a telemetry record in the buffer.

        Args:
            record: The telemetry record to store
        """
        self._ensure_buffer(record.node_id, record.reading_type)

        buffer = self._buffers[record.node_id][record.reading_type]
        stats = self._stats[record.node_id][record.reading_type]

        # Update rolling statistics (only for valid records)
        if record.is_valid and record.raw_value is not None:
            # If buffer is full, remove oldest value from stats
            if len(buffer) == self._buffer_size:
                oldest = buffer[0]
                if oldest.is_valid and oldest.raw_value is not None:
                    stats['count'] -= 1
                    stats['sum'] -= oldest.raw_value
                    stats['sum_sq'] -= oldest.raw_value ** 2

            # Add new value to stats
            stats['count'] += 1
            stats['sum'] += record.raw_value
            stats['sum_sq'] += record.raw_value ** 2
            stats['min'] = min(stats['min'], record.raw_value)
            stats['max'] = max(stats['max'], record.raw_value)

        buffer.append(record)

    def get_recent(
        self,
        node_id: str,
        reading_type: str,
        n: int = 10
    ) -> List[TelemetryRecord]:
        """
        Get the n most recent records for a node/type.

        Args:
            node_id: Node identifier
            reading_type: 'pressure' or 'flow'
            n: Number of records to retrieve

        Returns:
            List of recent records (newest first)
        """
        self._ensure_buffer(node_id, reading_type)
        buffer = self._buffers[node_id][reading_type]

        # Return newest n records
        records = list(buffer)[-n:]
        records.reverse()  # Newest first
        return records

    def get_statistics(
        self,
        node_id: str,
        reading_type: str
    ) -> Optional[Dict]:
        """
        Get statistics for a node/type combination.

        Returns:
            Dictionary with count, mean, std, min, max
        """
        self._ensure_buffer(node_id, reading_type)
        stats = self._stats[node_id][reading_type]

        if stats['count'] == 0:
            return None

        count = stats['count']
        mean = stats['sum'] / count

        # Variance using Welford's online algorithm formula
        variance = (stats['sum_sq'] / count) - (mean ** 2)
        std = variance ** 0.5 if variance > 0 else 0.0

        return {
            'count': count,
            'mean': mean,
            'std': std,
            'min': stats['min'],
            'max': stats['max']
        }

    def get_windowed_statistics(
        self,
        node_id: str,
        reading_type: str,
        window_size: int = None
    ) -> Optional[Dict]:
        """
        Get statistics over a sliding window.

        Args:
            node_id: Node identifier
            reading_type: 'pressure' or 'flow'
            window_size: Number of recent samples (default: zscore_window from config)

        Returns:
            Dictionary with count, mean, std
        """
        window_size = window_size or self.config.zscore_window
        records = self.get_recent(node_id, reading_type, window_size)

        # Filter valid records
        valid_values = [
            r.raw_value for r in records
            if r.is_valid and r.raw_value is not None
        ]

        if len(valid_values) < 2:
            return None

        import numpy as np
        values = np.array(valid_values)

        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    def get_all_nodes(self) -> List[str]:
        """Get list of all nodes with data."""
        return list(self._buffers.keys())

    def get_node_types(self, node_id: str) -> List[str]:
        """Get reading types available for a node."""
        if node_id in self._buffers:
            return list(self._buffers[node_id].keys())
        return []

    def clear(self, node_id: str = None, reading_type: str = None):
        """
        Clear buffer(s).

        Args:
            node_id: Specific node to clear (None = all)
            reading_type: Specific type to clear (None = all)
        """
        if node_id is None:
            self._buffers.clear()
            self._stats.clear()
        elif reading_type is None and node_id in self._buffers:
            del self._buffers[node_id]
            del self._stats[node_id]
        elif node_id in self._buffers and reading_type in self._buffers[node_id]:
            del self._buffers[node_id][reading_type]
            del self._stats[node_id][reading_type]

    def get_buffer_size(self, node_id: str, reading_type: str) -> int:
        """Get current number of records in a buffer."""
        self._ensure_buffer(node_id, reading_type)
        return len(self._buffers[node_id][reading_type])
