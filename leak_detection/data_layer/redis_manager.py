
import logging
import json
import redis
from typing import Optional, Dict, List, Any, Union
from dataclasses import asdict

from ..config import SystemConfig

logger = logging.getLogger(__name__)

class RedisManager:
    
    _instance = None
    
    def __new__(cls, host: str = 'localhost', port: int = 6379, db: int = 0):
        if cls._instance is None:
            cls._instance = super(RedisManager, cls).__new__(cls)
            cls._instance.host = host
            cls._instance.port = port
            cls._instance.db = db
            cls._instance.client = None
            cls._instance.connected = False
            cls._instance._in_memory_streams = {}
            cls._instance._in_memory_lists = {}
        return cls._instance

    def connect(self) -> bool:
        # Fast raw socket check to avoid redis-py hanging or retrying
        import socket
        try:
            sock = socket.create_connection((self.host, self.port), timeout=0.2)
            sock.close()
        except OSError:
             logger.warning(f"No Redis found at {self.host}:{self.port} (Socket refused). Using in-memory fallback.")
             self.connected = False
             if not hasattr(self, '_in_memory_streams'): self._in_memory_streams = {}
             if not hasattr(self, '_in_memory_lists'): self._in_memory_lists = {}
             return False

        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True,
                socket_connect_timeout=0.5,
                socket_timeout=0.5
            )
            self.client.ping()
            self.connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True
        except (redis.ConnectionError, redis.TimeoutError, Exception) as e:
            logger.warning(f"Using in-memory Redis fallback (connection failed: {e})")
            self.connected = False
            if not hasattr(self, '_in_memory_streams'): self._in_memory_streams = {}
            if not hasattr(self, '_in_memory_lists'): self._in_memory_lists = {}
            return False

    def store_reading(self, node_id: str, reading: Dict[str, Any]):
        stream_key = f"sensor:{node_id}:stream"
        
        # Sanitize inputs (numpy types to python types)
        sanitized = {}
        for k, v in reading.items():
            if v is None:
                continue
            if isinstance(v, bool):
                sanitized[k] = int(v)
            elif hasattr(v, 'item'):
                sanitized[k] = v.item()
            else:
                sanitized[k] = v
        
        if self.connected:
            try:
                self.client.xadd(stream_key, sanitized)
                self.client.xtrim(stream_key, maxlen=1000)
            except Exception as e:
                logger.error(f"Redis write error: {e}")
        else:
            # Fallback
            if stream_key not in self._in_memory_streams:
                self._in_memory_streams[stream_key] = []
            self._in_memory_streams[stream_key].append(sanitized)
            if len(self._in_memory_streams[stream_key]) > 1000:
                self._in_memory_streams[stream_key].pop(0)

    def get_latest_readings(self, node_id: str, count: int = 10) -> List[Dict]:
        stream_key = f"sensor:{node_id}:stream"
        
        if self.connected:
            try:
                data = self.client.xrevrange(stream_key, count=count)
                results = []
                for _, fields in data:
                    results.append(fields)
                return results
            except Exception as e:
                logger.error(f"Redis read error: {e}")
                return []
        else:
            # Fallback
            if stream_key not in self._in_memory_streams:
                return []
            return list(self._in_memory_streams[stream_key])[-count:][::-1]

    def store_alert(self, alert_data: Dict[str, Any]):
        if self.connected:
            try:
                self.client.lpush("alerts:history", json.dumps(alert_data))
                self.client.ltrim("alerts:history", 0, 99)
                self.client.publish("alerts:live", json.dumps(alert_data))
            except Exception as e:
                logger.error(f"Redis alert store error: {e}")
        else:
             # Fallback
            key = "alerts:history"
            if key not in self._in_memory_lists:
                self._in_memory_lists[key] = []
            self._in_memory_lists[key].insert(0, alert_data) # Front insert
            if len(self._in_memory_lists[key]) > 100:
                self._in_memory_lists[key].pop()

    def get_alerts(self, limit: int = 20) -> List[Dict]:
        if self.connected:
            try:
                raw_alerts = self.client.lrange("alerts:history", 0, limit - 1)
                return [json.loads(a) for a in raw_alerts]
            except Exception as e:
                logger.error(f"Redis alert get error: {e}")
                return []
        else:
            # Fallback
            key = "alerts:history"
            return self._in_memory_lists.get(key, [])[:limit]

    def clear_data(self):
        if self.connected:
            try:
                self.client.flushdb()
                logger.info("Redis database flushed.")
            except Exception as e:
                logger.error(f"Redis flush error: {e}")
        else:
            self._in_memory_streams = {}
            self._in_memory_lists = {}