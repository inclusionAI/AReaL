import logging
from queue import Empty as QueueEmpty
from typing import Any, Dict, List, Optional, Union

import orjson
import zmq
from zmq.utils.strtypes import asbytes

from realhf.base import logging

logger = logging.getLogger("ZMQ Push-Pull Stream")

# Type alias for JSON-compatible objects
JSONType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


class ZMQJsonPusher:
    """
    JSON pusher using ZeroMQ.

    Args:
        host: Host address (default: 'localhost')
        port: Port number (default: 5555)
        hwm: High-water mark for outgoing messages (default: 1000)
    """

    def __init__(self, host: str = "localhost", port: int = 5555, hwm: int = 1000):
        self.host = host
        self.port = port

        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.SNDHWM, hwm)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def push(self, data: JSONType) -> None:
        """
        Push JSON-compatible data efficiently.

        Args:
            data: JSON-serializable Python object

        Raises:
            TypeError: If data is not JSON-serializable
            zmq.ZMQError: If ZeroMQ operation fails
        """
        try:
            # Directly encode to bytes without intermediate string
            json_bytes = asbytes(orjson.dumps(data))
            self.socket.send(json_bytes, flags=zmq.NOBLOCK, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Data not JSON-serializable: {e}")
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                logger.warning("Push operation would block (queue full)")
            raise

    def close(self) -> None:
        """Clean up resources."""
        self.socket.close(linger=0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ZMQJsonPuller:
    """
    JSON puller using ZeroMQ with per-call timeout support in pull() method.

    Args:
        host: Host address (default: 'localhost')
        port: Port number (default: 5555)
        default_timeout_ms: Default receive timeout in milliseconds (default: 1000)
        hwm: High-water mark for incoming messages (default: 1000)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        default_timeout_ms: int = 1000,
        hwm: int = 1000,
    ):
        self.host = host
        self.port = port
        self.default_timeout_ms = default_timeout_ms

        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, hwm)
        self.socket.setsockopt(zmq.RCVTIMEO, self.default_timeout_ms)
        self.socket.bind(f"tcp://{self.host}:{self.port}")

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

    def pull(self, timeout_ms: Optional[int] = None) -> JSONType:
        """
        Pull and decode JSON data with configurable timeout.

        Args:
            timeout_ms: Optional timeout in seconds. If None, uses default_timeout_ms.

        Returns:
            Deserialized JSON-compatible Python object

        Raises:
            queue.Empty: If no message available within timeout
        """
        current_timeout = self.default_timeout_ms if timeout_ms is None else timeout_ms
        events = dict(self.poller.poll(current_timeout))
        if self.socket in events:
            msg = self.socket.recv(flags=zmq.NOBLOCK, copy=False)
            return orjson.loads(msg.bytes.decode("utf-8"))
        raise QueueEmpty(f"No data available after {current_timeout}ms timeout")

    def close(self) -> None:
        """Clean up resources."""
        self.socket.close(linger=0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
