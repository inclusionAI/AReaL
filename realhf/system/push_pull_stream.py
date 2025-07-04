import logging
import os
from queue import Empty as QueueEmpty
from typing import Any, Dict, List, Optional, Union

import orjson
import zmq
from zmq.utils.strtypes import asbytes

from realhf.base import constants, logging, name_resolve, names, network

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

    def __init__(
        self, host: str = "localhost", port: int = 5555, hwm: int = 1000, bind=False
    ):
        self.host = host
        self.port = port

        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.SNDHWM, hwm)
        if not bind:
            self.socket.connect(f"tcp://{self.host}:{self.port}")
        else:
            self.socket.bind(f"tcp://{self.host}:{self.port}")

    def push(self, data: JSONType) -> None:
        """
        Push JSON-compatible data efficiently.

        Args:
            data: JSON-serializable Python object

        Raises:
            TypeError: If data is not JSON-serializable
            zmq.ZMQError: If ZeroMQ operation fails
        """
        # Directly encode to bytes without intermediate string
        json_bytes = asbytes(orjson.dumps(data))
        self.socket.send(json_bytes, copy=False)

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
        bind: bool = True,
    ):
        self.host = host
        self.port = port
        self.default_timeout_ms = default_timeout_ms

        self.ctx = zmq.Context.instance()
        self.socket = self.ctx.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, hwm)
        self.socket.setsockopt(zmq.RCVTIMEO, self.default_timeout_ms)
        if bind:
            self.socket.bind(f"tcp://{self.host}:{self.port}")
        else:
            self.socket.connect(f"tcp://{self.host}:{self.port}")

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

    def pull(self, timeout_ms: Optional[int] = None):
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


def grouping(num_senders, num_receivers):
    groups = {}
    assert num_senders >= num_receivers
    # Each PULL gets multiple PUSH
    senders_per_receiver = num_senders // num_receivers
    for receiver_id in range(num_receivers):
        start = receiver_id * senders_per_receiver
        end = (receiver_id + 1) * senders_per_receiver
        groups[receiver_id] = list(range(start, end))
    # Distribute remaining senders
    remaining = num_senders % num_receivers
    for i in range(remaining):
        groups[i].append(num_receivers * senders_per_receiver + i)
    return groups


class NameResolvingZmqPusher(ZMQJsonPusher):
    def __init__(self, experiment_name, trial_name, pusher_index, pusher_cnt, **kwargs):
        pullers = name_resolve.get_subtree(
            names.stream_pullers(experiment_name, trial_name)
        )
        pullers = list(map(int, pullers))
        puller_cnt = len(pullers)
        assert sorted(pullers) == list(range(puller_cnt))
        groups = grouping(pusher_cnt, puller_cnt)
        puller_index = None
        for puller_index, pusher_indices in groups.items():
            if pusher_index in pusher_indices:
                break
        assert puller_index is not None
        name = names.push_pull_stream(
            experiment_name, trial_name, stream_name=f"puller{puller_index}"
        )
        addr = name_resolve.wait(name)
        host, port = addr.split(":")
        super().__init__(host, int(port), **kwargs)


class NameResolvingZmqPuller(ZMQJsonPuller):
    def __init__(self, args, puller_index, **kwargs):
        experiment_name = args.experiment_name
        trial_name = args.trial_name
        name = names.push_pull_stream(
            experiment_name, trial_name, stream_name=f"puller{puller_index}"
        )
        host, port = network.gethostip(), network.find_free_port(
            experiment_name=experiment_name,
            trial_name=trial_name,
            lockfile_root=os.path.join(constants.get_cache_path(args), "ports"),
        )
        addr = f"{host}:{port}"
        name_resolve.add(name, addr)
        super().__init__(host, port, **kwargs)
