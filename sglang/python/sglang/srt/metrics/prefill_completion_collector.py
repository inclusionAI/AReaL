#!/usr/bin/env python3
"""
Collector for per-request prefill completion statistics.

This module provides batching functionality for collecting and pushing
prefill completion stats to an external gRPC server.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PrefillCompletionCollector:
    """Collects and batches prefill completion stats for pushing to gRPC server."""

    def __init__(
        self,
        stats_pusher: Optional[Any],
        batch_size: int = 32,
        flush_interval_seconds: float = 5.0,
    ):
        """
        Initialize the prefill completion stats collector.

        Args:
            stats_pusher: StatsPusher instance for sending stats to gRPC server
            batch_size: Number of stats to collect before flushing
            flush_interval_seconds: Maximum time to wait before flushing (for low traffic)
        """
        self.stats_pusher = stats_pusher
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds

        # Buffer for collecting stats
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_lock = threading.Lock()

        # Track last flush time for periodic flushing
        self.last_flush_time = time.time()

    def collect(self, stats: Dict[str, Any]) -> None:
        """
        Collect prefill completion stats for a single request.

        Args:
            stats: Dictionary containing prefill completion metrics
        """
        if self.stats_pusher is None:
            return

        with self.buffer_lock:
            self.buffer.append(stats)

            # Check if we should flush
            if self.should_flush_locked():
                self._flush_locked()

    def should_flush_locked(self) -> bool:
        """
        Check if buffer should be flushed (must be called with lock held).

        Returns:
            True if buffer should be flushed
        """
        # Flush if batch size reached
        if len(self.buffer) >= self.batch_size:
            return True

        # Flush if flush interval exceeded and buffer is not empty
        if (
            self.buffer
            and time.time() - self.last_flush_time >= self.flush_interval_seconds
        ):
            return True

        return False

    def _flush_locked(self) -> None:
        """Flush the buffer to stats pusher (must be called with lock held)."""
        if not self.buffer:
            return

        try:
            # Make a copy of buffer for sending
            stats_to_send = self.buffer

            # Clear buffer immediately to avoid blocking collection
            self.buffer = []
            self.last_flush_time = time.time()

            # Send stats (release lock during network call would be better,
            # but StatsPusher.push_prefill_completion_stats_batch should be fast)
            self.stats_pusher.push_prefill_completion_stats_batch(stats_to_send)

            logger.debug(f"Flushed {len(stats_to_send)} prefill completion stats")

        except Exception as e:
            logger.debug(f"Failed to push prefill completion stats: {e}")

    def flush(self) -> None:
        """Manually flush the buffer (useful for shutdown or testing)."""
        with self.buffer_lock:
            self._flush_locked()

    def __del__(self):
        """Flush any remaining stats on destruction."""
        try:
            self.flush()
        except Exception:
            pass  # Ignore errors during cleanup
