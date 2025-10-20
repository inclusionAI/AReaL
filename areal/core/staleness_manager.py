"""Staleness-aware capacity manager for rollout generation.

This module provides the StalenessManager class which manages capacity
and staleness constraints for asynchronous rollout generation in RL training.
"""

from threading import Lock

from areal.api.io_struct import RolloutStat


class StalenessManager:
    """Manages rollout capacity based on staleness and concurrency constraints.

    The manager ensures that:
    1. The number of concurrent rollouts doesn't exceed the configured maximum
    2. Rollouts don't become too stale (off-policy) by limiting acceptance based on
       the current model version and maximum allowed offpolicyness

    Parameters
    ----------
    max_concurrent_rollouts : int
        Maximum number of concurrent rollouts allowed
    consumer_batch_size : int
        Expected batch size for consuming rollouts during training
    max_staleness : int
        Maximum allowed offpolicyness (version difference) for rollouts
    """

    def __init__(
        self,
        max_concurrent_rollouts: int,
        consumer_batch_size: int,
        max_staleness: int,
    ):
        """Initialize the staleness manager.

        Parameters
        ----------
        max_concurrent_rollouts : int
            Maximum number of concurrent rollouts allowed
        consumer_batch_size : int
            Expected batch size for consuming rollouts during training
        max_staleness : int
            Maximum allowed offpolicyness (version difference) for rollouts
        """
        self.max_concurrent_rollouts = max_concurrent_rollouts
        self.consumer_batch_size = consumer_batch_size
        self.max_staleness = max_staleness

        # Thread-safe access to rollout statistics
        self.lock = Lock()
        self.rollout_stat = RolloutStat()

    def get_capacity(self, current_version: int) -> int:
        """Calculate available capacity for new rollouts.

        This method considers both concurrency limits and staleness constraints
        to determine how many new rollouts can be accepted.

        The capacity calculation ensures:
        1. The number of running rollouts doesn't exceed max_concurrent_rollouts
        2. Samples don't become too stale by limiting based on:
           - current_version: The current model version
           - max_staleness: Maximum allowed version difference
           - consumer_batch_size: Expected batch size for training

        Parameters
        ----------
        current_version : int
            The current version of the model weights

        Returns
        -------
        int
            Number of new rollout slots available. Can be negative if over capacity.

        Notes
        -----
        The staleness control formula is:
        max_samples = (max_staleness + current_version + 1) * consumer_batch_size
        capacity = min(concurrency_limit, max_samples - current_samples)

        This ensures that by the time samples are consumed, they won't exceed
        the maximum allowed staleness.
        """
        with self.lock:
            # Calculate concurrency-based capacity
            max_concurrent_rollouts = max(1, self.max_concurrent_rollouts)
            concurrency_capacity = max_concurrent_rollouts - self.rollout_stat.running

            # Calculate staleness-based capacity
            ofp = self.max_staleness
            sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
            consumer_bs = max(1, self.consumer_batch_size)
            staleness_capacity = (ofp + current_version + 1) * consumer_bs - sample_cnt

            # Return the minimum of both constraints
            capacity = min(concurrency_capacity, staleness_capacity)
            return capacity

    def on_rollout_submitted(self) -> None:
        """Callback when a rollout is submitted for execution.

        Thread-safe method to increment the submitted and running counters.
        """
        with self.lock:
            self.rollout_stat.submitted += 1
            self.rollout_stat.running += 1

    def on_rollout_accepted(self) -> None:
        """Callback when a rollout completes successfully and is accepted.

        Thread-safe method to increment accepted counter and decrement running counter.
        """
        with self.lock:
            self.rollout_stat.accepted += 1
            self.rollout_stat.running -= 1

    def on_rollout_rejected(self) -> None:
        """Callback when a rollout completes but is rejected.

        Thread-safe method to decrement running counter only.
        This is called when a trajectory is filtered out by should_accept or
        when the workflow returns None. The rollout was never added to accepted,
        so we only need to decrement running.
        """
        with self.lock:
            self.rollout_stat.running -= 1

    def get_stats(self) -> RolloutStat:
        """Get a snapshot of current rollout statistics.

        Returns
        -------
        RolloutStat
            Current rollout statistics (submitted, accepted, running)
        """
        with self.lock:
            return RolloutStat(
                submitted=self.rollout_stat.submitted,
                accepted=self.rollout_stat.accepted,
                running=self.rollout_stat.running,
            )
