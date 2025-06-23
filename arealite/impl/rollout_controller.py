# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

import asyncio
import multiprocessing as mp
import threading
import time
import traceback
from queue import Empty as QueueEmpty
from typing import Any, List, Optional

import numpy as np

from arealite.api.cli_args import RolloutControllerConfig, TrainingArgs
from arealite.api.io_struct import Trajectory
from arealite.api.rollout_api import RolloutWorkflow
from realhf.base import datapack, logging, network
from realhf.system.push_pull_stream import ZMQJsonPuller, ZMQJsonPusher

logger = logging.getLogger("Rollout Controller")


class RolloutController:
    def __init__(
        self,
        args: TrainingArgs,
        config: RolloutControllerConfig,
        workflow: RolloutWorkflow,
    ):
        self.args = args
        self.config = config
        self.gconfig = config.gconfig
        self.workflow = workflow

        # Process-based execution
        self._exiting = mp.Event()
        self._lock = mp.Lock()
        self._buffer: List[List[Trajectory]] = []
        self._version = 0

        # Worker processes
        self._worker_processes: List[mp.Process] = []

        # PushPull communication for data to workers
        self._data_pusher = None
        self._data_pusher_port = None
        self._puller = None
        self._puller_port = None
        self._collector_thread = None

    ################### User Interfaces Start #################

    def generate_batch(
        self,
        batch_size: int,
        env_options: Optional[List[Any]] = None,
        seeds: Optional[List[int]] = None,
    ) -> List[Trajectory]:
        """Run episodes in batch using the workflow directly (for compatibility)."""
        if self.config.num_workers == 1:
            return self._generate_batch_sequential(batch_size, env_options, seeds)

        # For multi-worker, we rely on the continuous generation loop
        # This method should primarily be used for immediate batch generation
        return self._generate_batch_sequential(batch_size, env_options, seeds)

    def start_generate_loop(self):
        """Start worker processes that run generation loops."""
        logger.info("Starting worker processes...")

        # Start background thread to collect data from workers
        self._puller_port = network.find_free_port(
            experiment_name=self.args.experiment_name, trial_name=self.args.trial_name
        )
        self._collector_thread = threading.Thread(
            target=self._collect_from_workers, daemon=True
        )
        self._collector_thread.start()

        # Start worker processes
        self._data_pusher_port = network.find_free_port(
            experiment_name=self.args.experiment_name, trial_name=self.args.trial_name
        )
        self._data_pusher = ZMQJsonPusher(
            host="localhost", port=self._data_pusher_port, bind=True
        )
        logger.info(f"RolloutController sending data on port {self._data_pusher_port}")

        num_workers = self.config.num_workers
        for worker_id in range(num_workers):
            process = mp.Process(target=self._run_worker_process, args=(worker_id,))
            process.start()
            self._worker_processes.append(process)
            logger.info(f"Started worker process {worker_id}")

    def submit(self, data):
        """Submit data to worker processes for processing."""
        if self._data_pusher is None:
            raise RuntimeError(
                "Data pusher not initialized. Call start_generate_loop() first."
            )

        # Convert data to JSON-compatible format
        assert isinstance(data, list)
        for d in data:
            self._data_pusher.push(d)
        logger.info(f"Submitted {len(data)} data to workers")

    def prepare_batch(self, batch_size: int) -> List[Trajectory]:
        """Prepare and wait for a batch of trajectories."""
        buf_size = -1
        while buf_size < batch_size:
            with self._lock:
                buf_size = len(self._buffer)
            time.sleep(0.1)
        with self._lock:
            self._buffer = sorted(
                self._buffer, key=lambda x: np.mean([xx.stats.start_time for xx in x])
            )
            data, self._buffer = self._buffer[:batch_size], self._buffer[batch_size:]
        return datapack.flat2d(data)

    def stop_generate_loop(self):
        """Stop worker processes and cleanup."""
        logger.info("Stopping worker processes...")
        self._exiting.set()

        # Stop worker processes gracefully first, then forcefully if needed
        for i, process in enumerate(self._worker_processes):
            if process.is_alive():
                logger.info(f"Terminating worker process {i}...")
                try:
                    process.terminate()
                    process.join(timeout=1.0)
                except Exception:
                    process.kill()
        self._worker_processes.clear()

        if self._collector_thread is not None:
            # Wait for the thread to finish (with optional timeout)
            self._collector_thread.join(timeout=1.0)

        # Close communication channels
        if self._puller:
            self._puller.close()
        if self._data_pusher:
            self._data_pusher.close()
        logger.info("Cleanup completed")

    ################## User Interfaces End ##################

    def _run_worker_process(self, worker_id: int):
        """Run a worker process using multiprocessing."""
        from arealite.impl.rollout_worker import RolloutWorker

        # Create and run worker directly
        worker = RolloutWorker(
            worker_id=worker_id,
            args=self.args,
            config=self.config,
            pusher_host="localhost",
            pusher_port=self._puller_port,
            data_puller_host="localhost",
            data_puller_port=self._data_pusher_port,
        )
        logger.info(f"Worker {worker_id} starting generation loop...")
        worker.run_generation_loop()

    def _collect_from_workers(self):
        """Background thread to collect trajectories from workers."""
        # Find a free port
        self._puller = ZMQJsonPuller(host="localhost", port=self._puller_port)
        logger.info(f"RolloutController listening on port {self._puller_port}")

        while not self._exiting.is_set():
            try:
                # Pull data from workers
                data = self._puller.pull(timeout_ms=100)
                # Convert back to Trajectory objects
                trajs = [
                    Trajectory.from_json_compatible(traj_data)
                    for traj_data in data["trajs"]
                ]
                # Add to buffer
                with self._lock:
                    self._buffer.append(trajs)
                logger.info(
                    f"Received {len(trajs)} trajectories from worker {data['worker_id']}"
                )
            except QueueEmpty:
                # No data available, continue
                time.sleep(0.1)
                continue
            except Exception as e:
                if not self._exiting.is_set():
                    logger.error(f"Error in collector thread: {e}")
                    logger.error(traceback.format_exc())
                break

    def _generate_batch_sequential(
        self,
        batch_size: int,
        env_options: Optional[List[Any]] = None,
        seeds: Optional[List[int]] = None,
    ) -> List[Trajectory]:
        n_reqs = batch_size * self.gconfig.n_samples
        if env_options is None:
            env_options = [None] * n_reqs
        else:
            assert len(env_options) == batch_size
            env_options = [env_options[i % batch_size] for i in range(n_reqs)]
        if seeds is None:
            seeds = [None] * n_reqs
        else:
            assert len(seeds) == batch_size
            seeds = [seeds[i % batch_size] for i in range(n_reqs)]
        assert len(env_options) == len(seeds) == n_reqs
        trajs = []
        for env_option, seed in zip(env_options, seeds):
            trajs.append(
                self.workflow.run_episode(
                    self.gconfig.new(n_samples=1), env_option, seed
                )
            )
        return trajs

    def _generate_batch_parallel(
        self,
        batch_size: int,
        env_options: Optional[List[Any]] = None,
        seeds: Optional[List[int]] = None,
    ) -> List[Trajectory]:
        if env_options is None:
            env_options = [None] * batch_size
        else:
            assert len(env_options) == batch_size
        if seeds is None:
            seeds = [None] * batch_size
        else:
            assert len(seeds) == batch_size

        async def run_parallel_gen():
            tasks = [
                self._run_grouped_episode_async(None, env_option, seed)
                for env_option, seed in zip(env_options, seeds)
            ]
            results = await asyncio.gather(*tasks)
            return sum([r[1] for r in results], [])

        return asyncio.run(run_parallel_gen())

    async def _prepare_batch_async(self, batch_size: int):
        """Asynchronously wait for and return a batch of trajectories."""
        import numpy as np

        buf_size = -1
        while buf_size < batch_size:
            with self._lock:
                buf_size = len(self._buffer)
            await asyncio.sleep(0.1)

        with self._lock:
            self._buffer = sorted(
                self._buffer, key=lambda x: np.mean([xx.stats.start_time for xx in x])
            )
            data, self._buffer = self._buffer[:batch_size], self._buffer[batch_size:]

        # Flatten the list of lists
        return [traj for batch in data for traj in batch]
