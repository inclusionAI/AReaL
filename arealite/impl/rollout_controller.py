import asyncio
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Any, List, Optional

import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from arealite.api.cli_args import RolloutControllerConfig, TrainingArgs
from arealite.api.io_struct import Trajectory
from arealite.api.rollout_api import RolloutWorkflow, RolloutWorkflowFactory
from realhf.base import datapack, logging
from realhf.base.monitor import RolloutStat

logger = logging.getLogger("Rollout Controller")

ROLLOUT_POLL_WAIT_TIME = 0.4


class RolloutController:
    def __init__(self, args: TrainingArgs, config: RolloutControllerConfig):
        self.args = args
        self.config = config
        self.gconfig = config.gconfig

        self.train_batch_size = args.train_dataset.batch_size

        self._exiting = threading.Event()
        self._lock = threading.Lock()
        self._buffer: List[Trajectory] = []
        self._version = 0

    ################### User Interfaces Start #################

    def generate_batch(
        self,
        batch_size: int,
        env_options: Optional[List[Any]] = None,
        seeds: Optional[List[int]] = None,
    ) -> List[Trajectory]:
        """Run episodes in batch using efficient multiprocessing for CUDA tensors."""
        if self.config.num_workers == 1:
            return self._generate_batch_sequential(batch_size, env_options, seeds)

        return self._generate_batch_parallel(batch_size, env_options, seeds)

    def start_generate_loop(self, dataloader: DataLoader):
        """Start the episode loop in a separate thread."""
        self._generation_thread = threading.Thread(
            target=self._generate_until_complete, args=(dataloader,)
        )
        self._generation_thread.start()

    def stop_generate_loop(self):
        """Stop the episode loop and wait for it to finish."""
        self._exiting.set()
        if self._generation_thread.is_alive():
            self._generation_thread.join()

    def prepare_batch(self, batch_size: int) -> List[Trajectory]:
        """Prepare and wait for a batch of trajectories."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._prepare_batch_async(batch_size))
        finally:
            loop.close()

    def set_version(self, version: int):
        """Set the version of the trained model."""
        with self._lock:
            self._version = version

    ################## User Interfaces End ##################

    def _generate_batch_sequential(
        self,
        batch_size: int,
        env_options: Optional[List[Any]] = None,
        seeds: Optional[List[int]] = None,
    ) -> List[Trajectory]:
        n_reqs = batch_size * self.gconfig.n_samples
        factory = RolloutWorkflowFactory(self.args)
        collectors = [
            factory.make_workflow(self.config.workflow) for _ in range(n_reqs)
        ]
        if env_options is None:
            env_options = [None] * len(n_reqs)
        else:
            assert len(env_options) == batch_size
            env_options = [env_options[i % batch_size] for i in range(n_reqs)]
        if seeds is None:
            seeds = [None] * len(n_reqs)
        else:
            assert len(seeds) == batch_size
            seeds = [seeds[i % batch_size] for i in range(n_reqs)]
        assert len(env_options) == len(seeds) == n_reqs
        trajs = []
        for collector, env_option, seed in zip(collectors, env_options, seeds):
            collector: RolloutWorkflow
            trajs.append(
                collector.run_episode(self.gconfig.new(n_samples=1), env_option, seed)
            )
        return trajs

    def _generate_batch_parallel(
        self,
        batch_size: int,
        env_options: Optional[List[Any]] = None,
        seeds: Optional[List[int]] = None,
    ) -> List[Trajectory]:
        n_reqs = batch_size * self.gconfig.n_samples
        factory = RolloutWorkflowFactory(self.args)
        collectors = [
            factory.make_workflow(self.config.workflow) for _ in range(n_reqs)
        ]
        if env_options is None:
            env_options = [None] * len(n_reqs)
        else:
            assert len(env_options) == batch_size
            env_options = [env_options[i % batch_size] for i in range(n_reqs)]
        if seeds is None:
            seeds = [None] * len(n_reqs)
        else:
            assert len(seeds) == batch_size
            seeds = [seeds[i % batch_size] for i in range(n_reqs)]
        assert len(env_options) == len(seeds) == n_reqs

        def _generate_worker(args):
            collector, env_option, seed = args
            collector: RolloutWorkflow
            return collector.run_episode(
                self.gconfig.new(n_samples=1), env_option, seed
            )

        # Set sharing strategy for tensors - file_descriptor is more efficient for CUDA
        mp.set_sharing_strategy("file_descriptor")
        # Use ProcessPoolExecutor for better resource management
        with ProcessPoolExecutor(
            max_workers=self.config.num_workers, mp_context=mp.get_context("spawn")
        ) as executor:
            tasks = list(zip(collectors, env_options, seeds))
            trajectories = list(executor.map(_generate_worker, tasks))
        return trajectories

    async def _prepare_batch_async(self, batch_size: int):
        buf_size = -1
        while buf_size < batch_size:
            with self._lock:
                buf_size = len(self._buffer)
            await asyncio.sleep(0.1)
        with self._lock:
            self._buffer = sorted(self._buffer, lambda x: x.stats.start_time)
            data, self._buffer = self._buffer[:batch_size], self._buffer[batch_size:]
        return datapack.flat2d(data)

    def _generate_until_complete(self, dataloader: DataLoader):
        """Start an event loop to run episodes continuously."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._generate_loop(dataloader))
        finally:
            loop.close()

    async def _run_single_episode_async(self, rid, data):
        factory = RolloutWorkflowFactory(self.args)
        collector = factory.make_workflow(self.config.workflow)
        tasks = [
            asyncio.create_task(
                collector.run_episode_async(
                    self.gconfig.new(n_samples=1), env_option=data
                )
            )
            for _ in range(self.gconfig.n_samples)
        ]
        return rid, await asyncio.gather(tasks)
    
    # TODO: Fix bugs
    async def _generate_loop(self, dataloader):
        data_generator = iter(dataloader)
        data = None

        rollout_stat = RolloutStat()

        rollout_tasks = {}
        rid = 0

        while not self._exiting.is_set():
            # Load next data
            if data is None:
                try:
                    data = next(data_generator)
                except StopIteration:
                    data_generator = iter(dataloader)
                    data = next(data_generator)

            # Create rollout task if possible.
            # Check if we have capacity.
            world_size = dist.get_world_size()
            ofp = self.config.max_concurrent_rollouts
            can_rollout = len(rollout_tasks) < ofp // world_size
            # Staleness control.
            sample_cnt = rollout_stat.accepted + rollout_stat.running
            expected_version = sample_cnt // self.train_batch_size
            with self._lock:
                can_rollout &= expected_version <= ofp + self._version

            # Create the new rollout task.
            if can_rollout:
                task = asyncio.create_task(self._run_single_episode_async(rid, data))
                rollout_tasks[rid] = task
                task.add_done_callback(lambda t: rollout_tasks.pop(rid, None))

                rollout_stat.submitted += 1
                rollout_stat.running += 1
                logger.debug(
                    f"Submit a new rollout rid {rid}. "
                    f"Submit: {rollout_stat.submitted}, "
                    f"running: {rollout_stat.running}, "
                    f"accepted: {rollout_stat.accepted}."
                )

                rid += 1
                data = None

            # Wait for rollout completion.
            tasks = list(rollout_tasks.values())
            done = []
            if tasks:
                done, _ = await asyncio.wait(
                    tasks,
                    timeout=ROLLOUT_POLL_WAIT_TIME,
                    return_when=asyncio.FIRST_COMPLETED,
                )

            # Collect done results.
            for task in done:
                rid, trajs = await task
                assert isinstance(trajs, list) and isinstance(trajs[0], Trajectory)
                rollout_stat.running -= 1

                # Filter data according to episodic return.
                ret = np.mean([traj.stats.total_reward for traj in trajs])
                accepted = ret >= self.config.filter_reward_lb
                accepted &= ret <= self.config.filter_reward_ub
                if accepted:
                    with self._lock:
                        self._buffer.append(trajs)
                    rollout_stat.accepted += 1
                logger.debug(
                    f"Finish rollout for rid {rid}. "
                    f"Submit: {rollout_stat.submitted}, "
                    f"running: {rollout_stat.running}, "
                    f"accepted: {rollout_stat.accepted}."
                )
