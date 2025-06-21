import asyncio
import signal
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
    def __init__(
        self,
        args: TrainingArgs,
        config: RolloutControllerConfig,
        workflow: RolloutWorkflow,
    ):
        self.args = args
        self.config = config
        self.gconfig = config.gconfig

        # For staleness control
        self.train_batch_size = args.train_dataset.batch_size

        self.workflow = workflow

        self._exiting = threading.Event()
        self._lock = threading.Lock()
        self._buffer: List[List[Trajectory]] = []
        self._version = 0
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.stop_generate_loop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

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
            target=self._generate_until_complete,
            args=(dataloader,),
            daemon=True,
        )
        self._generation_thread.start()

    def stop_generate_loop(self):
        """Stop the episode loop and wait for it to finish."""
        logger.info("Stopping generation loop...")
        self._exiting.set()
        if self._generation_thread and self._generation_thread.is_alive():
            # Give the thread some time to finish gracefully
            self._generation_thread.join(timeout=10.0)

    def prepare_batch(self, batch_size: int) -> List[Trajectory]:
        """Prepare and wait for a batch of trajectories."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._prepare_batch_async(batch_size))
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
        return datapack.flat2d(data)

    def _generate_until_complete(self, dataloader):
        """Start an event loop to run episodes continuously."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._generate_loop(dataloader))
        finally:
            loop.close()

    async def _run_grouped_episode_async(self, rid, data, seed=None):
        factory = RolloutWorkflowFactory(self.args)
        collector = factory.make_workflow(self.config.workflow)
        tasks = [
            collector.run_episode_async(
                self.gconfig.new(n_samples=1),
                env_option=data,
                seed=seed,
            )
            for _ in range(self.gconfig.n_samples)
        ]
        return rid, await asyncio.gather(*tasks)

    async def _generate_loop(self, dataloader):
        data_generator = iter(dataloader)
        data = None

        rollout_stat = RolloutStat()

        rollout_tasks = {}
        rid = 0

        while not self._exiting.is_set():
            print(111111111111, flush=True)
            # Load next data
            if data is None:
                try:
                    data = next(data_generator)
                except StopIteration:
                    data_generator = iter(dataloader)
                    data = next(data_generator)

            # Create rollout task if possible.
            # Check if we have capacity.
            if dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = 1
            print(222222222222222, flush=True)
            capacity = self.config.max_concurrent_rollouts // world_size
            can_rollout = len(rollout_tasks) < capacity
            # Staleness control.
            ofp = self.config.max_head_offpolicyness
            sample_cnt = rollout_stat.accepted + rollout_stat.running
            expected_version = sample_cnt // self.train_batch_size
            with self._lock:
                can_rollout &= expected_version <= ofp + self._version
                if not can_rollout:
                    logger.info(
                        f"Cannot submit new rollouts. "
                        f"Capacity: {capacity}, current tasks: {len(tasks)}."
                        f"Sample cnt: {sample_cnt}, train batch size: {self.train_batch_size}, "
                        f"current version: {self._version}, max offpolicyness: {ofp}."
                    )
            print(33333333333, flush=True)

            # Create the new rollout task.
            if can_rollout:
                task = asyncio.create_task(self._run_grouped_episode_async(rid, data))
                rollout_tasks[rid] = task
                task.add_done_callback(lambda t: rollout_tasks.pop(rid, None))

                rollout_stat.submitted += 1
                rollout_stat.running += 1
                logger.info(
                    f"Submit a new rollout rid {rid}. "
                    f"Submit: {rollout_stat.submitted}, "
                    f"running: {rollout_stat.running}, "
                    f"accepted: {rollout_stat.accepted}."
                )

                rid += 1
                data = None

            print(44444444444444, flush=True)
            # Wait for rollout completion.
            tasks = list(rollout_tasks.values())
            done = []
            if tasks:
                done, _ = await asyncio.wait(
                    tasks,
                    timeout=ROLLOUT_POLL_WAIT_TIME,
                    return_when=asyncio.FIRST_COMPLETED,
                )

            print(5555555555555, flush=True)
            # Collect done results.
            for task in done:
                print(5.05, flush=True)
                rid, trajs = await task
                print(5.07, type(trajs), flush=True)
                trajs: List[Trajectory]
                assert isinstance(trajs, list) and isinstance(trajs[0], Trajectory)
                rollout_stat.running -= 1
                print(5.1, flush=True)

                # Filter data according to episodic return.
                ret = np.mean([traj.stats.total_reward for traj in trajs])
                accepted = ret >= self.config.filter_reward_lb
                accepted &= ret <= self.config.filter_reward_ub
                print(5.2, flush=True)
                if accepted:
                    with self._lock:
                        self._buffer.append(trajs)
                    rollout_stat.accepted += 1
                logger.info(
                    f"Finish rollout for rid {rid}. "
                    f"Submit: {rollout_stat.submitted}, "
                    f"running: {rollout_stat.running}, "
                    f"accepted: {rollout_stat.accepted}."
                )
        print(6666666666666, flush=True)
        for task in rollout_tasks.values():
            task.cancel()
