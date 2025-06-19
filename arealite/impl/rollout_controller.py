import asyncio
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from arealite.api.agentic_api import AgenticWorkflowFactory, AgenticWorkflow
from arealite.api.cli_args import RolloutControllerConfig, TrainingArgs
from arealite.api.io_struct import LLMRequest, Trajectory
from arealite.api.llm_client_api import LLMClient, LLMResponse
from realhf.base import logging
from realhf.base.monitor import RolloutStat

logger = logging.getLogger("Rollout Controller")

ROLLOUT_POLL_WAIT_TIME = 0.4


class RolloutController:
    def __init__(self, args: TrainingArgs, config: RolloutControllerConfig):
        self.args = args
        self.config = config

        self.train_batch_size = args.train_dataset.batch_size

        self._exiting = threading.Event()
        self._lock = threading.Lock()
        self._buffer: List[Trajectory] = []
        self._version = 0

    ################### User Interfaces Start #################
    # TODO: invocation of these APIs are too deep

    def generate_batch(self, llm_client: LLMClient, reqs: LLMRequest) -> List[LLMResponse]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = [self._generate_single(llm_client, req) for req in reqs]
        try:
            resps = loop.run_until_complete(asyncio.gather(*tasks))
            return resps
        finally:
            loop.close()

    def run_episode_batch(
        self,
        env_options: Optional[List[Any]] = None,
        seeds: Optional[List[int]] = None,
        num_workers: Optional[int] = None,
    ) -> List[Trajectory]:
        # TODO: consider how to unwrap this function to make it simpler
        """Run episodes in batch using efficient multiprocessing for CUDA tensors."""
        if num_workers is None:
            num_workers = min(len(collectors), mp.cpu_count())

        factory = AgenticWorkflowFactory(self.args, self.config.llm_client)
        collectors = [factory.make_workflow(self.config.workflow) for _ in range(num_workers)]

        if env_options is None:
            env_options = [None] * len(collectors)
        if seeds is None:
            seeds = [None] * len(collectors)

        def _run_episode_worker(args):
            collector, env_option, seed = args
            collector: AgenticWorkflow
            return collector.run_episode(env_option, seed)

        # Set sharing strategy for tensors - file_descriptor is more efficient for CUDA
        mp.set_sharing_strategy("file_descriptor")

        # Use ProcessPoolExecutor for better resource management
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn")) as executor:
            tasks = list(zip(collectors, env_options, seeds))
            trajectories = list(executor.map(_run_episode_worker, tasks))

        return trajectories

    def start_run_episode_loop(self, dataloader: DataLoader):
        """Start the episode loop in a separate thread."""
        self._generation_thread = threading.Thread(
            target=self._run_episode_until_complete, args=(dataloader,)
        )
        self._generation_thread.start()

    def stop_run_episode_loop(self):
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

    async def _generate_single(self, llm_client: LLMClient, req: LLMRequest):
        """A trick to make an async generation function."""
        return await asyncio.to_thread(llm_client.generate, req)

    async def _prepare_batch_async(self, batch_size: int):
        buf_size = -1
        while buf_size < batch_size:
            with self._lock:
                buf_size = len(self._buffer)
            await asyncio.sleep(0.1)
        with self._lock:
            self._buffer = sorted(self._buffer, lambda x: x.stats.start_time)
            data, self._buffer = self._buffer[:batch_size], self._buffer[batch_size:]
        return data

    def _run_episode_until_complete(self):
        """Start an event loop to run episodes continuously."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_episode_loop())
        finally:
            loop.close()

    async def _run_single_episode_async(self, rid, data):
        factory = AgenticWorkflowFactory(self.args, self.config.llm_client)
        collector = factory.make_workflow(self.config.workflow)
        return rid, await collector.run_episode_async(env_option=data)

    async def _run_episode_loop(self, dataloader):
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
                rid, traj = await task
                assert isinstance(traj, Trajectory)
                rollout_stat.running -= 1

                # Filter data according to episodic return.
                ret = traj.stats.total_reward
                accepted = ret >= self.config.filter_reward_lb
                accepted &= ret <= self.config.filter_reward_ub
                if accepted:
                    with self._lock:
                        self._buffer.append(traj)
                    rollout_stat.accepted += 1
                logger.debug(
                    f"Finish rollout for rid {rid}. "
                    f"Submit: {rollout_stat.submitted}, "
                    f"running: {rollout_stat.running}, "
                    f"accepted: {rollout_stat.accepted}."
                )
