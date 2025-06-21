import asyncio
import queue
from typing import Any, Dict, List, Optional

import numpy as np
import torch.distributed as dist

from arealite.api.cli_args import RolloutControllerConfig, TrainingArgs
from arealite.api.io_struct import Trajectory
from arealite.api.rollout_api import RolloutWorkflowFactory
from realhf.base import logging, name_resolve, names
from realhf.base.monitor import RolloutStat
from realhf.system.push_pull_stream import ZMQJsonPuller, ZMQJsonPusher

logger = logging.getLogger("RolloutWorker")

ROLLOUT_POLL_WAIT_TIME = 0.4


class RolloutWorker:
    """Standalone rollout worker that runs continuous generation loop."""

    def __init__(
        self,
        worker_id: int,
        args: TrainingArgs,
        config: RolloutControllerConfig,
        pusher_host: str = "localhost",
        pusher_port: int = 5555,
        data_puller_host: str = "localhost",
        data_puller_port: int = 5556,
    ):
        self.worker_id = worker_id
        self.args = args
        self.config = config
        self.gconfig = config.gconfig

        # For staleness control
        self.train_batch_size = args.train_dataset.batch_size

        self.pusher_host = pusher_host
        self.pusher_port = pusher_port
        self.data_puller_host = data_puller_host
        self.data_puller_port = data_puller_port

        self._shutdown = False
        self.pusher = None
        self.data_puller = None

    def _cleanup(self):
        """Clean up resources."""
        if self.pusher:
            self.pusher.close()
        if self.data_puller:
            self.data_puller.close()

    def run_generation_loop(self):
        """Run the continuous generation loop like the original _generate_loop."""
        try:
            asyncio.run(self._generate_loop())
        finally:
            self._cleanup()

    async def _run_grouped_episode_async(
        self, rid: int, data: Any, seed: Optional[int] = None
    ):
        """Run grouped episode asynchronously."""
        tasks = []
        for _ in range(self.gconfig.n_samples):
            # Create workflow
            factory = RolloutWorkflowFactory(self.args)
            workflow = factory.make_workflow(self.config.workflow)
            tasks += [
                workflow.run_episode_async(
                    self.gconfig.new(n_samples=1),
                    env_option=data,
                    seed=seed,
                )
                for _ in range(self.gconfig.n_samples)
            ]
        trajs = await asyncio.gather(*tasks)
        return rid, trajs

    def _get_model_version(self) -> int:
        name = names.model_version(
            self.args.experiment_name,
            self.args.trial_name,
            "actor",
        )
        try:
            return int(name_resolve.get(name))
        except name_resolve.NameEntryNotFoundError:
            return 0

    async def _generate_loop(self):
        """Main generation loop - similar to original RolloutController._generate_loop."""
        data = None

        # Communication with main process
        self.pusher = ZMQJsonPusher(host=self.pusher_host, port=self.pusher_port)
        self.data_puller = ZMQJsonPuller(
            host=self.data_puller_host, port=self.data_puller_port
        )

        rollout_stat = RolloutStat()
        rollout_tasks: Dict[int, asyncio.Task] = {}
        rid = 0

        try:
            while not self._shutdown:
                # Load next data from controller
                if data is None:
                    try:
                        data = self.data_puller.pull(timeout_ms=50)
                        logger.debug(f"Get data from puller: {data}")
                    except queue.Empty:
                        logger.debug(f"No data from puller stream.")
                        await asyncio.sleep(0.1)
                        continue
                    except Exception as e:
                        if not self._shutdown:
                            logger.error(
                                f"Worker {self.worker_id} error pulling data: {e}"
                            )
                        await asyncio.sleep(0.1)
                        continue

                # Check capacity
                if dist.is_initialized():
                    world_size = dist.get_world_size()
                else:
                    world_size = 1

                capacity = self.config.max_concurrent_rollouts // world_size
                can_rollout = len(rollout_tasks) < capacity

                # Staleness control
                version = self._get_model_version()
                ofp = self.config.max_head_offpolicyness
                sample_cnt = rollout_stat.accepted + rollout_stat.running
                expected_version = sample_cnt // self.train_batch_size
                can_rollout &= expected_version <= ofp + version

                if not can_rollout:
                    logger.debug(
                        f"Worker {self.worker_id}: Cannot submit new rollouts. "
                        f"Capacity: {capacity}, current tasks: {len(rollout_tasks)}."
                        f"Sample cnt: {sample_cnt}, train batch size: {self.train_batch_size}, "
                        f"current version: {version}, max offpolicyness: {ofp}."
                    )
                    await asyncio.sleep(0.1)
                    continue

                # Create new rollout task
                if can_rollout:
                    task = asyncio.create_task(
                        self._run_grouped_episode_async(rid, data)
                    )
                    rollout_tasks[rid] = task

                    rollout_stat.submitted += 1
                    rollout_stat.running += 1
                    logger.info(
                        f"Worker {self.worker_id}: Submit rollout {rid}. "
                        f"Submit: {rollout_stat.submitted}, "
                        f"running: {rollout_stat.running}, "
                        f"accepted: {rollout_stat.accepted}."
                    )

                    rid += 1
                    data = None

                # Wait for rollout completion
                tasks = list(rollout_tasks.values())
                done = []
                if tasks:
                    done, _ = await asyncio.wait(
                        tasks,
                        timeout=ROLLOUT_POLL_WAIT_TIME,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                # Collect done results
                for task in done:
                    task_rid, trajs = await task
                    trajs: List[Trajectory]
                    rollout_tasks.pop(task_rid, None)
                    rollout_stat.running -= 1

                    # Filter data according to episodic return
                    ret = np.mean([traj.stats.total_reward for traj in trajs])
                    accepted = ret >= self.config.filter_reward_lb
                    accepted &= ret <= self.config.filter_reward_ub

                    if accepted:
                        # Push trajectories to main process
                        trajectory_data = {
                            "worker_id": self.worker_id,
                            "trajs": [traj.to_json_compatible() for traj in trajs],
                        }
                        self.pusher.push(trajectory_data)
                        rollout_stat.accepted += 1

                    logger.info(
                        f"Worker {self.worker_id}: Finish rollout {task_rid}. "
                        f"Submit: {rollout_stat.submitted}, "
                        f"running: {rollout_stat.running}, "
                        f"accepted: {rollout_stat.accepted}."
                    )
        finally:
            # Cancel remaining tasks
            for task in rollout_tasks.values():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
