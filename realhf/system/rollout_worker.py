import asyncio
import json
import os
import queue
import time
from asyncio.queues import QueueEmpty
from typing import Dict, Hashable, List

import aiohttp
import numpy as np
import torch.utils.data
from aiohttp.client import ClientTimeout

from realhf.api.core.agent_api import make_agent
from realhf.api.core.data_api import SequenceSample, load_hf_tokenizer, make_dataset
from realhf.api.core.env_api import make_env
from realhf.api.core.system_api import ExpStatus
from realhf.api.core.system_api import RolloutWorker as RolloutWorkerConfig
from realhf.base import (
    constants,
    datapack,
    logging,
    name_resolve,
    names,
    recover,
    seeding,
)
from realhf.base.monitor import RolloutStat
from realhf.system.partial_rollout import PartialRolloutManager
from realhf.system.push_pull_stream import NameResolvingZmqPusher
from realhf.system.worker_base import AsyncWorker, PollResult

# NOTE: Register all implemented agents
import realhf.impl.environment  # isort:skip
import realhf.impl.agent  # isort:skip

logger = logging.getLogger("RolloutWorker")

# Should be equal to the poll time of partial rollout
ROLLOUT_POLL_WAIT_TIME = 0.4


class RolloutWorker(AsyncWorker):
    def _configure(self, config: RolloutWorkerConfig):
        self.model_name = config.model_name

        self.config = config
        self.worker_index = config.worker_info.worker_index
        self.worker_count = config.worker_info.worker_count

        self.experiment_name = config.worker_info.experiment_name
        self.trial_name = config.worker_info.trial_name

        self.env = make_env(config.env)
        self.agent = make_agent(config.agent)

        self.rollout_request_queue = asyncio.Queue(1024)
        self.rollout_response_queue = asyncio.Queue(1024)

        self.act_queues = {}
        self.rollout_tasks = {}

        self.inference_maker = PartialRolloutManager(
            worker_index=self.worker_index,
            request_queue=self.rollout_request_queue,
            reply_queue=self.rollout_response_queue,
            new_tokens_per_chunk=config.new_tokens_per_chunk,
            tokenizer=load_hf_tokenizer(config.tokenizer_path),
            timeout=self.config.rollout_request_timeout,
        )
        self.push_stream = None

        seeding.set_random_seed(
            config.base_seed, f"rollout_worker{config.worker_info.worker_index}"
        )

        self.data_generator = None
        self.is_new_epoch = False

        self._cur_data = None

        self.gserver_manager_addr = None
        self.rollout_tasks: Dict[Hashable, asyncio.Task] = {}

        # Since the rollout worker doesn't compute staleness,
        # we don't need to recover rollout_stat here.
        self.rollout_stat = RolloutStat()

        # recover info
        self.__recover_run, self.__recover_info = recover.load_recover_info(self.args)

        return config.worker_info

    def make_datasets(self):
        # Make datasets.
        datasets = [
            make_dataset(
                d,
                # NOTE: we must use the same seed to ensure the same dataset split
                self.config.base_seed,
                self.worker_index,
                self.worker_count,
                self.config.tokenizer_path,
            )
            for d in self.config.datasets
        ]
        if len(self.config.datasets) == 1:
            self.dataset = datasets[0]
        else:
            self.dataset = torch.utils.data.ConcatDataset(datasets)
        self.dataset_size = len(self.dataset)
        g = torch.Generator()
        g.manual_seed(seeding.get_seed())
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=None,
            shuffle=True,
            generator=g,
        )
        self.data_generator = enumerate(self.dataloader)

        # Recover indices for dynamic dataset
        if hasattr(self.dataset, "filter"):
            dataset_indices_path = os.path.join(
                constants.get_log_path(self.args),
                f"dataset_indices_{self.worker_index}.npy",
            )
            if os.path.exists(dataset_indices_path):
                indices = np.load(dataset_indices_path).tolist()
                logger.info(
                    f"DP rank {self.worker_index} updating dataset indices upon recover, "
                    f"size {len(self.dataset.active_indices)} -> {len(indices)}"
                )
                self.dataset.active_indices = indices

    def load_next_data(self):
        # Create an epoch-wise barrier to prevent data over-consumption.
        if self.is_new_epoch:
            if len(self.rollout_tasks) > 0:
                return None
            self.is_new_epoch = False

        # Fetch.
        try:
            _, cur_sample = next(self.data_generator)
        except StopIteration:
            self.is_new_epoch = True
            # Upon the first fetch request, filter dataset and create dataloader.
            eval_scores_path = os.path.join(
                constants.get_log_path(self.args),
                "dataset_eval_scores.json",
            )
            dataset_indices_path = os.path.join(
                constants.get_log_path(self.args),
                f"dataset_indices_{self.worker_index}.npy",
            )
            if hasattr(self.dataset, "filter") and os.path.exists(eval_scores_path):
                # Don't filter dataset on the first poll after recover.
                with open(eval_scores_path, "r", encoding="utf-8") as f:
                    dataset_eval_scores = json.load(f)
                self.dataset.filter(dataset_eval_scores)
                # Save the dataset indices after filtering
                np.save(
                    dataset_indices_path,
                    self.dataset.active_indices,
                )
            g = torch.Generator()
            g = g.set_state(self.dataloader.generator.get_state())
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=None,
                shuffle=True,
                generator=g,
            )
            self.data_generator = enumerate(self.dataloader)
            return None

        # NOTE: no need to ignore ids during recover, because model workers will do so
        data_id = cur_sample.ids[0]
        if self.__recover_run and data_id in self.__recover_info.hash_vals_to_ignore:
            self.__recover_info.hash_vals_to_ignore.remove(data_id)
            return None
        assert data_id not in self.rollout_tasks
        return cur_sample

    async def allocate_new_rollout(self, qid) -> bool:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{self.gserver_manager_addr}/allocate_rollout",
                json=dict(qid=qid),
                timeout=ClientTimeout(
                    total=self.config.rollout_request_timeout,
                    sock_connect=self.config.rollout_request_timeout,
                ),
            ) as resp:
                resp.raise_for_status()
                res = await resp.json()
                if not res["success"]:
                    logger.debug(
                        f"Cannot allocate new rollout because: {res['reason']}"
                    )
                return res["success"]

    async def _poll_async(self):
        # Lazily initializing dataset to avoid over long configuration time.
        if self.data_generator is None:
            tik = time.perf_counter()
            logger.info(f"Rollout worker {self.worker_index} making datasets..")
            self.make_datasets()
            logger.info(
                f"Rollout worker {self.worker_index} finishes making datasets. "
                f"Time consumed: {time.perf_counter() - tik}s"
            )

        # Check experiment finish.
        name = names.experiment_status(
            constants.experiment_name(), constants.trial_name()
        )
        try:
            exp_status = name_resolve.wait(name, timeout=300)
            if exp_status != str(ExpStatus.RUNNING):
                self.exit()
                return PollResult(0, 0)
        except TimeoutError:
            raise TimeoutError(
                f"Waiting for experiment status timeout. "
                "This indicates that the master worker is not running. Exit the worker."
            )

        if self.push_stream is None:
            # Initialize stream after configure to ensure that puller names have been written.
            self.push_stream = NameResolvingZmqPusher(
                self.experiment_name,
                self.trial_name,
                pusher_index=self.worker_index,
                pusher_cnt=self.worker_count,
            )

        if self.gserver_manager_addr is None:
            name = names.gen_server_manager(self.experiment_name, self.trial_name)
            self.gserver_manager_addr = name_resolve.wait(name)

        # Create new trajectory collection tasks.
        # Load only one data in each poll to avoid over consumption.
        if self._cur_data is None:
            self._cur_data = self.load_next_data()

        if self._cur_data is not None:
            data = self._cur_data
            qid = data.ids[0]
            can_rollout = await self.allocate_new_rollout(qid)
            if can_rollout:
                assert qid not in self.act_queues
                self.act_queues[qid] = asyncio.Queue(1024)

                task = asyncio.create_task(self.rollout_task(qid, data))
                assert qid not in self.rollout_tasks
                self.rollout_tasks[qid] = task

                self._cur_data = None

                self.rollout_stat.submitted += 1
                self.rollout_stat.running += 1
                logger.debug(
                    f"Submit a new rollout for qid {qid}. "
                    f"Submit: {self.rollout_stat.submitted}, "
                    f"running: {self.rollout_stat.running}, "
                    f"accepted: {self.rollout_stat.accepted}."
                )

        # Run rollouts and wait
        done, *_ = await asyncio.gather(
            self.poll_rollout_task(),
            self.poll_queue_dispatch_task(),
            self.poll_inference_task(),
        )

        # Process done tasks.
        batch_count = sample_count = 0
        for task in done:
            qid, trajs = await task
            trajs: List[SequenceSample]
            assert len(set(traj.ids[0] for traj in trajs)) == len(trajs), [
                traj.ids[0] for traj in trajs
            ]
            self.rollout_tasks.pop(qid)
            self.act_queues.pop(qid)

            self.rollout_stat.running -= 1

            accepted = False
            if len(trajs) > 0:
                accepted = True
                self.push_stream.push([traj.as_json_compatible() for traj in trajs])
                self.rollout_stat.accepted += 1

            n_tokens = 0
            for traj in trajs:
                seqlens = [sum(datapack.flat2d(ss)) for ss in traj.seqlens.values()]
                n_tokens += max(seqlens)
            info = dict(qid=qid, accepted=accepted, n_tokens=n_tokens)
            async with aiohttp.ClientSession(
                f"http://{self.gserver_manager_addr}"
            ) as session:
                async with session.post(
                    "/finish_rollout",
                    json=info,
                    timeout=ClientTimeout(
                        total=self.config.rollout_request_timeout,
                        sock_connect=self.config.rollout_request_timeout,
                    ),
                ) as resp:
                    resp.raise_for_status()
                    assert (await resp.json())["success"]
            logger.debug(
                f"Finish rollout for qid {qid}. "
                f"Submit: {self.rollout_stat.submitted}, "
                f"running: {self.rollout_stat.running}, "
                f"accepted: {self.rollout_stat.accepted}."
            )

            for traj in trajs:
                batch_count += traj.bs
                sample_count += max(
                    [sum(datapack.flat2d(slens)) for slens in traj.seqlens.values()]
                )

        return PollResult(batch_count, sample_count)

    async def rollout_task(self, qid, data):
        return qid, await self.agent.collect_trajectory(
            env=self.env,
            prompt=data,
            act_queue=self.act_queues[qid],
            obs_queue=self.rollout_request_queue,
        )

    async def poll_inference_task(self):
        await self.inference_maker.run_step()

    async def poll_rollout_task(self):
        tasks = list(self.rollout_tasks.values())
        done = []
        if tasks:
            done, _ = await asyncio.wait(
                tasks,
                timeout=ROLLOUT_POLL_WAIT_TIME,
                return_when=asyncio.FIRST_COMPLETED,
            )
        return done

    async def poll_queue_dispatch_task(self):
        for _ in range(20):
            try:
                resp = self.rollout_response_queue.get_nowait()
                self.act_queues[resp.qid].put_nowait(resp)
            except QueueEmpty:
                await asyncio.sleep(0.02)

    async def _exit_async_tasks(self):
        for task in self.rollout_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _exit_hook(self, exit_status):
        if self.push_stream is not None:
            self.push_stream.close()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._exit_async_tasks())
