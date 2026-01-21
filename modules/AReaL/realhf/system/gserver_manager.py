# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
import asyncio
import os
import shutil
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import aiohttp
import numpy as np

from realhf.api.core.model_api import GenReqMeta, GenRespMeta, ModelVersionReq
from realhf.api.core.system_api import ExpStatus
from realhf.api.core.system_api import GserverManager as GserverManagerConfig
from realhf.base import constants, logging, name_resolve, names, network, recover
from realhf.base.monitor import RolloutStat
from realhf.system.worker_base import AsyncWorker, PollResult, Worker

logger = logging.getLogger("Generation Manager", "system")

STALENESS_WARNED = defaultdict(lambda: False)


@dataclass
class AllocateRolloutInput:
    qid: str


class GserverManager(Worker):
    """This worker has the following functionalities:
    1. As a router, it schedules generation requests and returns the
       best server urls to clients for submitting generation requests.
    2. It manages the weight update requests of generation servers.
       The weight update manager must be unique in each experiment.

    This is currently a hack usage of SGLang. We can integrate the
    functionalities into sgl-router and srt in the future.
    """

    def _configure(self, config: GserverManagerConfig):
        self.config = config
        self.model_name = config.model_name

        assert self.config.worker_info.worker_count == 1

        self.threading_lock = threading.Lock()
        self.rollout_stat = RolloutStat()

        self.schedule_policy = config.schedule_policy

        self._last_param_realloc_step = 0

        self._qid_to_server_url = {}

        self._server_token_usage = defaultdict(float)
        self._server_request_counts = defaultdict(int)

        self._last_thpt_output_time = time.time()
        self._gen_tokens = 0

        self.experiment_name = config.worker_info.experiment_name
        self.trial_name = config.worker_info.trial_name

        # manager server
        self.manager_http_server = None
        self.thread = None

        self.server_urls = []

        # recover info
        self.__recover_run, self.__recover_info = recover.load_recover_info(self.args)
        if self.__recover_run:
            # update weights will be automatically triggered upon the first schedule_request
            # self._last_param_realloc_step will also be updated
            name = names.model_version(
                constants.experiment_name(),
                constants.trial_name(),
                self.model_name.role,
            )
            name_resolve.add(name, self.__recover_info.last_step_info.global_step)

            self._loaded_recover_weights = False
            hist_rollouts = (
                self.config.train_batch_size
                * self.__recover_info.last_step_info.global_step
            )
            self.rollout_stat.submitted = hist_rollouts
            self.rollout_stat.accepted = hist_rollouts

        return config.worker_info

    def _discover_servers(self, n_servers: int, timeout: int = 300) -> List[str]:
        logger.info(f"Waiting for {n_servers} generation servers...")
        name = names.gen_servers(self.experiment_name, self.trial_name)
        cnt = 0
        while len(name_resolve.find_subtree(name)) < n_servers:
            time.sleep(1)
            cnt += 1
            if cnt >= timeout:
                raise TimeoutError("Waiting generation servers timeout.")
        urls = name_resolve.get_subtree(name)
        assert len(set(urls)) == len(urls), (len(urls), len(set(urls)), urls)
        return urls

    def _get_recover_ckpt_path(self, role: str):
        assert self.__recover_run
        epoch = self.__recover_info.last_step_info.epoch + 1
        epochstep = self.__recover_info.last_step_info.epoch_step + 1
        globalstep = self.__recover_info.last_step_info.global_step + 1
        save_root = constants.get_save_path(self.args)
        role_path = os.path.join(save_root, role)
        if not os.path.exists(role_path):
            raise RuntimeError(
                f"Guessed checkpoint path {role_path} does not exist. "
                "Skip loading checkpoints in the recovered run."
            )
        model_path = os.path.join(
            role_path,
            f"epoch{epoch}epochstep{epochstep}globalstep{globalstep}",
        )
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Guessed checkpoint path {model_path} does not exist. "
                "Skip loading checkpoints in the recovered run."
            )
        return model_path

    def check_new_params(self) -> str | None:
        name = names.model_version(
            constants.experiment_name(),
            constants.trial_name(),
            self.model_name.role,
        )
        try:
            realloc_version = int(name_resolve.get(name))
        except name_resolve.NameEntryNotFoundError:
            return None

        # Update the model weights after parameter realloction.
        if realloc_version > self._last_param_realloc_step:
            if self.__recover_run and not self._loaded_recover_weights:
                realloc_dir = self._get_recover_ckpt_path(self.model_name.role)
                self._loaded_recover_weights = True
            else:
                realloc_dir = os.path.join(
                    constants.get_param_realloc_path(self.args),
                    self.model_name.role,
                    str(realloc_version),
                )
            self._last_param_realloc_step = realloc_version
            return realloc_dir

        return None

    async def flush_requests_and_update_weights(self, server_url, new_param_path):
        async with aiohttp.ClientSession(
            server_url,
            timeout=aiohttp.ClientTimeout(
                total=self.config.flush_request_timeout,
                sock_connect=self.config.flush_request_timeout,
            ),
        ) as session:
            (await session.post("/pause_generation")).raise_for_status()
            async with session.post(
                f"/update_weights_from_disk",
                json=dict(model_path=new_param_path),
            ) as resp:
                resp.raise_for_status()
                assert (await resp.json())["success"]
            (await session.post("/continue_generation")).raise_for_status()

    def _round_robin_schedule(self, req_meta: GenReqMeta) -> int:
        if not hasattr(self, "round_robin_idx"):
            self.round_robin_idx = 0
        r = self.round_robin_idx
        self.round_robin_idx += 1
        self.round_robin_idx %= self.config.n_servers
        return r

    def _least_requests_schedule(self, req_meta: GenReqMeta) -> int:
        counts = [
            self._server_request_counts[server_url] for server_url in self.server_urls
        ]
        return int(np.argmin(counts))

    def _least_token_usage_schedule(self, req_meta: GenReqMeta) -> int:
        url = min(self.server_urls, key=lambda k: self._server_token_usage[k])
        return self.server_urls.index(url)

    def _poll(self):
        if not self.thread:
            # Find addresses of generation servers
            self.server_urls = self._discover_servers(self.config.n_servers)
            self.thread = threading.Thread(
                target=self._run_routing_service, daemon=True
            )
            self.thread.start()
            time.sleep(3)  # Wait briefly for server to start
            # Write address for clients
            name = names.gen_server_manager(self.experiment_name, self.trial_name)
            name_resolve.add(name, self.manager_addr)
            logger.info(
                f"GserverManager HTTP service started in background thread at {self.manager_addr}"
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

        # Check weights.
        with self.threading_lock:
            # FIXME: we create a sync point across servers to update weights,
            # but we can acutally update them individually
            new_param_path = self.check_new_params()
            if new_param_path is not None:

                def _run_in_thread():
                    # Create a new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    tasks = [
                        self.flush_requests_and_update_weights(base_url, new_param_path)
                        for base_url in self.server_urls
                    ]
                    try:
                        return new_loop.run_until_complete(asyncio.gather(*tasks))
                    finally:
                        new_loop.close()

                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor() as executor:
                    future = executor.submit(_run_in_thread)
                    _ = future.result()
                logger.info(f"Generaion server updated weights from: {new_param_path}")

        if self.schedule_policy == "least_token_usage":
            tasks = [
                self._get_server_token_usage(server_url)
                for server_url in self.server_urls
            ]
            loop = asyncio.get_event_loop()
            token_usages = loop.run_until_complete(asyncio.gather(*tasks))
            with self.threading_lock:
                for server_url, token_usage in zip(self.server_urls, token_usages):
                    self._server_token_usage[server_url] = token_usage

        if time.time() - self._last_thpt_output_time > 30:
            interval = time.time() - self._last_thpt_output_time
            logger.info(
                f"Generation throughput: {self._gen_tokens / interval:.2f} tokens/s"
            )
            self._last_thpt_output_time = time.time()
            self._gen_tokens = 0

        # clear old weights
        realloc_root = os.path.join(
            constants.get_param_realloc_path(self.args),
            self.model_name.role,
        )
        if os.path.exists(realloc_root):
            for realloc_version in os.listdir(realloc_root):
                # Lock-free is safe here.
                # Remain one checkpoint for recover.
                if (
                    os.path.isdir(os.path.join(realloc_root, realloc_version))
                    and int(realloc_version) < self._last_param_realloc_step - 1
                ):
                    shutil.rmtree(os.path.join(realloc_root, realloc_version))
                    logger.info(
                        f"Removed previous reallocated "
                        f"checkpoint: {os.path.join(realloc_root, realloc_version)}"
                    )

        time.sleep(5)

        return PollResult(0, 0)

    async def _get_server_token_usage(self, server_url):
        async with aiohttp.ClientSession(
            server_url,
            timeout=aiohttp.ClientTimeout(
                total=self.config.flush_request_timeout,
                sock_connect=self.config.flush_request_timeout,
            ),
        ) as session:
            async with session.get("/metrics") as resp:
                resp.raise_for_status()
                text = await resp.text()
                for l in text.split("\n"):
                    if l.startswith("sglang:num_used_tokens"):
                        return float(l.split(" ")[1])
        raise RuntimeError(f"Failed to get token usage metrics from {server_url}")

    async def _get_server_num_running_requests(self, server_url):
        async with aiohttp.ClientSession(
            server_url,
            timeout=aiohttp.ClientTimeout(
                total=self.config.flush_request_timeout,
                sock_connect=self.config.flush_request_timeout,
            ),
        ) as session:
            async with session.get(f"/metrics") as resp:
                resp.raise_for_status()
                text = await resp.text()
                for line in text.split("\n"):
                    if line.startswith("sglang:num_running_reqs"):
                        return float(line.split(" ")[1])
        raise RuntimeError(
            f"Failed to get num running requests metrics from {server_url}"
        )

    def get_training_sample_cnt(self):
        name = names.training_samples(self.experiment_name, self.trial_name)
        try:
            return int(name_resolve.get(name))
        except name_resolve.NameEntryNotFoundError:
            return 0

    def is_staled(self):
        # Use counter written by the trainer, local counter is inaccurate
        global_sample_cnt = self.get_training_sample_cnt() + self.rollout_stat.running
        expected_version = global_sample_cnt // self.config.train_batch_size
        version = self._last_param_realloc_step
        staled = expected_version > self.config.max_head_offpolicyness + version
        global STALENESS_WARNED
        if staled and not STALENESS_WARNED[version]:
            logger.warning(
                f"expected version ({expected_version}) = "
                f"global sample cnt ({global_sample_cnt}) // batch size ({self.config.train_batch_size}), "
                f"current latest version {version}, "
                f"offpolicyness {self.config.max_head_offpolicyness}. Staled? {staled}"
            )
            STALENESS_WARNED[version] = True
        return staled

    def _run_routing_service(self):
        """Expose an API for clients to find the destination server."""
        import uvicorn
        from fastapi import FastAPI

        self.app = FastAPI()

        @self.app.post("/schedule_request")
        async def schedule_request(req_meta: GenReqMeta):
            with self.threading_lock:
                if (
                    req_meta.previous_server_url
                    and req_meta.previous_version == self._last_param_realloc_step
                ):
                    return dict(
                        url=req_meta.previous_server_url,
                        version=req_meta.previous_version,
                    )

                if self.schedule_policy == "round_robin":
                    server_idx = self._round_robin_schedule(req_meta)
                elif self.schedule_policy == "least_token_usage":
                    server_idx = self._least_token_usage_schedule(req_meta)
                elif self.schedule_policy == "least_requests":
                    server_idx = self._least_requests_schedule(req_meta)
                else:
                    raise NotImplementedError(
                        f"Unknown schedule policy {self.schedule_policy}"
                    )

                server_url = self.server_urls[server_idx]
                # qid prompt (n samples) use the same dst server
                self._qid_to_server_url[req_meta.qid] = server_url
                self._server_request_counts[server_url] += 1
                self._server_token_usage[server_url] += (
                    req_meta.prompt_len
                    + req_meta.new_token_budget * req_meta.group_size * 0.4
                )

                version = self._last_param_realloc_step
            return dict(url=server_url, version=version)

        @self.app.post("/get_model_version")
        async def get_model_version(req: ModelVersionReq):
            with self.threading_lock:
                # FIXME: we may have different versions for different servers
                version = self._last_param_realloc_step
            return dict(version=version)

        @self.app.post("/allocate_rollout")
        async def allocate_rollout(req: AllocateRolloutInput):
            with self.threading_lock:
                has_capacity = (
                    self.rollout_stat.running < self.config.max_concurrent_rollouts
                )
                is_staled = self.is_staled()
                reason = ""
                if has_capacity and not is_staled:
                    self.rollout_stat.submitted += 1
                    self.rollout_stat.running += 1
                    logger.debug(
                        f"Allocate rollout for qid {req.qid}. "
                        f"Submitted: {self.rollout_stat.submitted}, "
                        f"running: {self.rollout_stat.running}, "
                        f"accepted: {self.rollout_stat.accepted}."
                    )
                    return dict(success=True, reason=reason)
                else:
                    if not has_capacity:
                        reason += f"capacity: {self.rollout_stat.running} >= {self.config.max_concurrent_rollouts}"
                    if is_staled:
                        global_sample_cnt = (
                            self.get_training_sample_cnt() + self.rollout_stat.running
                        )
                        expected_version = (
                            global_sample_cnt // self.config.train_batch_size
                        )
                        version = self._last_param_realloc_step
                        reason += (
                            f" and staled: expected version ({expected_version}) = "
                            f"global sample cnt ({global_sample_cnt}) // batch size ({self.config.train_batch_size}), "
                            f"current latest version {version}, "
                            f"offpolicyness {self.config.max_head_offpolicyness}."
                        )
                    return dict(success=False, reason=reason)

        @self.app.post("/finish_rollout")
        async def finish_rollout(resp_meta: GenRespMeta):
            with self.threading_lock:
                server_url = self._qid_to_server_url[resp_meta.qid]
                self._server_request_counts[server_url] -= 1
                assert (
                    self._server_request_counts[server_url] >= 0
                ), "server request count < 0"
                self._qid_to_server_url.pop(resp_meta.qid)
                self._gen_tokens += resp_meta.n_tokens
                self.rollout_stat.running -= 1
                if resp_meta.accepted:
                    self.rollout_stat.accepted += 1
                logger.debug(
                    f"Finish rollout for qid {resp_meta.qid}. "
                    f"Submit: {self.rollout_stat.submitted}, "
                    f"running: {self.rollout_stat.running}, "
                    f"accepted: {self.rollout_stat.accepted}"
                )
                return dict(success=True)

        port = network.find_free_port(
            experiment_name=self.experiment_name,
            trial_name=self.trial_name,
            lockfile_root=os.path.join(constants.get_cache_path(self.args), "ports"),
        )
        self.manager_addr = f"{network.gethostip()}:{port}"

        config = uvicorn.Config(
            self.app,
            host=self.manager_addr.split(":")[0],
            port=int(self.manager_addr.split(":")[1]),
            log_level="warning",
        )
        self.manager_http_server = uvicorn.Server(config)
        self.manager_http_server.run()

    def _exit_hook(self, exit_status):
        if self.manager_http_server:
            self.manager_http_server.should_exit = True
        if self.thread:
            self.thread.join(timeout=3)
        logger.info("Server stopped")
