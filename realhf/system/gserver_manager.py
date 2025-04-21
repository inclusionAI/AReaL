# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").
import asyncio
import os
import shutil
import threading
import time
from collections import defaultdict
from typing import List

import aiohttp

from realhf.api.core.model_api import GenReqMeta, GenRespMeta, ModelVersionReq
from realhf.api.core.system_api import GserverManager as GserverManagerConfig
from realhf.base import constants, logging, name_resolve, names, network, recover
from realhf.system.worker_base import AsyncWorker, PollResult, Worker

logger = logging.getLogger("Generation Manager", "colored")

STALENESS_WARNED = defaultdict(lambda: False)


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

        self.async_lock = asyncio.Lock()
        self.threading_lock = threading.Lock()
        self.n_total_rollouts = 0
        self.n_running_rollouts = 0
        self.accepted_rollouts = 0

        self.schedule_policy = config.schedule_policy

        self._last_param_realloc_step = 0

        self.experiment_name = config.worker_info.experiment_name
        self.trial_name = config.worker_info.trial_name

        # manager server
        self.server = None
        self.thread = None

        # recover info
        self.__recover_run, self.__recover_info = recover.load_recover_info()
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
            self.n_total_rollouts = self.accepted_rollouts = (
                self.config.train_batch_size
                * self.__recover_info.last_step_info.global_step
            )

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
        assert len(set(urls)) == len(urls), urls
        return urls

    def _get_recover_ckpt_path(self, role: str):
        assert self.__recover_run
        epoch = self.__recover_info.last_step_info.epoch + 1
        epochstep = self.__recover_info.last_step_info.epoch_step + 1
        globalstep = self.__recover_info.last_step_info.global_step + 1
        save_root = os.path.join(
            constants.MODEL_SAVE_ROOT,
            constants.experiment_name(),
            constants.trial_name(),
        )
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
                    constants.PARAM_REALLOC_PATH,
                    constants.experiment_name(),
                    constants.trial_name(),
                    self.model_name.role,
                    str(realloc_version),
                )
            self._last_param_realloc_step = realloc_version
            return realloc_dir

        return None

    async def flush_requests_and_update_weights(
        self, server_url, new_param_path, update_weights_retries=5
    ):
        # HACK: urls are designed for SGLang
        server_index = self.server_urls.index(server_url)
        async with aiohttp.ClientSession(server_url) as session:
            running_requests = None
            tik = time.perf_counter()
            while running_requests is None or running_requests > 0:
                if time.perf_counter() - tik > self.config.flush_request_timeout:
                    raise RuntimeError(
                        f"Waiting for flush requests failed. {running_requests} requests "
                        f"remain after {self.config.flush_request_timeout} secs waiting. "
                        f"Please try to reduce `new_tokens_per_chunk`."
                    )
                if running_requests is not None and running_requests > 0:
                    logger.info(
                        f"Waiting for {running_requests} requests on gen server {server_index}... "
                        f"Time taken so far: {time.perf_counter() - tik:.4f}s"
                    )
                    await asyncio.sleep(0.5)
                async with session.get(f"/metrics") as resp:
                    resp.raise_for_status()
                    text = await resp.text()
                    for line in text.split("\n"):
                        if line.startswith("sglang:num_running_reqs"):
                            running_requests = float(line.split(" ")[1])
                            break

            success = False
            for _ in range(update_weights_retries):
                async with session.post(
                    f"/update_weights_from_disk",
                    json=dict(model_path=new_param_path),
                ) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        success = res["success"]
                        if success:
                            return
                        logger.warning(
                            f"Update weights failed: {res['message']}. Retrying."
                        )
                    logger.warning(f"Update weights failed: {resp.reason}. Retrying.")
                time.sleep(0.1)
            raise RuntimeError("Update weights failed.")

    def _round_robin_schedule(self, req_meta: GenReqMeta) -> int:
        if not hasattr(self, "round_robin_idx"):
            self.round_robin_idx = 0
        r = self.round_robin_idx
        self.round_robin_idx += 1
        self.round_robin_idx %= self.config.n_servers
        return r

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

        # Check weights.
        with self.threading_lock:
            # FIXME: we create a sync point across servers to update weights,
            # but we can acutally update them individually
            new_param_path = self.check_new_params()
            if new_param_path is not None:
                tasks = [
                    self.flush_requests_and_update_weights(base_url, new_param_path)
                    for base_url in self.server_urls
                ]
                loop = asyncio.get_event_loop()
                loop.run_until_complete(asyncio.gather(*tasks))
                logger.info(f"Generaion server updated weights from: {new_param_path}")

        # clear old weights
        realloc_root = os.path.join(
            constants.PARAM_REALLOC_PATH,
            constants.experiment_name(),
            constants.trial_name(),
            self.model_name.role,
        )
        if os.path.exists(realloc_root):
            for realloc_version in os.listdir(realloc_root):
                if (
                    os.path.isdir(os.path.join(realloc_root, realloc_version))
                    and int(realloc_version) < self._last_param_realloc_step
                ):
                    shutil.rmtree(os.path.join(realloc_root, realloc_version))
                    logger.info(
                        f"Removed previous reallocated "
                        f"checkpoint: {os.path.join(realloc_root, realloc_version)}"
                    )

        # TODO: we may want to update server status
        # in the main thread.

        time.sleep(1)

        return PollResult(0, 0)

    async def is_staled(self):
        global_sample_cnt = self.n_total_rollouts
        expected_version = global_sample_cnt // self.config.train_batch_size
        staled = (
            expected_version
            > self.config.max_head_offpolicyness + self._last_param_realloc_step
        )
        global STALENESS_WARNED
        if staled and not STALENESS_WARNED[self._last_param_realloc_step]:
            logger.warning(
                f"expected version ({expected_version}) = "
                f"global sample cnt ({global_sample_cnt}) // batch size ({self.config.train_batch_size}), "
                f"current version {self._last_param_realloc_step}, "
                f"offpolicyness {self.config.max_head_offpolicyness}. Staled? {staled}"
            )
            STALENESS_WARNED[self._last_param_realloc_step] = True
        return staled

    def _run_routing_service(self):
        """Expose an API for clients to find the destination server."""
        import uvicorn
        from fastapi import FastAPI

        self.app = FastAPI()

        @self.app.post("/schedule_request")
        async def schedule_request(req_meta: GenReqMeta):
            with self.threading_lock:
                async with self.async_lock:
                    version = self._last_param_realloc_step
                    # FIXME: We only implement a round-robin scheduler that
                    # ignores server status and request metadata
                    server_idx = self._round_robin_schedule(req_meta)
            return dict(url=self.server_urls[server_idx], version=max(0, version))

        @self.app.post("/get_model_version")
        async def get_model_version(req: ModelVersionReq):
            with self.threading_lock:
                async with self.async_lock:
                    # FIXME: we may have different versions for different servers
                    version = self._last_param_realloc_step
            return dict(version=version)

        @self.app.get("/allocate_rollout")
        async def allocate_rollout():
            with self.threading_lock:
                async with self.async_lock:
                    has_capacity = (
                        self.n_running_rollouts < self.config.max_concurrent_rollouts
                    )
                    is_staled = await self.is_staled()
                    reason = ""
                    if has_capacity and not is_staled:
                        self.n_running_rollouts += 1
                        self.n_total_rollouts += 1
                        return dict(success=True, reason=reason)
                    else:
                        if not has_capacity:
                            reason += f"capacity: {self.n_running_rollouts} >= {self.config.max_concurrent_rollouts}"
                        if is_staled:
                            global_sample_cnt = self.n_total_rollouts
                            expected_version = (
                                global_sample_cnt // self.config.train_batch_size
                            )
                            reason += (
                                f" and staled: expected version ({expected_version}) = "
                                f"global sample cnt ({global_sample_cnt}) // batch size ({self.config.train_batch_size}), "
                                f"current version {self._last_param_realloc_step}, "
                                f"offpolicyness {self.config.max_head_offpolicyness}."
                            )
                        return dict(success=False, reason=reason)

        @self.app.post("/finish_rollout")
        async def finish_rollout(resp_meta: GenRespMeta):
            with self.threading_lock:
                async with self.async_lock:
                    self.n_running_rollouts -= 1
                    if resp_meta.accepted:
                        self.accepted_rollouts += 1
                    return dict(success=True)

        self.manager_addr = f"{network.gethostip()}:{network.find_free_port()}"

        config = uvicorn.Config(
            self.app,
            host=self.manager_addr.split(":")[0],
            port=int(self.manager_addr.split(":")[1]),
            log_level="warning",
        )
        self.server = uvicorn.Server(config)
        self.server.run()

    def _exit_hook(self, exit_status):
        if self.server:
            self.server.should_exit = True
        if self.thread:
            self.thread.join(timeout=3)
        logger.info("Server stopped")
