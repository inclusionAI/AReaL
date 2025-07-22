# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import time
from asyncio.queues import QueueEmpty
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, Hashable, List

import aiohttp
from aiohttp.client import ClientTimeout
from transformers import PreTrainedTokenizerFast

from realhf.api.cli_args import GenerationHyperparameters
from realhf.api.core.model_api import (
    APIGenerateInput,
    APIGenerateOutput,
    BundledGenerationOutputs,
    GenReqMeta,
)
from realhf.base import constants, logging, name_resolve, names

logger = logging.getLogger(__name__)

GENERATION_POLL_WAIT_TIME = 0.05


class PartialRolloutManager:
    """Manages the partial rollout for a client.

    It will submit generation requests in chunks, i.e.,
    generating at most `new_tokens_per_chunk` tokens each time.
    In this way, we can reduce the overhead of flushing all requests
    upon model weights update.

    This is a hack usage. We don't need it if the server can pause
    requests, update weights, and recompute kv caches at any time.
    """

    def __init__(
        self,
        worker_index: int,
        request_queue: asyncio.Queue,
        reply_queue: asyncio.Queue,
        new_tokens_per_chunk: int,
        tokenizer: PreTrainedTokenizerFast,
        timeout: int,
    ):
        self.worker_index = worker_index

        # qid -> {group_idx -> aiohttp Task}
        self.gen_requests: Dict[Hashable, Dict[int, asyncio.Task]]
        self.gen_requests = defaultdict(dict)

        # NOTE: Grouped generations are managed separately. Store early returned
        # answers in this cache and pop the result when the whole group is done.
        self.gen_cache: Dict[Hashable, Dict[int, APIGenerateOutput]]
        self.gen_cache = defaultdict(dict)

        self.tokenizer = tokenizer

        self.request_queue = request_queue
        self.reply_queue = reply_queue

        self.new_tokens_per_chunk = new_tokens_per_chunk

        self.gserver_manager_addr = None
        self.timeout = timeout

    async def _schedule_request(self, req_meta: GenReqMeta):
        if self.gserver_manager_addr is None:
            # Get the address of gserver manager to schedule requests
            name = names.gen_server_manager(
                constants.experiment_name(), constants.trial_name()
            )
            self.gserver_manager_addr = name_resolve.wait(name, timeout=300)
            time.sleep(1)  # Wait for the server to start
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{self.gserver_manager_addr}/schedule_request",
                json=asdict(req_meta),
                timeout=ClientTimeout(total=self.timeout, sock_connect=self.timeout),
            ) as response:
                response.raise_for_status()
                res = await response.json()
                return res

    def get_num_gen_requests(self):
        return len(self.gen_requests)

    async def _run_gen(
        self,
        url,
        qid,
        group_idx,
        prompt_ids,
        input_ids,
        prev_logprobs,
        version_start,
        cur_server_version,
        raw_gconfig,
    ):
        from realhf.impl.model.backend.sglang import SGLangAPIClient

        max_new_tokens = min(raw_gconfig.max_new_tokens, self.new_tokens_per_chunk)
        max_new_tokens = min(
            max_new_tokens,
            raw_gconfig.max_new_tokens - len(input_ids) + len(prompt_ids),
        )
        gconfig = raw_gconfig.new(
            n=1,
            max_new_tokens=max_new_tokens,
        )
        assert self.tokenizer.pad_token_id is not None
        assert self.tokenizer.eos_token_id is not None
        # Don't need to request updating weights
        async with SGLangAPIClient(
            generate_url=f"{url}/generate", update_weights_url=""
        ) as api_client:
            res = await api_client.async_add_generate_request(
                APIGenerateInput(
                    qid=qid,
                    prompt_ids=prompt_ids,
                    input_ids=input_ids,
                    gconfig=gconfig,
                    stop_token_ids=[
                        self.tokenizer.pad_token_id,
                        self.tokenizer.eos_token_id,
                    ],
                    return_logprob=True,
                    version_start=version_start,
                    prev_logprobs=prev_logprobs,
                    metadata=dict(
                        group_idx=group_idx,
                        raw_gconfig=raw_gconfig,
                        server_url=url,
                        version=cur_server_version,
                    ),
                ),
                stream=False,
            )
            res.version_end = [cur_server_version for _ in range(res.group_size)]
            return res

    async def _issue_generation(
        self,
        url: str,
        qid: Hashable,
        group_idx: int,
        prompt_ids: List[int],
        input_ids: List[int],
        prev_logprobs: List[float],
        version_start: int,
        raw_gconfig: GenerationHyperparameters,
        cur_server_version: int,
    ):
        """Issue a generation request.

        `input_ids` can be a partial prefix and longer than `prompt_ids`.
        If model weights are updated, the KV cache will be refreshed,
        otherwise the server will reuse the radix cache with no additional overhead.
        """

        task = asyncio.create_task(
            self._run_gen(
                url,
                qid,
                group_idx,
                prompt_ids,
                input_ids,
                prev_logprobs,
                version_start=version_start,
                cur_server_version=cur_server_version,
                raw_gconfig=raw_gconfig,
            )
        )
        self.gen_requests[qid][group_idx] = task
        await asyncio.sleep(0)

    async def refresh_generation(self):
        tasks = []
        for group_requests in self.gen_requests.values():
            tasks += list(group_requests.values())

        done = []
        if tasks:
            # No new checkpoint available, try to wait for the next complete sequence
            done, _ = await asyncio.wait(
                tasks,
                timeout=GENERATION_POLL_WAIT_TIME,
                return_when=asyncio.FIRST_COMPLETED,
            )

        for task in done:
            s: APIGenerateOutput = await task
            group_idx = s.metadata["group_idx"]
            raw_gconfig = s.metadata["raw_gconfig"]
            previous_version = s.metadata["version"]

            assert s.group_size == 1
            no_eos = s.no_eos[0]
            gen_len = s.gen_lens[0]

            self.gen_requests[s.qid].pop(group_idx)
            if len(self.gen_requests[s.qid]) == 0:
                self.gen_requests.pop(s.qid)

            if no_eos and gen_len < raw_gconfig.max_new_tokens:
                # Unfinished request due to chunked generation.
                # Send it back to continue.
                req_meta = GenReqMeta(
                    qid=s.qid,
                    prompt_len=s.prompt_len,
                    group_size=raw_gconfig.n,
                    new_token_budget=raw_gconfig.max_new_tokens,
                    predicted_new_tokens=None,
                    previous_server_url=s.metadata["server_url"],
                    previous_version=previous_version,
                )
                info = await self._schedule_request(req_meta)
                cur_version = info["version"]
                server_url = info["url"]

                if len(s.output_logprobs) > 0:
                    prev_logprobs = s.prev_logprobs + s.output_logprobs[0]
                else:
                    prev_logprobs = s.prev_logprobs
                    if prev_logprobs is None:
                        prev_logprobs = []
                await self._issue_generation(
                    server_url,
                    s.qid,
                    group_idx,
                    s.prompt_ids,
                    s.input_ids + s.output_ids[0],
                    version_start=s.version_start,
                    prev_logprobs=prev_logprobs,
                    raw_gconfig=raw_gconfig,
                    cur_server_version=cur_version,
                )
            else:
                # Generation finishes. Save to cache for later fetching.
                self.gen_cache[s.qid][group_idx] = s
                if len(self.gen_cache[s.qid]) >= raw_gconfig.n:
                    gen_results = self.gen_cache.pop(s.qid)
                    output = BundledGenerationOutputs.from_api_outputs(
                        list(gen_results.values())
                    )
                    self.reply_queue.put_nowait(output)

    async def poll_fresh_requests_task(self):
        for _ in range(8):
            try:
                qid, prompt_token_ids, gconfig = self.request_queue.get_nowait()
                req_meta = GenReqMeta(
                    qid=qid,
                    prompt_len=len(prompt_token_ids),
                    group_size=gconfig.n,
                    new_token_budget=gconfig.max_new_tokens,
                    predicted_new_tokens=None,
                )
                dst_server_info = await self._schedule_request(req_meta)

                for group_idx in range(gconfig.n):
                    await self._issue_generation(
                        dst_server_info["url"],
                        qid,
                        group_idx,
                        prompt_token_ids,
                        prompt_token_ids,
                        version_start=dst_server_info["version"],
                        prev_logprobs=[],
                        raw_gconfig=gconfig,
                        cur_server_version=dst_server_info["version"],
                    )
            except QueueEmpty:
                break

    async def poll_old_requests_task(self):
        for _ in range(8):
            await self.refresh_generation()

    async def run_step(self):

        await asyncio.gather(
            self.poll_fresh_requests_task(),
            self.poll_old_requests_task(),
        )
