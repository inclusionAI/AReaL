# Copyright 2025 Ant Group Inc.

import asyncio
import dataclasses
import itertools
import json
import os
import time
import uuid
from collections import defaultdict
from typing import Dict, Hashable, List, Set, Tuple

from tensorboardX import SummaryWriter

import realhf.api.core.config as config_api
import realhf.api.core.data_api as data_api
import realhf.api.core.dfg as dfg
import realhf.api.core.system_api as config_pkg
import realhf.base.recover as recover
import realhf.system.request_reply_stream as request_reply_stream
from realhf.api.core.config import ModelName, ModelShardID
from realhf.api.core.model_api import ReaLModelConfig
from realhf.base import constants, logging, stats_tracker, topology
from realhf.system.buffer import AsyncIOSequenceBuffer
from realhf.system.flops_counter import FlopsCounter
from realhf.system.redistributor import RedistribPlanner, RedistribStep

logger = logging.getLogger(__name__, "system")
blogger = logging.getLogger("benchmark")


@dataclasses.dataclass
class RPCCorountineControl:
    # for counting the number of finished training steps
    # one training step corresponds to traversal of the whole DFG
    train_count: asyncio.Queue
    # For flushing requests
    topo_level_count: asyncio.Queue

    lock: asyncio.Lock
    # for training data management and data cleaning after each step
    ids_to_clear: Set[Hashable] = dataclasses.field(default_factory=set)
    flops_counter: FlopsCounter = dataclasses.field(default_factory=FlopsCounter)

    should_save: bool = False
    should_eval: bool = False
    should_ckpt: bool = False
    step_info: recover.StepInfo = dataclasses.field(default_factory=recover.StepInfo)

    # recover information
    used_hash_vals_this_epoch: List[int] = dataclasses.field(default_factory=list)


class ModelFunctionCall:
    def __init__(
        self,
        args,
        rpc: dfg.MFCDef,
        src_rpc: dfg.MFCDef,
        stream: request_reply_stream.NameResolvingRequestClient,
        msid2mwid: Dict[config_pkg.ModelShardID, int],
        model_topos: Dict[str, topology.ProcessTopology],
        model_configs: Dict[str, None | ReaLModelConfig],
        ctrl: RPCCorountineControl,
        buffers: List[AsyncIOSequenceBuffer],
        redistrib_planner: RedistribPlanner,
        summary_writer: SummaryWriter | None,
    ):

        self.args = args

        self.rpc = rpc
        self.src_rpc = src_rpc
        self.stream = stream

        self.n_model_workers = len(set(msid2mwid.values()))

        self.msid2mwid = msid2mwid
        self.model_topos = model_topos
        self.model_configs = model_configs

        self.mwid2msids = defaultdict(list)
        for msid, mwid in msid2mwid.items():
            self.mwid2msids[mwid].append(msid)

        self.rpc_ctrl = ctrl
        self.buffers = buffers
        self.redistrib_planner = redistrib_planner

        self.summary_writer = summary_writer

    @property
    def dp_size(self):
        return self.model_topos[self.rpc.model_name].get_dim("data")

    @property
    def pp_size(self):
        return self.model_topos[self.rpc.model_name].get_dim("pipe")

    def attach_payloads_with_hooks(
        self,
        payloads: Dict[config_api.ModelShardID, request_reply_stream.Payload],
        mwids: List[int],
        main_handlers: List[config_pkg.ModelShardID],
        hook_type: str,
    ) -> Tuple[Dict[config_api.ModelShardID, request_reply_stream.Payload], List[int]]:
        assert hook_type in ["pre", "post"], hook_type

        rpc = self.rpc
        model_topos = self.model_topos
        model_configs = self.model_configs

        main_mwids = set([self.msid2mwid[h] for h in main_handlers])
        for hook in getattr(rpc, f"_{hook_type}_hooks"):
            if isinstance(hook, dfg.ParamReallocHook):
                assert (hook.source is None) != (hook.target is None), hook
                if hook.source is None:
                    src_topo = model_topos[rpc.model_name]
                    dst_topo = model_topos[hook.target]
                    dst_config = model_configs[hook.target]
                    src_model_name, dst_model_name = rpc.model_name, hook.target
                    other_model_name = hook.target
                    other_topo = dst_topo
                else:
                    src_topo = model_topos[hook.source]
                    dst_topo = model_topos[rpc.model_name]
                    dst_config = model_configs[rpc.model_name]
                    src_model_name, dst_model_name = hook.source, rpc.model_name
                    other_model_name = hook.source
                    other_topo = src_topo

                ps_data = {
                    "from_model_name": src_model_name,
                    "to_model_name": dst_model_name,
                    "from_topo": src_topo,
                    "to_topo": dst_topo,
                    "to_model_config": dst_config,
                    "eta": hook.eta,
                }
                for h in main_handlers:
                    getattr(payloads[h], f"{hook_type}_hooks").append("param_realloc")
                    getattr(payloads[h], f"{hook_type}_hook_data").append(ps_data)
                other_handlers = [
                    config_api.ModelShardID.from_parallelism_rank(
                        other_model_name, other_topo, j
                    )
                    for j in range(other_topo.world_size())
                ]
                for h in other_handlers:
                    if self.msid2mwid[h] not in mwids:
                        payloads[h] = request_reply_stream.Payload(
                            handler=h,
                            handle_name="empty",
                        )
                        setattr(payloads[h], f"{hook_type}_hooks", ["param_realloc"])
                        setattr(payloads[h], f"{hook_type}_hook_data", [ps_data])
                        mwids.append(self.msid2mwid[h])
                    elif self.msid2mwid[h] not in main_mwids:
                        hh = next(
                            hh
                            for hh in payloads
                            if self.msid2mwid[hh] == self.msid2mwid[h]
                        )
                        getattr(payloads[hh], f"{hook_type}_hooks").append(
                            "param_realloc"
                        )
                        getattr(payloads[hh], f"{hook_type}_hook_data").append(ps_data)

            elif isinstance(hook, dfg.OffloadHook):
                for h in main_handlers:
                    getattr(payloads[h], f"{hook_type}_hooks").append("offload")
                    getattr(payloads[h], f"{hook_type}_hook_data").append(
                        dict(model_name=h.model_name)
                    )
            else:
                raise NotImplementedError(f"Unknown hook type: {hook}")
        return payloads, mwids

    def request(
        self,
        data_transfer_plan: List[RedistribStep],
        partitioned_ids: List[Hashable],
        meta_sample: data_api.SequenceSample,
        handlers: List[config_pkg.ModelShardID],
    ) -> Tuple[List[uuid.UUID], List[uuid.UUID]]:

        rpc = self.rpc
        ctrl = self.rpc_ctrl

        dt_data = {
            "target": rpc.model_name,
            "plan": [json.dumps(dataclasses.asdict(x)) for x in data_transfer_plan],
            "partitioned_ids": partitioned_ids,
            "handle_name": rpc.interface_type.value,
            "meta_sample": meta_sample,
            "partitioned_ids": partitioned_ids,
        }

        payloads = {
            handler: request_reply_stream.Payload(
                handler=handler,
                handle_name=rpc.interface_type.value,
                pre_hooks=["data_transfer"],
                pre_hook_data=[dt_data],
                data=rpc.name,
            )
            for handler in handlers
        }
        if ctrl.should_eval:
            for p in payloads.values():
                p.post_hooks.append("evaluate")
                p.post_hook_data.append(dict(model_name=rpc.model_name))
        if (
            ctrl.should_save or ctrl.should_ckpt
        ) and rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP:
            for p in payloads.values():
                p.post_hooks.append("save")
                save_dir = os.path.join(
                    constants.get_log_path(self.args),
                    rpc.model_name.role,
                    f"epoch{ctrl.step_info.epoch + 1}"
                    f"epochstep{ctrl.step_info.epoch_step + 1}"
                    f"globalstep{ctrl.step_info.global_step + 1}",
                )
                p.post_hook_data.append(
                    dict(
                        model_name=rpc.model_name,
                        save_dir=save_dir,
                        recover_only=not ctrl.should_save,
                    )
                )
        mwids = [self.msid2mwid[h] for h in handlers]
        assert len(mwids) == len(set(mwids))

        for step in data_transfer_plan:
            if step.root not in mwids:
                handler = self.mwid2msids[step.root][0]
                payloads[handler] = request_reply_stream.Payload(
                    handler=handler,
                    handle_name="empty",
                    pre_hooks=["data_transfer"],
                    pre_hook_data=[dt_data],
                )
                mwids.append(step.root)
            if step.comm_type == "gather":
                for src in step.srcs:
                    if src not in mwids:
                        handler = self.mwid2msids[src][0]
                        payloads[handler] = request_reply_stream.Payload(
                            handler=handler,
                            handle_name="empty",
                            pre_hooks=["data_transfer"],
                            pre_hook_data=[dt_data],
                        )
                        mwids.append(src)

        payloads, mwids = self.attach_payloads_with_hooks(
            payloads,
            mwids,
            main_handlers=handlers,
            hook_type="pre",
        )
        payloads, mwids = self.attach_payloads_with_hooks(
            payloads,
            mwids,
            main_handlers=handlers,
            hook_type="post",
        )
        main_payloads = [p for h, p in payloads.items() if h in handlers]
        other_payloads = [p for h, p in payloads.items() if h not in handlers]
        all_req_ids = self.stream.request(
            payloads=main_payloads + other_payloads,
        )
        return all_req_ids[: len(main_payloads)], all_req_ids[len(main_payloads) :]

    def data_parallel_dispatch(
        self, buf_indices: List[int], sample: data_api.SequenceSample
    ) -> Tuple[List[int], data_api.SequenceSample, List[Tuple[int, int]]]:
        # Dispatch data to different data parallel ranks.
        if self.rpc.is_generate():
            # The workload of generation is decided by batch size, instead of the generated length.
            lens = [1 for _ in range(sample.bs)]
            samples, forward_indices, _ = sample.split_with_lengths(
                mb_spec=data_api.MicroBatchSpec(n_mbs=self.dp_size),
                lens=lens,
            )
        else:
            samples, forward_indices, _ = sample.split(
                data_api.MicroBatchSpec(n_mbs=self.dp_size)
            )
        blogger.debug(
            f"DP split (DP size {self.dp_size}) for RPC {self.rpc.name}: "
            f"#seqs: {[s.bs for s in samples]}, "
            f"#tokens: {[sum([sum(lens) for lens in s.seqlens[s._get_split_key()]]) for s in samples]}"
        )
        sample = data_api.SequenceSample.gather(samples)
        buf_indices = [buf_indices[i] for i in forward_indices]

        partitions = data_api.SequenceSplitSpec(
            sizes=[s.bs for s in samples]
        ).partitions
        return buf_indices, sample, partitions

    async def run_step(self, buf_indices, sample, buffer_id: int):
        rpc = self.rpc
        topo = self.model_topos[rpc.model_name]
        ctrl = self.rpc_ctrl

        handlers = [
            config_pkg.ModelShardID.from_parallelism_rank(rpc.model_name, topo, j)
            for j in range(topo.world_size())
        ]

        dp_head_indices = [
            topo.get_rank(data=i, pipe=topo.get_dim("pipe") - 1, tensor=0)
            for i in range(self.dp_size)
        ]

        async with ctrl.lock:
            ctrl.flops_counter.add_rpc(rpc, sample, self.model_configs[rpc.model_name])

        # logger.info(f"Model rpc {rpc.name} requesting.")

        # Sample may be reordered here.
        buf_indices, sample, partitions = self.data_parallel_dispatch(
            buf_indices, sample
        )

        # Build data destinations: GPU id -> List[data ids]
        partitioned_ids = []
        dests = {}
        for dp_rank, (st, ed) in enumerate(partitions):
            ranks = topo.filter_match(data=dp_rank)
            for rank in ranks:
                h = config_pkg.ModelShardID.from_parallelism_rank(
                    model_name=rpc.model_name, topo=topo, parallelism_rank=rank
                )
                gpu_id = self.msid2mwid[h]
                assert gpu_id not in dests
                dests[gpu_id] = sample.ids[st:ed]
            partitioned_ids.append(sample.ids[st:ed])
        for i in range(self.n_model_workers):
            if i not in dests:
                dests[i] = []

        pattern = "gather-scatter"
        data_transfer_plan = self.redistrib_planner.derive_plan(
            dests,
            keys=rpc.input_keys,
            pattern=pattern,
        )
        blogger.debug(f"Data tranfer plan for `{rpc.name}`: {data_transfer_plan}.")

        # Update storage tracker for transferred data.
        if pattern == "bcast":
            # NOTE: since the data we loaded may be unevenly distributed across DP ranks,
            # we should change the owner of the data to the src RPC.
            for i in range(topo.world_size()):
                h = ModelShardID.from_parallelism_rank(
                    model_name=rpc.model_name, topo=topo, parallelism_rank=i
                )
                is_dp_head = h.tp_rank == 0 and h.pp_rank == topo.get_dim("pipe") - 1
                gpu_id = self.msid2mwid[h]
                for key in rpc.input_keys:
                    await self.redistrib_planner.storage_tracker.add_data(
                        gpu_id, partitioned_ids[h.dp_rank], key=key, is_owner=is_dp_head
                    )
        else:
            for step in data_transfer_plan:
                if step.comm_type == "scatter":
                    for gpu_id, ids in zip(step.dsts, step.ids):
                        for key in step.keys:
                            await self.redistrib_planner.storage_tracker.add_data(
                                gpu_id, ids, key=key, is_owner=False
                            )
                elif step.comm_type == "gather":
                    for key in step.keys:
                        await self.redistrib_planner.storage_tracker.add_data(
                            step.root,
                            list(itertools.chain.from_iterable(step.ids)),
                            key=key,
                            is_owner=False,
                        )

        await asyncio.sleep(0)
        # send partitioned data to model workers
        req_ids, other_req_ids = self.request(
            data_transfer_plan=data_transfer_plan,
            partitioned_ids=partitioned_ids,
            meta_sample=sample,
            handlers=handlers,
        )
        tik = time.perf_counter()

        await ctrl.topo_level_count.put(1)
        logger.info(f"Model rpc {rpc.name} requested.")

        # Then, wait for all main requests to finish.
        responses = await self.stream.gather_async(request_ids=req_ids)
        # logger.info(f"rpc {rpc.name} received responses {req_ids}")

        # Filter out responses other than DP heads.
        # Other repsonses are duplicated or None.
        responses, time_records = list(zip(*[responses[i] for i in dp_head_indices]))

        # If the returned data is a SequenceSample, it is the data returned by
        # model function calls. The data should be amended into buffer.
        # Otherwise, it's the train statistics and should be reduced and logged.
        if isinstance(responses[-1], data_api.SequenceSample):
            # Update storage tracker for generated data.
            for dp_rank, x in enumerate(responses):
                pp_size = topo.get_dim("pipe")
                ranks = topo.filter_match(data=dp_rank, pipe=pp_size - 1, tensor=0)
                for rank in ranks:
                    h = config_pkg.ModelShardID.from_parallelism_rank(
                        model_name=rpc.model_name, topo=topo, parallelism_rank=rank
                    )
                    gpu_id = self.msid2mwid[h]
                    for k in rpc.output_keys:
                        await self.redistrib_planner.storage_tracker.add_data(
                            gpu_id,
                            x.ids,
                            key=k,
                            is_owner=True,
                        )
            res = data_api.SequenceSample.gather(responses)
        elif isinstance(responses[0], dict):
            res = data_api.gather_stat(responses)
        else:
            assert isinstance(responses[0], list)
            res = [
                data_api.gather_stat([r[i] for r in responses])
                for i in range(len(responses[0]))
            ]

        if rpc.log_return_value:
            if isinstance(res, dict):
                logger.info(
                    f"RPC name {rpc.name} returns\n{data_api.tabulate_stats(res)}"
                )
                logging.log_swanlab_wandb_tensorboard(
                    res,
                    step=ctrl.step_info.global_step,
                    summary_writer=self.summary_writer,
                )
            elif isinstance(res, list):
                for j, r in enumerate(res):
                    logger.info(
                        f"RPC name {rpc.name} returns ({j + 1}/{len(res)})\n{data_api.tabulate_stats(r)}"
                    )
                    offset = len(res) * ctrl.step_info.global_step
                    logging.log_swanlab_wandb_tensorboard(
                        r,
                        step=offset + j,
                        summary_writer=self.summary_writer,
                    )
            else:
                logger.info(f"RPC name {rpc.name} returns\n{res}")

        # Log rpc execution time.
        for time_record in time_records:
            stats_tracker.scalar(**time_record)
        time_stats = stats_tracker.export()
        logging.log_swanlab_wandb_tensorboard(
            time_stats,
            summary_writer=self.summary_writer,
        )
        logger.info(
            f"Model rpc {rpc.name} finished. "
            f"Request-reply time {time.perf_counter() - tik:.4f}s. "
            f"Detailed time stats:\n{data_api.tabulate_stats(time_stats, floatfmt='.2f')}."
        )

        # If this RPC is the final node in the dataflow graph,
        # update the train counter.
        # Otherwise, amend data in the buffer.
        if rpc.is_dst:
            async with ctrl.lock:
                ctrl.ids_to_clear = ctrl.ids_to_clear.union(sample.ids)
            await ctrl.train_count.put(1)
        else:
            logger.info(f"Amending RPC {rpc.name} output keys: {res.keys}")
            await self.buffers[buffer_id].amend_batch(buf_indices, res.unpack())

        # Wait for all side-effect requests to finish.
        # Side-effect or empty requests are required for data transfer
        # and parameter synchronization.
        # Wait them after the main request to log the oorrect MFC time.
        await self.stream.gather_async(other_req_ids)

    async def run(self, buffer_id: int):
        rpc = self.rpc
        topo = self.model_topos[rpc.model_name]

        logger.info(
            f"Running Model RPC, interface_type=#{rpc.interface_type}# "
            f"(dp,tp,pp) = *({topo.get_dim('data')},{topo.get_dim('tensor')},{topo.get_dim('pipe')})*"
        )

        consumed = 0
        while True:
            buf_indices, sample = await self.buffers[buffer_id].get_batch_for_rpc(rpc)

            await self.run_step(buf_indices, sample, buffer_id)
            consumed += sample.bs

            # Ensure that parent RPCs will not be over-consumed.
            if all(consumed >= c.n_seqs for c in rpc.all_successors()):
                break
