# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import asyncio
import copy
import gc
import os
import time
from typing import Dict

import colorama
import networkx as nx
import numpy as np
import wandb
from tensorboardX import SummaryWriter

import realhf.api.core.dfg as dfg
import realhf.api.core.model_api as model_api
import realhf.api.core.system_api as config_pkg
import realhf.base.recover as recover
import realhf.system.request_reply_stream as request_reply_stream
import realhf.system.worker_base as worker_base
from realhf.api.core.config import ModelName
from realhf.api.core.model_api import ReaLModelConfig
from realhf.api.core.system_api import ExpStatus
from realhf.base import (
    constants,
    logging,
    name_resolve,
    names,
    seeding,
    timeutil,
    topology,
)
from realhf.system.buffer import AsyncIOSequenceBuffer
from realhf.system.function_executor import FunctionExecutor
from realhf.system.model_function_call import RPCCorountineControl

try:
    import swanlab
except (ModuleNotFoundError, ImportError):
    swanlab = None

logger = logging.getLogger("master worker", "system")
blogger = logging.getLogger("benchmark")


class MasterWorker(worker_base.AsyncWorker):
    global_exp_tik = time.perf_counter()

    def _configure(self, config: config_pkg.MasterWorker):
        self.config = config

        seeding.set_random_seed(self.config.base_seed, "master_worker")

        self.__model_topos: Dict[ModelName, topology.ProcessTopology] = (
            config.model_topos
        )

        # Build execution graph and initialize concurrency utilities.
        self.__model_rpcs = config.model_rpcs

        # Sort all MFCs in the topological order and
        # calculate the width of each level.
        # These numbers will determine when to flush MFC requests.
        self.__topo_widths = []
        for generation in nx.topological_generations(self.__model_rpcs[0]._G):
            self.__topo_widths.append(len(generation))
        logger.debug("Topological widths: " + str(self.__topo_widths))

        self.__rpc_srcs = list(filter(lambda rpc: rpc.is_src, self.__model_rpcs))
        self.__rpc_dsts = list(filter(lambda rpc: rpc.is_dst, self.__model_rpcs))

        # Save and eval control.
        self.__total_train_epochs = config.exp_ctrl.total_train_epochs
        self.__save_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.exp_ctrl.save_freq_epochs,
            freq_step=config.exp_ctrl.save_freq_steps,
            freq_sec=config.exp_ctrl.save_freq_secs,
        )
        if (
            config.exp_ctrl.ckpt_freq_epochs is None
            and config.exp_ctrl.ckpt_freq_steps is None
            and config.exp_ctrl.ckpt_freq_secs is None
        ):
            self.__ckpt_ctl = timeutil.EpochStepTimeFreqCtl(
                freq_epoch=config.exp_ctrl.save_freq_epochs,
                freq_step=config.exp_ctrl.save_freq_steps,
                freq_sec=config.exp_ctrl.save_freq_secs,
            )
        else:
            self.__ckpt_ctl = timeutil.EpochStepTimeFreqCtl(
                freq_epoch=config.exp_ctrl.ckpt_freq_epochs,
                freq_step=config.exp_ctrl.ckpt_freq_steps,
                freq_sec=config.exp_ctrl.ckpt_freq_secs,
            )
        self.__eval_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.exp_ctrl.eval_freq_epochs,
            freq_step=config.exp_ctrl.eval_freq_steps,
            freq_sec=config.exp_ctrl.eval_freq_secs,
        )

        self.__initialized = False
        self.__recover_run, self.__recover_info = recover.load_recover_info(self.args)
        if self.__recover_info is not None:
            logger.info(
                f"Loaded recover info: recover_start={self.__recover_info.recover_start}, "
                f"last_step_info={self.__recover_info.last_step_info}."
            )
            logger.info(
                f"Number of used data in recover info: {len(self.__recover_info.hash_vals_to_ignore)}. "
                f"The previous experiment probably ran for {len(self.__recover_info.hash_vals_to_ignore) // self.__rpc_srcs[0].n_seqs} steps in the epoch."
            )

        # Create corountine control objects for running the dataflow graph.
        self.__rpc_ctrl = RPCCorountineControl(
            train_count=asyncio.Queue(maxsize=len(self.__rpc_dsts)),
            topo_level_count=asyncio.Queue(maxsize=sum(self.__topo_widths)),
            lock=asyncio.Lock(),
            # NOTE: We should accumulate the used data hashes in the same epoch
            # to prevent loading data used before.
            used_hash_vals_this_epoch=(
                copy.deepcopy(self.__recover_info.hash_vals_to_ignore)
                if self.__recover_run
                else list()
            ),
        )

        if self.__recover_run:
            self.__rpc_ctrl.step_info = copy.deepcopy(self.__recover_info.recover_start)

            self.__eval_ctl.load_state_dict(self.__recover_info.eval_ctl_info)
            self.__save_ctl.load_state_dict(self.__recover_info.save_ctl_info)
            self.__ckpt_ctl.load_state_dict(self.__recover_info.ckpt_ctl_info)

            logger.info(
                f"Recovering from previous run. "
                f"Epoch: {self.__rpc_ctrl.step_info.epoch + 1}, "
                f"Epoch Step: {self.__rpc_ctrl.step_info.epoch_step + 1} "
                f"Global Step: {self.__rpc_ctrl.step_info.global_step + 1}."
            )

            # Recover the previous number of training samples
            train_rpcs = list(
                filter(
                    lambda rpc: rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP,
                    self.__model_rpcs,
                )
            )
            train_batch_size = train_rpcs[0].n_seqs
            hist_samples = (
                train_batch_size * self.__recover_info.last_step_info.global_step
            )
            training_sample_name = names.training_samples(
                constants.experiment_name(), constants.trial_name()
            )
            name_resolve.add(training_sample_name, str(hist_samples), replace=True)

        # for benchmark
        self.e2e_time_history = []
        self.__benchmark_steps = config.exp_ctrl.benchmark_steps
        self.__benchmark_n_seqs = config.exp_ctrl.benchmark_n_seqs

        return config.worker_info

    def initialize_models(self):
        # Initialize model backends.
        model_names = list(self.__model_topos.keys())
        self.logger.debug(f"Initialize model backends with order: {model_names}.")
        train_rpcs = list(
            filter(
                lambda rpc: rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP,
                self.__model_rpcs,
            )
        )
        assert all(rpc.n_seqs == train_rpcs[0].n_seqs for rpc in train_rpcs)
        if len(train_rpcs) > 0:
            ft_spec = model_api.FinetuneSpec(
                total_train_epochs=self.config.exp_ctrl.total_train_epochs,
                dataset_size=self._dataset_size,
                train_batch_size=train_rpcs[0].n_seqs,
            )
        else:
            ft_spec = model_api.FinetuneSpec(
                total_train_epochs=self.config.exp_ctrl.total_train_epochs,
                dataset_size=self._dataset_size,
                train_batch_size=self.__src_rpc.n_seqs,
            )
        _initialized_roles = []
        for model_name in model_names:
            topo = self.config.model_topos[model_name]
            # Build FinetuneSpec, which is required to initialize backends.
            _handlers = [
                config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, j)
                for j in range(topo.world_size())
            ]

            init_payloads = [
                request_reply_stream.Payload(
                    handler=_h,
                    handle_name="initialize",
                    data=ft_spec,
                )
                for _h in _handlers
            ]

            # Send initialization requests then immediately flush them.
            self.__stream.request(
                payloads=init_payloads,
            )
            self.__stream.request(
                handlers=_handlers,
                handle_type="flush",
                no_syn=True,
            )

            _initialized_roles.append(model_name.role)

        self._ft_spec = ft_spec
        logger.info("Initializations of models and backends complete.")

    def get_dataset_model_info(self):
        src_rpc = self.__rpc_srcs[0]
        src_rpc_topo = self.config.model_topos[src_rpc.model_name]
        src_rpc_dp_size = src_rpc_topo.get_dim("data")

        # Request training specification from data workers.
        specs = self.__stream.call(
            handlers=[f"__data{i}__" for i in range(src_rpc_dp_size)],
            datas=[None for i in range(src_rpc_dp_size)],
            handle_type="spec",
        )
        assert all(x["n_datasets"] == specs[0]["n_datasets"] for x in specs), specs
        self._dataset_size = sum(x["dataset_size"] for x in specs)
        self._n_datasets = specs[0]["n_datasets"]

        self._steps_per_epoch = self._dataset_size // src_rpc.n_seqs

        # Request model configs from model workers.
        # Return None if the model is not a ReaLModel.
        self.__model_configs: Dict[ModelName, None | ReaLModelConfig] = {}
        for model_name, topo in self.config.model_topos.items():
            h = config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, 0)
            self.__model_configs[model_name] = self.__stream.call(
                handlers=[h],
                datas=[None],
                handle_type="model_config",
            )[0]

    def __lazy_init(self):
        # Set up streams.
        handler_routing = copy.deepcopy(self.config.msid2mwid)
        src_rpc = self.__rpc_srcs[0]
        src_rpc_topo = self.config.model_topos[src_rpc.model_name]
        src_rpc_dp_size = src_rpc_topo.get_dim("data")
        src_rpc_pp_size = src_rpc_topo.get_dim("pipe")
        for i in range(src_rpc_dp_size):
            rank = src_rpc_topo.get_rank(data=i, pipe=src_rpc_pp_size - 1, tensor=0)
            handler_routing[f"__data{i}__"] = self.config.msid2mwid[
                config_pkg.ModelShardID.from_parallelism_rank(
                    model_name=src_rpc.model_name,
                    topo=src_rpc_topo,
                    parallelism_rank=rank,
                )
            ]
        handler_routing.update({i: i for i in range(self.config.n_model_workers)})
        self.__stream = request_reply_stream.make_master_stream(
            self.config.worker_info,
            n_subscribers=self.config.n_model_workers,
            handler_routing=handler_routing,
        )
        self.__stream: request_reply_stream.NameResolvingRequestClient

        self.__src_rpc = src_rpc = [
            rpc for rpc in self.config.model_rpcs if rpc.is_src
        ][0]

        self.get_dataset_model_info()

        self.initialize_models()

        self.__seqbuffers = [
            AsyncIOSequenceBuffer(
                self.__model_rpcs,
                max_size=int(os.getenv("REAL_MASTER_BUFFER_SIZE", str(int(1e7)))),
            )
            for _ in range(self._n_datasets)
        ]

        # wandb init, connect to remote wandb host
        if self.wandb_config.mode != "disabled":
            wandb.login()
        wandb.init(
            mode=self.wandb_config.mode,
            entity=self.wandb_config.entity,
            project=self.wandb_config.project or constants.experiment_name(),
            name=self.wandb_config.name or f"{constants.trial_name()}_train",
            job_type=self.wandb_config.job_type,
            group=self.wandb_config.group
            or f"{constants.experiment_name()}_{constants.trial_name()}",
            notes=self.wandb_config.notes,
            tags=self.wandb_config.tags,
            config=self.wandb_config.config,
            dir=constants.get_log_path(self.args),
            force=True,
            id=f"{constants.experiment_name()}_{constants.trial_name()}_train",
            resume="allow",
            settings=wandb.Settings(start_method="fork"),
        )

        # swanlab init, connect to remote or local swanlab host
        if self.swanlab_config.mode != "disabled" and self.swanlab_config.api_key:
            if swanlab is not None:
                swanlab.login(self.swanlab_config.api_key)
            else:
                logger.warning(
                    "swanlab not installed but enabled. Ignore swanlab logging."
                )
        if self.swanlab_config.config is None:
            import yaml

            with open(
                os.path.join(
                    constants.get_log_path(self.args),
                    "config.yaml",
                ),
                "r",
            ) as f:
                __config = yaml.safe_load(f)
        else:
            __config = self.swanlab_config.config
        __config["FRAMEWORK"] = "AReaL"
        if swanlab is not None:
            swanlab.init(
                project=self.swanlab_config.project or constants.experiment_name(),
                experiment_name=self.swanlab_config.name
                or f"{constants.trial_name()}_train",
                config=__config,
                logdir=self.swanlab_config.logdir
                or os.path.join(
                    constants.get_log_path(self.args),
                    "swanlab",
                ),
                mode=self.swanlab_config.mode,
            )
        # tensorboard logging
        self.__summary_writer = None
        if self.tensorboard_config.path is not None:
            self.__summary_writer = SummaryWriter(log_dir=self.tensorboard_config.path)

        # Create coroutines for model RPCs.
        logger.debug(f"Creating asyncio coroutines...")
        self.func_executor = FunctionExecutor(
            args=self.args,
            rpcs=self.__model_rpcs,
            msid2mwid=self.config.msid2mwid,
            stream=self.__stream,
            buffers=self.__seqbuffers,
            model_topos=self.__model_topos,
            model_configs=self.__model_configs,
            ctrl=self.__rpc_ctrl,
            summary_writer=self.__summary_writer,
            shuffle_dataset=self.config.shuffle_dataset,
        )
        if self.__recover_run:
            self.func_executor.data_loading_dp_idx = (
                self.__recover_info.data_loading_dp_idx
            )
        logger.debug(f"Coroutines created. The master worker is ready to run.")

        self.__initialized = True
        self._train_start_time = time.perf_counter()

        self.__last_step_info = recover.StepInfo(
            epoch=-1,
            epoch_step=-1,
            global_step=-1,
        )

    async def __poll_async(self):
        is_new_epoch = False

        if not self.__initialized:
            self.__lazy_init()

        # Main execution steps. The graph runs under-the-hood in RPC & stream threads.
        # Wait for the finish of the traversal of the execution graph.
        execution_start = time.perf_counter()

        is_new_epoch = self._ft_spec.is_new_epoch(self.__rpc_ctrl.step_info)
        is_epoch_last_step = self._ft_spec.is_epoch_last_step(self.__rpc_ctrl.step_info)

        # Check whether we should evaluate or save models.
        self.__rpc_ctrl.should_eval = self.__eval_ctl.check(
            epochs=int(is_epoch_last_step), steps=1
        )
        self.__rpc_ctrl.should_save = self.__save_ctl.check(
            epochs=int(is_epoch_last_step), steps=1
        )
        self.__rpc_ctrl.should_ckpt = self.__ckpt_ctl.check(
            epochs=int(is_epoch_last_step), steps=1
        )

        # Log eval/save info.
        epoch = self.__rpc_ctrl.step_info.epoch + 1
        epoch_step = self.__rpc_ctrl.step_info.epoch_step + 1
        global_step = self.__rpc_ctrl.step_info.global_step + 1
        if is_new_epoch:
            epoch += 1
            epoch_step = 1
        s = f"The next step is epoch {epoch}/{self.config.exp_ctrl.total_train_epochs} "
        s += f"step {epoch_step}/{self._steps_per_epoch} "
        s += f"(global step {global_step}). "
        s += f"Should checkpoint? {self.__rpc_ctrl.should_ckpt}. "
        s += f"Should save? {self.__rpc_ctrl.should_save}. "
        s += f"Should run evaluation? {self.__rpc_ctrl.should_eval}. "
        s += f"Is the first step in epoch? {is_new_epoch}. "
        s += f"Is the last step in epoch? {is_epoch_last_step}. "
        self.logger.info(s)

        # Traverse over the dataflow graph for once.
        await self.func_executor.execute_step()

        # Post-process.
        if self.__rpc_ctrl.should_save or self.__rpc_ctrl.should_ckpt:
            self.__last_step_info = copy.deepcopy(self.__rpc_ctrl.step_info)

        if is_epoch_last_step:
            self.__rpc_ctrl.used_hash_vals_this_epoch = (
                self.__rpc_ctrl.used_hash_vals_this_epoch[self._dataset_size :]
            )

        if is_new_epoch:
            self.__rpc_ctrl.step_info.epoch += 1
            self.__rpc_ctrl.step_info.epoch_step = 0

        # Logging.
        time_since_configure = time.perf_counter() - self._train_start_time
        e2e_time = time.perf_counter() - execution_start
        self.e2e_time_history.append(e2e_time)

        self._log_training_stats(e2e_time, time_since_configure)

        # Updata counters.
        self.__rpc_ctrl.step_info.epoch_step += 1
        self.__rpc_ctrl.step_info.global_step += 1

        if self.__rpc_ctrl.should_save or self.__rpc_ctrl.should_ckpt:
            self.__recover_save()

        # Pause the worker if experiment or system-wise benchmark completes.
        if (
            (
                self.__benchmark_steps is not None
                and self.__rpc_ctrl.step_info.global_step >= self.__benchmark_steps
            )
            or (
                self.__rpc_ctrl.step_info.global_step * self.__src_rpc.n_seqs
                >= self.__total_train_epochs * self._dataset_size
            )
            or (
                self.__benchmark_n_seqs is not None
                and self.__rpc_ctrl.step_info.global_step
                * self._ft_spec.train_batch_size
                >= self.__benchmark_n_seqs
            )
        ):
            # We don't know whether it is the last step of the current epoch,
            # so we exit at the first step of the next epoch.
            if (
                self.__benchmark_steps is not None
                or self.__benchmark_n_seqs is not None
            ):
                logger.info(
                    f"Finished benchmark {self.__benchmark_steps}. "
                    f"Time consumption of this setup: {time_since_configure:.3f}"
                )
                logger.info(f"avg #e2e# time *{np.mean(self.e2e_time_history):.3f}*")
            # TODO: inform generation workers to exit
            return self.experiment_complete_exit()

        return worker_base.PollResult(sample_count=1, batch_count=1)

    async def _poll_async(self):
        name = names.experiment_status(
            constants.experiment_name(), constants.trial_name()
        )
        name_resolve.add(name, ExpStatus.RUNNING, replace=True)
        try:
            r = await self.__poll_async()
        except Exception as e:
            name_resolve.add(name, ExpStatus.ABORTED, replace=True)
            raise e
        return r

    def _log_training_stats(self, e2e_time: float, time_since_configure: float):
        # calculate flops
        #########################################
        if not all(
            isinstance(v, ReaLModelConfig) for v in self.__model_configs.values()
        ):
            logger.warning(
                f"Not all models are ReaLModels. Unable to calculate FLOP/s."
            )
            flops = None
            tflops_per_gpu = float("inf")
        else:
            flops = self.__rpc_ctrl.flops_counter.get_flops()
            tflops = flops / (e2e_time * (10**12))
            tflops_per_gpu = flops / (e2e_time * self.config.n_model_workers * (10**12))
        self.__rpc_ctrl.flops_counter.clear()
        #########################################

        epoch = self.__rpc_ctrl.step_info.epoch + 1
        epoch_step = self.__rpc_ctrl.step_info.epoch_step + 1
        global_step = self.__rpc_ctrl.step_info.global_step + 1
        s = f"Epoch {epoch}/{self.config.exp_ctrl.total_train_epochs} "
        s += f"step {epoch_step}/{self._steps_per_epoch} "
        s += f"(global step {global_step}) finishes. "
        s += f"#End to end# execution time: *{e2e_time:.3f}*s. "
        s += f"Total time consumption: {time_since_configure:.3f}s. "
        logging.log_swanlab_wandb_tensorboard({"timeperf/e2e": e2e_time})
        if len(self.e2e_time_history) > 2:
            remaining_steps = self._steps_per_epoch - epoch_step
            remaining_epochs = self.__total_train_epochs - epoch
            avg_t = np.mean(self.e2e_time_history[2:])
            remain_t = avg_t * remaining_steps
            remain_t += avg_t * self._steps_per_epoch * remaining_epochs
            s += f"Estimated remaining time: {remain_t:.3f}s. "
        if flops is not None:
            s += f"TFLOP/s per GPU: {tflops_per_gpu:.2f}, total TFLOP/s: {tflops:.2f}."
        self.logger.info(s)
        self.logger.info(
            f"Time taken so far across all configurations: {time.perf_counter() - self.global_exp_tik:.2f}s"
        )

    def experiment_complete_exit(self):
        logger.info(
            colorama.Style.RESET_ALL
            + colorama.Fore.YELLOW
            + colorama.Style.BRIGHT
            + "\033[1m"
            + "Experiment Completes! Yeah!!!!!!!!"
            + colorama.Style.RESET_ALL
        )

        # Update experiment status to inform other workers
        name = names.experiment_status(
            constants.experiment_name(), constants.trial_name()
        )
        name_resolve.add(name, ExpStatus.COMPLETE, replace=True)

        # Send requests to pause model workers.
        # Model workers will not respond to this message.
        # FIXME: request to model workers is unnecessary
        self.__stream.request(
            handlers=list(range(self.config.n_model_workers)),
            handle_type="reset",
            datas=[None for _ in list(range(self.config.n_model_workers))],
        )
        self.__stream.close()
        constants.reset_run()
        # Reset names used for distributed training.
        # The next round of training will set up a new distributed environment.
        name_resolve.clear_subtree(
            names.distributed_root(constants.experiment_name(), constants.trial_name())
        )
        name_resolve.clear_subtree(
            names.request_reply_stream_root(
                constants.experiment_name(), constants.trial_name()
            )
        )

        wandb.finish()
        if swanlab is not None:
            swanlab.finish()
        if self.__summary_writer is not None:
            self.__summary_writer.close()
        gc.collect()
        self.__initialized = False
        self.pause()
        return worker_base.PollResult(0, 0)

    def __recover_save(self):
        # save step info for recover
        if os.getenv("REAL_SAVE_RECOVER_STATES", "0") != "1":
            return
        # save step info for recover
        this_step_info = copy.deepcopy(self.__rpc_ctrl.step_info)
        recover_info = recover.RecoverInfo(
            recover_start=this_step_info,
            last_step_info=self.__last_step_info,
            save_ctl_info=self.__save_ctl.state_dict(),
            ckpt_ctl_info=self.__ckpt_ctl.state_dict(),
            eval_ctl_info=self.__eval_ctl.state_dict(),
            data_loading_dp_idx=self.func_executor.data_loading_dp_idx,
            hash_vals_to_ignore=self.__rpc_ctrl.used_hash_vals_this_epoch,
        )

        recover.dump_recover_info(self.args, recover_info)
        logger.info("Dumped recover info to file.")
        logger.info(f"Will recover from: {recover_info.recover_start}")
        logger.info(
            f"Number of data used in this epoch: {len(recover_info.hash_vals_to_ignore)}"
        )
