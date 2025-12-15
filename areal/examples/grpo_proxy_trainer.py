import concurrent.futures
import json
import os
import pprint
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from realhf.api.core.data_api import load_hf_tokenizer

from areal.api.cli_args import (
    SchedulingStrategy,
    load_expr_config,
)
from areal.api.engine_api import WeightUpdateMeta
from areal.api.io_struct import AllocationMode, FinetuneSpec, SaveLoadMeta
from areal.extension.asystem.api.cli_args import GRPOConfig
from areal.extension.asystem.ascheduler import AsystemScheduler
from areal.extension.asystem.controller import RolloutController, TrainController
from areal.extension.asystem.recover import latest_checkpoint, periodic_checkpoint
from areal.extension.asystem.remote_hybrid_inference_worker import (
    RemoteHybridInferenceWorker,
)
from areal.extension.asystem.remote_hybrid_train_worker import RemoteHybridTrainWorker
from areal.extension.asystem.utils.align_tools import summarize_rewards
from areal.extension.asystem.utils.util import ShuffleSampler, wait_future_ordered
from areal.utils import logging, stats_tracker
from areal.utils.data import cycle_dataloader
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("Trainer")


def clear_ref_resource(last_ref_model_path: str):
    if last_ref_model_path is not None and os.path.exists(last_ref_model_path):
        def cleanup_ref_model_path(path):
            try:
                if os.path.exists(path):
                    shutil.rmtree(path)
                    logger.info(
                        f"Async cleanup completed for ref model path: {path}"
                    )
            except Exception as e:
                logger.error(
                    f"Async cleanup failed for old ref model path {path}: {str(e)}"
                )

        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(
            cleanup_ref_model_path, last_ref_model_path
        )
        executor.shutdown(wait=False)
        logger.info(
            f"Started async cleanup for old ref model path: {last_ref_model_path}"
        )


def clear_dir(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


def load_dataset(path: str) -> Dataset:
    try:
        train_dataset = Dataset.load_from_disk(path)
        logger.info(f"Loaded dataset from disk: {path}")
        return train_dataset
    except Exception:
        logger.warning(f"Failed to load dataset from disk: {path}")
        from datasets import load_dataset

        if path.endswith(".json") or path.endswith(".jsonl"):
            logger.info(f"Loading dataset from local file: {path}")
            ## load local json/jsonl file
            return load_dataset("json", data_files=path, split="train")

        logger.info(f"Loading dataset from Hugging Face: {path}")
        ## load gaia2 dataset
        return load_dataset(path, "search", split="validation")


@dataclass
class ProxyAgentConfig(GRPOConfig):
    tool_call_parser: str = field(
        default="qwen25",
    )

    reasoning_parser: str = field(
        default="qwen3",
    )

    agent_process_pool_size: int = field(
        default=256,
        metadata={"help": "Number of parallel processes for running agents."},
    )

    agent_module_path: Any = field(
        default="examples.any_agents.agent.math.math_agent",
        metadata={"help": "Module path for the agent definition."},
    )

    export_style: str = field(
        default="concat",
        metadata={"help": "Export style for the proxy server."},
    )


def main(args):
    try:
        from loguru import logger as tau2_logger

        tau2_logger.remove()
    except ImportError:
        pass

    config, _ = load_expr_config(args, ProxyAgentConfig)
    config: ProxyAgentConfig

    if config.enable_colocate_mode:
        config.rollout.engine_config["enable_memory_saver"] = True

    logger.info(
        "Loaded config:\n" + pprint.pformat(config, indent=2, width=120, depth=6)
    )

    # init scheduler
    scheduler = AsystemScheduler(
        {
            "endpoint": config.scheduler.endpoint,
            "expr_name": config.experiment_name,
            "trial_name": config.trial_name,
            "extra_envs": {
                "REWARD_MODEL_PATH": config.scheduler.reward_model_path,
                "REWARD_MODEL_SERVICE_URL": config.scheduler.reward_model_service_url,
                "FUNCTIONCALL_SERVICE_DOMAIN": config.scheduler.functioncall_service_domain,
                "REWARD_FUNCTIONCALL_CONFIG": json.dumps(
                    config.scheduler.reward_functioncall_config
                ),
            },
            "storage_prefix": config.storage_prefix,
        }
    )

    try:
        if config.weight_update_type != "disk":
            from areal.extension.asystem.meta_server import start_meta_server

            host, port = start_meta_server()
            meta_server_addr = f"{host}:{port}"

            asystem_hybrid_config = {
                "meta_server_addr": meta_server_addr,
                "weights_exchange_comm_backend": config.weight_update_type,
                "weights_validation_steps": 0,
                "enable_debug_mode": True,
            }
            config.rollout.engine_config["asystem_hybrid_config"] = (
                asystem_hybrid_config
            )
            config.actor.hybrid_engine.remote_megatron_config[
                "asystem_train_config"
            ] = asystem_hybrid_config

        global_step = 0
        os.environ["WANDB_API_KEY"] = config.stats_logger.wandb.wandb_api_key
        os.environ["WANDB_BASE_URL"] = config.stats_logger.wandb.wandb_base_url

        allocation_mode = config.allocation_mode
        allocate_mode = AllocationMode.from_str(allocation_mode)

        tokenizer = load_hf_tokenizer(config.tokenizer_path)
        train_dataset = load_dataset(path=config.train_dataset.path)

        dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=config.train_dataset.batch_size,
            sampler=ShuffleSampler(train_dataset),
            collate_fn=lambda x: x,
            drop_last=config.train_dataset.drop_last,
        )

        epoch_num = config.total_train_epochs
        steps_per_epoch = len(dataloader)
        ref_model_path = None
        last_ref_model_path = None
        ############################## recover #########################################
        recover_meta_info_path = config.recover.recover_meta_info_path
        enable_recover = True
        can_recover = False
        recover_epoch = 0
        recover_step = 0
        if recover_meta_info_path != "" and enable_recover:
            can_recover, recover_meta_info = latest_checkpoint.Recover.load(
                recover_meta_info_path
            )
            if not can_recover:
                logger.warning(
                    f"recover file: {recover_meta_info_path} not exists, skip recover."
                )
            if can_recover and recover_meta_info.global_step == 0:
                logger.warning("global_step 0 does not require recovery, skip recover.")
                can_recover = False

        if can_recover:
            recover_epoch = recover_meta_info.epoch
            recover_global_step = recover_meta_info.global_step + 1
            recover_step = recover_meta_info.epoch_step + 1
            global_step = recover_meta_info.global_step + 1
            config.actor.hybrid_engine.global_step = global_step
            logger.info(
                f"🚀Recover success! global_step: {recover_meta_info.global_step}, epoch: {recover_meta_info.epoch}, step: {recover_meta_info.epoch_step}, "
                f"start new exp: global_step: {recover_global_step}, cur_step: {recover_step}"
            )
        latest_recover = latest_checkpoint.Recover(config.recover)
        if can_recover:
            dataloader.load_state_dict(recover_meta_info.dataloader_state)
            config.actor.hybrid_engine.recover_dir = recover_meta_info.checkpoint_path
            if (
                config.ref.enable_update_ref_model
                and hasattr(recover_meta_info, "ref_model_path")
                and recover_meta_info.ref_model_path
            ):
                ref_model_path = recover_meta_info.ref_model_path
                config.ref.hybrid_engine.recover_dir = ref_model_path
                logger.info(
                    f"Recover ref model from path: {ref_model_path}"
                )

        periodic_recover = periodic_checkpoint.Recover(config.recover)
        ##################################################################################

        ft_spec = FinetuneSpec(
            total_train_epochs=config.total_train_epochs,
            dataset_size=len(dataloader) * config.train_dataset.batch_size,
            train_batch_size=config.train_dataset.batch_size,
        )

        stats_logger = StatsLogger(
            config,
            ft_spec,
        )

        logger.info(f"total_epochs={epoch_num} step_per_epoch={steps_per_epoch}")

        inference_config = config.rollout
        inference_config.dp_size = allocate_mode.gen.dp_size  # TODO: use allocate_mode
        inference_config.tp_size = allocate_mode.gen.tp_size
        inference_config.pp_size = allocate_mode.gen.pp_size
        inference_config.storage_path = f"{config.rollout.storage_path}/{config.experiment_name}/{config.trial_name}"
        inference_config.seed = config.seed
        inference_config.scheduling_strategy = (
            SchedulingStrategy(type="colocation", target="actor")
            if config.enable_colocate_mode
            else inference_config.scheduling_strategy
        )

        rollout = RolloutController(
            RemoteHybridInferenceWorker,
            inference_config,
            scheduler,
        )

        actor = TrainController(
            RemoteHybridTrainWorker,
            config.actor,
            scheduler,
        )

        ref = None
        if config.actor.hybrid_engine.wrap_policy.kl_ctl > 0:
            ref = TrainController(
                RemoteHybridTrainWorker,
                config.ref,
                scheduler,
            )

        allocation_mode = AllocationMode.from_str(config.allocation_mode)

        def init_train_and_rollout_controller_helper(actor, rollout):
            logger.info("initializing trainer controller and rollout controller")
            actor.initialize(
                role="actor",
                parallel_strategy=allocation_mode["actor"].parallel,
                ft_spec=ft_spec,
                group_size=config.gconfig.n_samples,
                enable_colocate_mode=config.enable_colocate_mode,
            )
            rollout.initialize(
                role="rollout",
                alloc_mode=allocation_mode,
                enable_colocate_mode=config.enable_colocate_mode,
            )

        if config.enable_colocate_mode:
            logger.info(
                f"initializing all controllers in colocation mode {config.enable_colocate_mode}"
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        init_train_and_rollout_controller_helper, actor, rollout
                    ),
                ]
                if ref is not None:
                    futures.append(
                        executor.submit(
                            ref.initialize,
                            role="ref",
                            parallel_strategy=allocation_mode["ref"].parallel,
                            group_size=config.gconfig.n_samples,
                            ft_spec=ft_spec,
                            enable_colocate_mode=config.enable_colocate_mode,
                        )
                    )

                wait_future_ordered(futures)
            logger.info(
                f"initialized all controllers in colocation mode {config.enable_colocate_mode}"
            )
        else:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(
                        actor.initialize,
                        role="actor",
                        parallel_strategy=allocation_mode["actor"].parallel,
                        ft_spec=ft_spec,
                        group_size=config.gconfig.n_samples,
                        enable_colocate_mode=config.enable_colocate_mode,
                    ),
                    executor.submit(
                        rollout.initialize,
                        role="rollout",
                        alloc_mode=allocation_mode,
                        enable_colocate_mode=config.enable_colocate_mode,
                    ),
                ]
                if ref is not None:
                    futures.append(
                        executor.submit(
                            ref.initialize,
                            role="ref",
                            parallel_strategy=allocation_mode["ref"].parallel,
                            ft_spec=ft_spec,
                            group_size=config.gconfig.n_samples,
                            enable_colocate_mode=config.enable_colocate_mode,
                        )
                    )

                wait_future_ordered(futures)
            logger.info(
                f"initialized all controllers in colocation mode {config.enable_colocate_mode}"
            )

        if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
        if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

        weight_update_config = WeightUpdateMeta(
            type=config.weight_update_type,
            path=f"{config.storage_prefix}/checkpoints/{config.experiment_name}/{config.trial_name}",
            alloc_mode=None,
        )

        for epoch in range(recover_epoch, epoch_num):
            data_generator = cycle_dataloader(dataloader)
            start_step = recover_step if can_recover and epoch == recover_epoch else 0
            for step in range(start_step, steps_per_epoch):
                with (
                    stats_tracker.record_timing("e2e"),
                    stats_tracker.scope("grpo_actor"),
                ):
                    rollout.set_version(global_step)
                    logger.info(f"rollout set_version: {global_step} succeeded")

                    with (
                        stats_tracker.record_timing("weights_update_step"),
                        stats_tracker.scope("weights_update"),
                    ):
                        logger.info(
                            f"start to update weight, step: {step}, epoch: {epoch}"
                        )
                        weight_update_config.path = f"{config.storage_prefix}/checkpoints/{config.experiment_name}/{config.trial_name}/{global_step}"
                        if weight_update_config.type == "disk":
                            actor.upload_weights(weight_update_config)
                            weight_update_config.path = f"{config.storage_prefix}/checkpoints/{config.experiment_name}/{config.trial_name}/{global_step}"
                            rollout.update_weights(weight_update_config)
                            logger.info(
                                f"disk mode update weight succeeded, step: {step}, epoch: {epoch}"
                            )
                            clear_dir(weight_update_config.path)
                        else:
                            with concurrent.futures.ThreadPoolExecutor(
                                max_workers=2
                            ) as executor:
                                upload_future = executor.submit(
                                    actor.upload_weights, weight_update_config
                                )
                                update_future = executor.submit(
                                    rollout.update_weights, weight_update_config
                                )
                                wait_future_ordered([upload_future, update_future])
                            logger.info(
                                f"{weight_update_config.type} update weight succeeded, step: {step}, epoch: {epoch}"
                            )

                    # Update ref model if enabled
                    if (
                        ref is not None
                        and config.actor.hybrid_engine.wrap_policy.kl_ctl > 0
                        and config.ref.enable_update_ref_model
                        and (
                        global_step
                        * config.actor.hybrid_engine.wrap_policy.n_minibatches
                        % config.ref.update_ref_model_interval
                        == 0
                    )
                    ):
                        with stats_tracker.record_timing("update_ref_model"):
                            logger.info(
                                f"start to update ref model, step: {step}, epoch: {epoch}, global_step: {global_step}"
                            )
                            last_ref_model_path = ref_model_path
                            # Save actor weights to disk first
                            ref_model_path = f"{config.storage_prefix}/checkpoints/ref_update/{config.experiment_name}/{config.trial_name}/{global_step}"
                            actor_save_config = SaveLoadMeta(
                                path=ref_model_path,
                                weight_format="mcore",
                                global_step=global_step,
                                with_optim=True,
                                base_model_path=None,
                            )
                            actor.save(actor_save_config)
                            logger.info(
                                f"saved actor weights to {ref_model_path} for ref model update"
                            )

                            ref.update_ref_model(ref_model_path)
                            logger.info(
                                f"update ref model succeeded, step: {step}, epoch: {epoch}, global_step: {global_step}"
                            )

                    with (
                        stats_tracker.record_timing("rollout_step"),
                        stats_tracker.scope("rollout"),
                    ):
                        with (
                            stats_tracker.record_timing("notify_rollout_start_event"),
                        ):
                            logger.info(
                                f"start to notify_rollout_start_event, step: {step}, epoch: {epoch}"
                            )
                            rollout.notify_event("rollout_start", global_step)
                            logger.info(
                                f"notify_rollout_start_event succeeded, step: {step}, epoch: {epoch}"
                            )
                        with stats_tracker.record_timing("rollout_main"):
                            logger.info(
                                f"start to rollout, step: {step}, epoch: {epoch}"
                            )
                            workflow_kwargs = dict(
                                gconfig=config.gconfig,
                                tokenizer=config.tokenizer_path,
                                tool_call_parser=config.tool_call_parser,
                                reasoning_parser=config.reasoning_parser,
                                chat_template_type=(
                                    "concat"
                                    if config.export_style == "concat"
                                    else "hf"
                                ),
                                run_agent_return_reward_path=config.agent_module_path,
                                process_pool_executor_size=config.agent_process_pool_size,
                                dump_dir="generated",
                                rollout_stat_scope="rollout",
                                export_style=config.export_style,
                            )
                            if config.async_training:
                                batch = rollout.prepare_batch(
                                    dataloader,
                                    workflow_path="areal.extension.asystem.workflow.proxy.ProxyRLVRWorkflow",
                                    workflow_kwargs=workflow_kwargs,
                                )
                            else:
                                batch = rollout.rollout_batch(
                                    next(data_generator),
                                    workflow_path="areal.extension.asystem.workflow.proxy.ProxyRLVRWorkflow",
                                    workflow_kwargs=workflow_kwargs,
                                )

                    # with (
                    #     stats_tracker.scope("training_data"),
                    # ):
                    #     calc_training_data_metrics(batch.get_data())
                    #     calc_training_data_group_metrics(
                    #         batch.get_data, config.gconfig.n_samples
                    #     )
                    #     calc_training_data_version_metrics(batch.get_data, global_step)

                    logger.info(
                        "rollout batch reward summary: %s",
                        summarize_rewards(batch["rewards"]),
                    )
                    with (
                        stats_tracker.record_timing("notify_rollout_end_event"),
                    ):
                        logger.info(
                            f"start to notify_rollout_end_event, step: {step}, epoch: {epoch}"
                        )
                        rollout.notify_event("rollout_end", global_step)
                        logger.info(
                            f"notify_rollout_end_event succeeded, step: {step}, epoch: {epoch}"
                        )

                    if config.actor.hybrid_engine.wrap_policy.kl_ctl > 0:
                        with (
                            stats_tracker.record_timing("reference_step"),
                            stats_tracker.scope("reference"),
                        ):
                            logger.info(
                                f"start to compute_logprobs_with_distributed, step: {step}, epoch: {epoch}"
                            )
                            logp = ref.compute_logp(batch)
                            logp.to("cpu")
                            batch["ref_logprobs"] = logp
                            logger.info(
                                f"compute ref logprobs succeeded, step: {step}, epoch: {epoch}, ref logp shape: {logp.shape}"
                            )

                    with (
                        stats_tracker.record_timing("train_step"),
                        stats_tracker.scope("train"),
                    ):
                        with (
                            stats_tracker.record_timing("notify_train_start_event"),
                        ):
                            logger.info(
                                f"start to notify_train_start_event, step: {step}, epoch: {epoch}"
                            )
                            actor.notify_event("train_start", global_step)
                            logger.info(
                                f"notify_train_start_event succeeded, step: {step}, epoch: {epoch}"
                            )

                        with (
                            stats_tracker.record_timing("train_distributed_batch"),
                        ):
                            logger.info(f"start to train, step: {step}, epoch: {epoch}")
                            actor.train_batch(
                                batch,
                                loss_fn=lambda logits, batch_data: None,
                                loss_weight_fn=lambda batch_data: None,
                            )
                            logger.info(
                                f"train succeeded, step: {step}, epoch: {epoch}"
                            )

                        with stats_tracker.record_timing("latest_recover_save"):
                            if (
                                config.recover.latest_save_interval is not None
                                and global_step % config.recover.latest_save_interval
                                == 0
                                or global_step == config.total_train_steps
                            ):
                                logger.info(
                                    f"[Trainer] start save latest_checkpoint recover info, epoch:{epoch}, epoch_step: {step}, global_step:{global_step}"
                                )
                                latest_recover.save(
                                    actor,
                                    epoch,
                                    step,
                                    global_step,
                                    dataloader.state_dict(),
                                    "latest_checkpoint",
                                    disable_save_hf=config.recover.latest_disable_save_hf,
                                    ref_model_path=ref_model_path,
                                )
                                logger.info(
                                    f"[Trainer] latest_checkpoint recover save success, epoch:{epoch}, epoch_step: {step}, global_step:{global_step}"
                                )

                        with stats_tracker.record_timing("periodic_recover_save"):
                            if (
                                config.recover.periodic_save_interval is not None
                                and global_step > 0
                                and global_step % config.recover.periodic_save_interval
                                == 0
                                or global_step == config.total_train_steps
                            ):
                                logger.info(
                                    f"[Trainer] start save periodic_checkpoint recover info, epoch:{epoch}, epoch_step: {step}, global_step:{global_step}"
                                )
                                periodic_recover.save(
                                    actor,
                                    epoch,
                                    step,
                                    global_step,
                                    dataloader.state_dict(),
                                    "periodic_checkpoint",
                                    disable_save_hf=config.recover.periodic_disable_save_hf,
                                    ref_model_path="",
                                )
                                logger.info(
                                    f"[Trainer] periodic_checkpoint recover save success, epoch:{epoch}, epoch_step: {step}, global_step:{global_step}"
                                )

                        with (
                            stats_tracker.record_timing("notify_train_end_event"),
                        ):
                            logger.info(
                                f"start to notify_train_end_event, step: {step}, epoch: {epoch}"
                            )
                            actor.notify_event("train_end", global_step)
                            logger.info(
                                f"notify_train_end_event succeeded, step: {step}, epoch: {epoch}"
                            )

                        with (
                            stats_tracker.record_timing("cleanup_last_ref_resources"),
                        ):
                            clear_ref_resource(last_ref_model_path)

                metric = stats_tracker.export()
                stats_logger.commit(epoch, step, global_step, metric)
                global_step += 1
        stats_logger.close()
    except Exception as e:
        logger.info(
            "An error occurred during training, scheduler begin cleanup resource."
        )
        scheduler.cleanup_jobs()
        raise e


if __name__ == "__main__":
    main(sys.argv[1:])
