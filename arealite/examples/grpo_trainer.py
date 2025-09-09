import json
import os
import pprint
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import (
    GRPOConfig,
    RemoteHybridInferenceConfig,
    RemoteMegatronEngineConfig,
    RolloutControllerConfig,
    SaverConfig,
    StatsLoggerConfig,
    TensorBoardConfig,
    TrainControllerConfig,
    WandBConfig,
    load_expr_config,
)
from arealite.api.engine_api import WeightUpdateMeta
from arealite.api.io_struct import AllocationMode, FinetuneSpec
from arealite.controller.reference_controller import DistributedReferenceController
from arealite.controller.rollout_controller import DistributedRolloutController
from arealite.controller.train_controller import DistributedTrainController
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from arealite.dataset.utils import ShuffleSampler
from arealite.extension.asystem.math_reward import reward_fn
from arealite.extension.asystem.remote_hybrid_inference_worker import (
    RemoteHybridInferenceWorker,
)
from arealite.extension.asystem.remote_hybrid_train_worker import (
    RemoteHybridTrainWorker,
)
from arealite.recover import latest_checkpoint, periodic_checkpoint
from arealite.scheduler.asystem import AsystemScheduler
from arealite.utils.stats_logger import StatsLogger
from arealite.utils.util import clear_dir, custom_collate_fn
from arealite.workflow.rlvr import RLVRWorkflow
from realhf.api.core.data_api import load_hf_tokenizer
from arealite.api.engine_api import WeightUpdateMeta
from arealite.extension.asystem.math_reward import reward_fn
from arealite.scheduler.asystem import AsystemScheduler
from arealite.recover import periodic_checkpoint, latest_checkpoint
from arealite.dataset.utils import ShuffleSampler
from arealite.utils.metric import calc_training_data_metrics, calc_training_data_version_metrics, calc_training_data_group_metrics

from realhf.base import logging, stats_tracker

logger = logging.getLogger("Trainer")


def custom_collate_fn(batch):
    all_keys = set().union(*(d.keys() for d in batch))
    collated_batch = {}
    for key in all_keys:
        collated_batch[key] = [d.get(key) for d in batch]
    return collated_batch


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    config.gconfig.max_tokens = (
        config.gconfig.max_new_tokens + config.train_dataset.max_prompt_len
    )

    if config.enable_colocate_mode:
        config.rollout.engine_config["enable_memory_saver"] = True

    # Print full config
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
                "FUNCTIONCALL_SERVICE_DOMAIN": config.scheduler.functioncall_service_domain,
                "REWARD_FUNCTIONCALL_CONFIG": json.dumps(
                    config.scheduler.reward_functioncall_config
                ),
            },
        }
    )

    if config.weight_update_type != "disk":
        from arealite.extension.asystem.meta_server import start_meta_server

        host, port = start_meta_server()
        meta_server_addr = f"{host}:{port}"

        asystem_hybrid_config = {
            "meta_server_addr": meta_server_addr,
            "weights_exchange_comm_backend": config.weight_update_type,
            "weights_validation_steps": 0,
            "enable_debug_mode": True,
        }
        config.rollout.engine_config["asystem_hybrid_config"] = asystem_hybrid_config
        config.actor.hybrid_engine.remote_megatron_config["asystem_train_config"] = (
            asystem_hybrid_config
        )

    step_num = config.total_train_steps
    epoch_num = config.total_train_epochs
    global_step = 0
    os.environ["WANDB_API_KEY"] = config.stats_logger.wandb.wandb_api_key
    os.environ["WANDB_BASE_URL"] = config.stats_logger.wandb.wandb_base_url

    allocation_mode = config.allocation_mode
    allocate_mode = AllocationMode.from_str(allocation_mode)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    dataset = load_dataset("json", data_files=config.train_dataset.path)
    train_dataset = dataset["train"]
    train_dataset = train_dataset.filter(
        lambda x: len(tokenizer.encode(x["prompt"]))
        <= config.train_dataset.max_prompt_len
    )
    dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=1,
        sampler=ShuffleSampler(train_dataset),
        collate_fn=custom_collate_fn,
    )

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

    if can_recover:
        recover_epoch = recover_meta_info.epoch
        recover_global_step = recover_meta_info.global_step + 1
        recover_step = recover_meta_info.epoch_step + 1
        global_step = recover_meta_info.global_step + 1
        config.actor.hybrid_engine.global_step = global_step
        logger.info(
            f"🚀[Trainer] Recover success! global_step: {recover_meta_info.global_step}, epoch: {recover_meta_info.epoch}, step: {recover_meta_info.epoch_step}, "
            f"start new exp: global_step: {recover_global_step}, cur_step: {recover_step}"
        )
    latest_recover = latest_checkpoint.Recover(config.recover)
    if can_recover:
        dataloader.load_state_dict(recover_meta_info.dataloader_state)
        config.actor.hybrid_engine.recover_dir = recover_meta_info.checkpoint_path

    periodic_recover = periodic_checkpoint.Recover(config.recover)
    ##################################################################################

    stats_logger = StatsLogger(
        StatsLoggerConfig(
            experiment_name=config.experiment_name,
            trial_name=config.trial_name,
            fileroot=config.stats_logger.fileroot,
            wandb=WandBConfig(
                mode=config.stats_logger.wandb.mode,
                id_suffix=config.stats_logger.wandb.id_suffix,
            ),
            tensorboard=TensorBoardConfig(path=config.stats_logger.tensorboard.path),
        ),
        FinetuneSpec(
            total_train_epochs=epoch_num,
            dataset_size=step_num * config.train_bs_n_seqs,
            train_batch_size=config.train_bs_n_seqs,
        ),
    )
    stats_logger.info(f"[Trainer] total_epochs={epoch_num} step_per_epoch={step_num}")

    inference_config = config.rollout
    inference_config.dp_size = allocate_mode.gen_dp_size  # TODO: use allocate_mode
    inference_config.tp_size = allocate_mode.gen_tp_size
    inference_config.pp_size = allocate_mode.gen_pp_size
    inference_config.storage_path = (
        f"{config.rollout.storage_path}/{config.experiment_name}/{config.trial_name}"
    )
    inference_config.seed = config.seed
    rollout = DistributedRolloutController(
        RemoteHybridInferenceWorker(inference_config),
        RolloutControllerConfig(
            experiment_name=config.experiment_name,
            trial_name=config.trial_name,
            allocation_mode=config.allocation_mode,
            enable_colocate_mode=config.enable_colocate_mode,
            storage_prefix=config.storage_prefix,
        ),
        scheduler,
    )
    actor = DistributedTrainController(
        RemoteHybridTrainWorker(config.actor.hybrid_engine),
        TrainControllerConfig(
            experiment_name=config.experiment_name,
            trial_name=config.trial_name,
            allocation_mode=config.allocation_mode,
            enable_colocate_mode=config.enable_colocate_mode,
            group_size=config.actor.hybrid_engine.group_size,
            storage_prefix=config.storage_prefix,
        ),
        scheduler,
    )

    # engine initialize
    # initialize actor first for colocation mode
    # actor.initialize()
    # rollout.initialize(colocation_with=actor if config.enable_colocate_mode else None)

    ref = None
    if config.actor.hybrid_engine.wrap_policy.kl_ctl > 0:
        ref = DistributedReferenceController(
            RemoteHybridTrainWorker(config.ref.hybrid_engine),
            TrainControllerConfig(
                experiment_name=config.experiment_name,
                trial_name=config.trial_name,
                allocation_mode=config.allocation_mode,
                enable_colocate_mode=False,
                group_size=config.actor.hybrid_engine.group_size,
                storage_prefix=config.storage_prefix,
            ),
            scheduler,
        )
        # ref.initialize()

    # 共卡：actor -> rollout 按顺序，reference 可以并行。
    # 分卡：actor、rollout、reference 三者并行。

    # helper function for initializing Reference controller
    def init_ref_controller_helper(ref):
        if ref is not None:
            logger.info("ref is not none, initializing reference controller")
            ref.initialize()

    # helper function for initializing Train Controller (actor) & Rollout Controller
    def init_train_and_rollout_controller_helper(actor, rollout):
        logger.info("initializing trainer controller and rollout controller")
        actor.initialize()
        rollout.initialize(
            colocation_with=actor if config.enable_colocate_mode else None
        )

    # helper function for initializing Rollout controller
    def init_rollout_controller_helper(rollout):
        logger.info("initializing rollout controller")
        rollout.initialize(
            colocation_with=actor if config.enable_colocate_mode else None
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
                executor.submit(init_ref_controller_helper, ref),
            ]
            for future in futures:
                future.result()
        logger.info(
            f"initialized all controllers in colocation mode {config.enable_colocate_mode}"
        )
    else:
        logger.info(
            f"initializing all controllers in colocation mode {config.enable_colocate_mode}"
        )
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(actor.initialize),
                executor.submit(init_rollout_controller_helper, rollout),
                executor.submit(init_ref_controller_helper, ref),
            ]
            for future in futures:
                future.result()
        logger.info(
            f"initialized all controllers in colocation mode {config.enable_colocate_mode}"
        )

    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    workflow = RLVRWorkflow(
        reward_fn=reward_fn,
        gconfig=config.gconfig,
        tokenizer_path=config.tokenizer_path,
        exp_name=config.experiment_name,
        trial_name=config.trial_name,
    )

    for epoch in range(recover_epoch, epoch_num):
        data_generator = iter(dataloader)
        start_step = recover_step if can_recover and epoch == recover_epoch else 0
        for step in range(start_step, step_num):
            with (
                stats_tracker.record_timing("e2e"),
                stats_tracker.scope("grpo_actor"),
            ):
                batch_data = []
                for _ in range(config.train_bs_n_seqs):
                    try:
                        batch = next(data_generator)
                    except StopIteration:
                        data_generator = iter(dataloader)
                        batch = next(data_generator)
                    batch_data.append(batch)

                weight_update_config = WeightUpdateMeta(
                    type=config.weight_update_type,
                    path=f"/storage/openpsi/checkpoints/{config.experiment_name}/{config.trial_name}",
                    alloc_mode=None,
                    comm_backend=None,
                    model_version=global_step
                )

                with (
                    stats_tracker.record_timing("weights_update_step"),
                    stats_tracker.scope("weights_update"),
                ):
                    logger.info(f"start to update weight, step: {step}, epoch: {epoch}")
                    weight_update_config.path = f"/storage/openpsi/checkpoints/{config.experiment_name}/{config.trial_name}/{step}"
                    if weight_update_config.type == "disk":
                        actor.upload_weights(weight_update_config)
                        weight_update_config.path = f"/storage/openpsi/checkpoints/{config.experiment_name}/{config.trial_name}/{step}"
                        rollout.update_weights(weight_update_config)
                        logger.info(
                            f"disk mode update weight succeeded, step: {step}, epoch: {epoch}"
                        )
                        clear_dir(weight_update_config.path)
                    else:
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=2
                        ) as executor:
                            upload_future = executor.submit(
                                actor.upload_weights, weight_update_config
                            )
                            update_future = executor.submit(
                                rollout.update_weights, weight_update_config
                            )
                            concurrent.futures.wait([upload_future, update_future])
                            # Check for exceptions
                            for future in [upload_future, update_future]:
                                if future.exception() is not None:
                                    raise future.exception()
                        logger.info(
                            f"{weight_update_config.type} update weight succeeded (parallel), step: {step}, epoch: {epoch}"
                        )

                with (
                    stats_tracker.record_timing("rollout_step"),
                    stats_tracker.scope("rollout"),
                ):
                    with (stats_tracker.record_timing("notify_rollout_start_event"),):
                        logger.info(
                            f"start to notify_rollout_start_event, step: {step}, epoch: {epoch}"
                        )
                        rollout.notify_event("rollout_start", global_step)
                        logger.info(
                            f"notify_rollout_start_event succeeded, step: {step}, epoch: {epoch}"
                        )

                    with stats_tracker.record_timing("rollout_main"):
                        logger.info(f"start to rollout, step: {step}, epoch: {epoch}")
                        rollout_res = rollout.rollout(batch_data, workflow=workflow)
                        logger.info(f"rollout succeeded, step: {step}, epoch: {epoch}")

                with (stats_tracker.scope("training_data"),):
                    calc_training_data_metrics(rollout_res)
                    calc_training_data_group_metrics(rollout_res, config.gconfig.n_samples)
                    calc_training_data_version_metrics(rollout_res, global_step)

                with(stats_tracker.record_timing("post_data_process")):
                    rollout_res_dict = rollout_res.to_dict()
                    for k, v in rollout_res_dict.items():
                        if isinstance(v, torch.Tensor) and v.ndim > 1 and v.shape[0] == 1:
                            rollout_res_dict[k] = v.squeeze(0)
                    dis_batch = DistributedBatchMemory(rollout_res_dict)

                    with (
                        stats_tracker.record_timing("notify_rollout_end_event"),
                    ):
                        logger.info(f"start to notify_rollout_end_event, step: {step}, epoch: {epoch}")
                        rollout.notify_event("rollout_end", global_step)
                        logger.info(f"notify_rollout_end_event succeeded, step: {step}, epoch: {epoch}")

                if config.actor.hybrid_engine.wrap_policy.kl_ctl > 0:
                    with (
                        stats_tracker.record_timing("reference_step"),
                        stats_tracker.scope("reference"),
                    ):
                        logger.info(
                            f"start to compute_logprobs_with_distributed, step: {step}, epoch: {epoch}"
                        )
                        logp = ref.compute_logprobs_with_distributed(dis_batch)
                        logp.to("cpu")
                        rollout_res_dict["ref_logprobs"] = logp
                        dis_batch = DistributedBatchMemory(rollout_res_dict)
                        logger.info(
                            f"compute ref logprobs succeeded, step: {step}, epoch: {epoch}, ref logp shape: {logp.shape}"
                        )

                with (
                    stats_tracker.record_timing("train_step"),
                    stats_tracker.scope("train"),
                ):
                    with (stats_tracker.record_timing("notify_train_start_event"),):
                        logger.info(
                            f"start to notify_train_start_event, step: {step}, epoch: {epoch}"
                        )
                        actor.notify_event("train_start", global_step)
                        logger.info(
                            f"notify_train_start_event succeeded, step: {step}, epoch: {epoch}"
                        )

                    with (stats_tracker.record_timing("train_distributed_batch"),):
                        logger.info(f"start to train, step: {step}, epoch: {epoch}")
                        actor.train_distributed_batch(dis_batch)
                        logger.info(f"train succeeded, step: {step}, epoch: {epoch}")

                    with stats_tracker.record_timing("latest_recover_save"):
                        if (
                            config.recover.latest_save_interval is not None
                            and global_step % config.recover.latest_save_interval == 0
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
                            )
                            logger.info(
                                f"[Trainer] latest_checkpoint recover save success, epoch:{epoch}, epoch_step: {step}, global_step:{global_step}"
                            )

                    with stats_tracker.record_timing("periodic_recover_save"):
                        if (
                            config.recover.periodic_save_interval is not None
                            and global_step > 0
                            and global_step % config.recover.periodic_save_interval == 0
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
                            )
                            logger.info(
                                f"[Trainer] periodic_checkpoint recover save success, epoch:{epoch}, epoch_step: {step}, global_step:{global_step}"
                            )

                    with (stats_tracker.record_timing("notify_train_end_event"),):
                        logger.info(
                            f"start to notify_train_end_event, step: {step}, epoch: {epoch}"
                        )
                        actor.notify_event("train_end", global_step)
                        logger.info(
                            f"notify_train_end_event succeeded, step: {step}, epoch: {epoch}"
                        )

            metric = stats_tracker.export()
            stats_logger.commit(epoch, step, global_step, metric)
            global_step += 1

    stats_logger.close()


if __name__ == "__main__":
    main(sys.argv[1:])
