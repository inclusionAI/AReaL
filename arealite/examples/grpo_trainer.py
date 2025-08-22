import sys
import os
import pprint

import torch
from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import SaverConfig, \
    RolloutControllerConfig, TrainControllerConfig, RemoteMegatronEngineConfig, StatsLoggerConfig, WandBConfig, \
    RemoteHybridInferenceConfig
from arealite.api.io_struct import FinetuneSpec, AllocationMode
from arealite.controller.rollout_controller import DistributedRolloutController
from arealite.controller.train_controller import DistributedTrainController
from arealite.extension.asystem.remote_hybrid_inference_worker import RemoteHybridInferenceWorker
from arealite.extension.asystem.remote_hyprid_train_worker import RemoteHypridTrainWorker
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from arealite.utils.stats_logger import StatsLogger
from arealite.workflow.rlvr import RLVRWorkflow
from realhf.api.core.data_api import load_hf_tokenizer
from arealite.api.engine_api import WeightUpdateMeta
from arealite.extension.asystem.math_reward import reward_fn
from arealite.scheduler.asystem import AsystemScheduler
from arealite.utils.recover import Recover
from arealite.dataset.utils import ShuffleSampler

from realhf.base import logging, stats_tracker
from arealite.api.cli_args import GRPOConfig, load_expr_config
from arealite.utils.util import clear_dir

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
    
    # Print full config
    logger.info("Loaded config:\n" + pprint.pformat(config, indent=2, width=120, depth=6))

    # init scheduler
    scheduler = AsystemScheduler(
        {
            "endpoint": config.scheduler.endpoint,
            "expr_name": config.experiment_name,
            "trial_name": config.trial_name,
            "extra_envs": {
                "FUNCTIONCALL_SERVICE_DOMAIN": config.scheduler.functioncall_service_domain,
                # "REWARD_MODEL_PATH": config.scheduler.reward_model_path,
                # "REWARD_MODEL_SERVICE_URL": config.scheduler.reward_model_service_url
            },
        }
    )

    if config.weight_update_type == "nccl":
        from arealite.extension.asystem.meta_server import start_meta_server
        host, port = start_meta_server()
        meta_server_addr = f"{host}:{port}"

        asystem_hybrid_config = {
            "meta_server_addr": meta_server_addr,
            "weights_exchange_comm_backend": "nccl",
            "weights_validation_steps": 0,
            "enable_debug_mode": True,
        }
        config.rollout.engine_config['asystem_hybrid_config'] = asystem_hybrid_config
        config.actor.hybrid_engine.remote_megatron_config['asystem_train_config'] = asystem_hybrid_config

    step_num = config.total_train_steps
    epoch_num = config.total_train_epochs
    global_step = 0
    os.environ['WANDB_API_KEY'] = config.stats_logger.wandb.wandb_api_key
    os.environ["WANDB_BASE_URL"] = config.stats_logger.wandb.wandb_base_url
    deploy_mode = config.scheduler.deploy_mode
    allocation_mode = config.allocation_mode
    allocate_mode = AllocationMode.from_str(allocation_mode)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    dataset = load_dataset("json", data_files=config.train_dataset.path)
    train_dataset = dataset['train']
    train_dataset = train_dataset.filter(
        lambda x: len(tokenizer.encode(x["prompt"])) <= config.train_dataset.max_prompt_len)
    dataloader = StatefulDataLoader(train_dataset, batch_size=1, sampler=ShuffleSampler(train_dataset), collate_fn=custom_collate_fn)

    ############################## recover #########################################
    recover_meta_info_path = config.recover.recover_meta_info_path
    enable_recover = True

    recover_cfg = SaverConfig(
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
        fileroot=config.recover.fileroot,
        freq_epochs=config.recover.freq_epochs,
        freq_steps=config.recover.freq_steps,
        freq_secs=config.recover.freq_secs,
    )
    can_recover = False
    recover_epoch = 0
    recover_step = 0
    if recover_meta_info_path != "" and enable_recover:
        can_recover, recover_meta_info = Recover.load(recover_meta_info_path)

    if can_recover:
        recover_epoch = recover_meta_info.epoch
        recover_global_step = recover_meta_info.global_step + 1
        recover_step = recover_meta_info.epoch_step + 1
        logger.info(
            f"🚀[Trainer] Recover success! global_step: {recover_meta_info.global_step}, epoch: {recover_meta_info.epoch}, step: {recover_meta_info.epoch_step}, "
            f"start new exp: global_step: {recover_global_step}, cur_step: {recover_step}")
    recover = Recover(config=recover_cfg,
                      ft_spec=FinetuneSpec(total_train_epochs=config.total_train_epochs,
                                           dataset_size=step_num * config.train_bs_n_seqs,
                                           train_batch_size=config.train_bs_n_seqs))
    if can_recover:
        recover.load_ctl_states(recover_meta_info)
        dataloader.load_state_dict(recover_meta_info.dataloader_state)
        config.actor.hybrid_engine.remote_megatron_config["recover_dir"] = recover_meta_info.checkpoint_path
    ##################################################################################

    stats_logger = StatsLogger(StatsLoggerConfig(
        experiment_name=config.experiment_name, trial_name=config.trial_name,
        fileroot=config.stats_logger.fileroot,
        wandb=WandBConfig(
            mode=config.stats_logger.wandb.mode,
        ),

    ), FinetuneSpec(total_train_epochs=epoch_num,
                    dataset_size=step_num * config.train_bs_n_seqs,
                    train_batch_size=config.train_bs_n_seqs))
    stats_logger.info(f"[Trainer] total_epochs={epoch_num} step_per_epoch={step_num}")

    storage_path = f"{config.rollout.storage_path}/{config.experiment_name}/{config.trial_name}"
    rollout = DistributedRolloutController(
        RemoteHybridInferenceWorker(
            RemoteHybridInferenceConfig(experiment_name=config.experiment_name, trial_name=config.trial_name,
                                        model_path=config.rollout.model_path,
                                        storage_path=storage_path,
                                        dp_size=allocate_mode.gen_dp_size, tp_size=allocate_mode.gen_tp_size,
                                        pp_size=allocate_mode.gen_pp_size, seed=config.seed,
                                        engine_config=config.rollout.engine_config)),
        RolloutControllerConfig(experiment_name=config.experiment_name, trial_name=config.trial_name,
                                allocation_mode=config.allocation_mode),
        scheduler,
    )
    actor = DistributedTrainController(
        RemoteHypridTrainWorker(
            RemoteMegatronEngineConfig(
                experiment_name=config.experiment_name,
                trial_name=config.trial_name,
                loss_configs=config.actor.hybrid_engine.loss_configs,
                remote_megatron_config=config.actor.hybrid_engine.remote_megatron_config,
                wrap_policy=config.actor.hybrid_engine.wrap_policy,
                max_tokens_per_mb=config.actor.hybrid_engine.max_tokens_per_mb,
                group_size=config.actor.hybrid_engine.group_size)
        ),
        TrainControllerConfig(experiment_name=config.experiment_name,
                              trial_name=config.trial_name,
                              allocation_mode=config.allocation_mode),
        scheduler,
        group_size=config.actor.hybrid_engine.group_size
    )

    # engine initialize
    rollout.initialize()
    actor.initialize(colocation_with=rollout if deploy_mode == "colocation" else None)

    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    workflow = RLVRWorkflow(
        reward_fn=reward_fn,
        gconfig=config.gconfig,
        tokenizer_path=config.tokenizer_path,
    )

    for epoch in range(recover_epoch, epoch_num):
        data_generator = iter(dataloader)
        start_step = recover_step + 1 if can_recover and epoch == recover_epoch else 0
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
                        logger.info(f"disk mode update weight succeeded, step: {step}, epoch: {epoch}")
                        clear_dir(weight_update_config.path)
                    elif weight_update_config.type == "nccl":
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                            upload_future = executor.submit(actor.upload_weights, weight_update_config)
                            update_future = executor.submit(rollout.update_weights, weight_update_config)
                            concurrent.futures.wait([upload_future, update_future])
                            # Check for exceptions
                            for future in [upload_future, update_future]:
                                if future.exception() is not None:
                                    raise future.exception()
                        logger.info(f"nccl mode update weight succeeded (parallel), step: {step}, epoch: {epoch}")

                with (
                    stats_tracker.record_timing("rollout_step"),
                    stats_tracker.scope("rollout"),
                ):
                    with (
                        stats_tracker.record_timing("notify_rollout_start_event"),
                        stats_tracker.scope("rollout"),
                    ):
                        logger.info(f"start to notify_rollout_start_event, step: {step}, epoch: {epoch}")
                        rollout.notify_event("rollout_start", global_step)
                        logger.info(f"notify_rollout_start_event succeeded, step: {step}, epoch: {epoch}")

                    with(stats_tracker.record_timing("main")):
                        logger.info(f"start to rollout, step: {step}, epoch: {epoch}")
                        rollout_res = rollout.rollout(batch_data, workflow=workflow)
                        logger.info(f"rollout succeeded, step: {step}, epoch: {epoch}")

                    with(stats_tracker.record_timing("post_data_process")):
                        rollout_res_dict = rollout_res.to_dict()
                        for k, v in rollout_res_dict.items():
                            if isinstance(v, torch.Tensor) and v.ndim > 1 and v.shape[0] == 1:
                                rollout_res_dict[k] = v.squeeze(0)
                        torch.set_printoptions(threshold=float('inf'))
                        logger.info(f"after rollout rewards: {rollout_res_dict["rewards"]}")
                        dis_batch = DistributedBatchMemory(rollout_res_dict)

                        with (
                            stats_tracker.record_timing("notify_rollout_end_event"),
                            stats_tracker.scope("rollout"),
                        ):
                            logger.info(f"start to notify_rollout_end_event, step: {step}, epoch: {epoch}")
                            rollout.notify_event("rollout_end", global_step)
                            logger.info(f"notify_rollout_end_event succeeded, step: {step}, epoch: {epoch}")

                with (
                    stats_tracker.record_timing("train_step"),
                    stats_tracker.scope("train"),
                ):
                    with (
                        stats_tracker.record_timing("notify_train_start_event"),
                        stats_tracker.scope("train"),
                    ):
                        logger.info(f"start to notify_train_start_event, step: {step}, epoch: {epoch}")
                        actor.notify_event("train_start", global_step)
                        logger.info(f"notify_train_start_event succeeded, step: {step}, epoch: {epoch}")

                    logger.info(f"start to train, step: {step}, epoch: {epoch}")
                    actor.train_distributed_batch(dis_batch)
                    logger.info(f"train succeeded, step: {step}, epoch: {epoch}")

                    with stats_tracker.record_timing("recover_save"):
                        if recover.freq_ctl.check(
                            epochs=int(step == recover.ft_spec.steps_per_epoch - 1), steps=1
                        ):
                            logger.info(
                                f"[Trainer] start save recover info, epoch:{epoch}, epoch_step: {step}, global_step:{global_step}")
                            recover.save(actor, epoch, step, global_step, dataloader.state_dict())
                            logger.info(
                                f"[Trainer] recover save success, epoch:{epoch}, epoch_step: {step}, global_step:{global_step}")

                    with (
                        stats_tracker.record_timing("notify_train_end_event"),
                        stats_tracker.scope("train"),
                    ):
                        logger.info(f"start to notify_train_end_event, step: {step}, epoch: {epoch}")
                        actor.notify_event("train_end", global_step)
                        logger.info(f"notify_train_end_event succeeded, step: {step}, epoch: {epoch}")

                metric = stats_tracker.export()
                stats_logger.commit(epoch, step, global_step, metric)

            global_step += 1

    stats_logger.close()


if __name__ == "__main__":
    main(sys.argv[1:])


