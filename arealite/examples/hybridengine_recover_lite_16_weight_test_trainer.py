import resource
import sys
import time
import os
import shutil

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
from arealite.api.cli_args import GenerationHyperparameters
from realhf.api.core.data_api import load_hf_tokenizer
from arealite.api.engine_api import WeightUpdateMeta
from arealite.extension.asystem.math_reward import reward_fn
from arealite.scheduler.asystem import AsystemScheduler
from arealite.utils.recover import Recover

from realhf.base import logging, stats_tracker


def clear_dir(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


logger = logging.getLogger("Trainer")

from arealite.extension.asystem.meta_server import start_meta_server
host, port = start_meta_server()
meta_server_addr = f"{host}:{port}"
print(f'meta_server_addr {meta_server_addr}')

asystem_hybrid_config = {
    "meta_server_addr": meta_server_addr,
    "weights_exchange_comm_backend": "nccl",
    "weights_validation_steps": 0,
    "enable_debug_mode": True,
}



engine_config = {
    "attention_backend": "triton",
    "disable_custom_all_reduce": True,
    "enable_metrics": True,
    "mem_fraction_static": 0.7,
    "triton_attention_num_kv_splits": 16,
    "disable_shared_experts_fusion": True,
}
engine_config['asystem_hybrid_config'] = asystem_hybrid_config

loss_configs = {
    "kl_ctl": 0.0,
    "adaptive_kl_target": 6,
    "adaptive_kl_horizon": 10000,
    "eps_clip": 0.2,
    "temperature": 1,
    "token_normalize_scope": "dp"
}

remote_megatron_config = {
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_eps": 1.0e-08,
    "add_bias_linear": False,
    "add_position_embedding": True,
    "apply_rope_fusion": True,
    "async_save": False,
    "attention_backend": "flash",
    "attention_dropout": 0.0,
    "attention_softmax_in_fp32": True,
    "auto_detect_ckpt_format": True,
    "bf16": True,
    "clip_grad": 1.0,
    "context_parallel_size": 1,
    "cp_comm_type": "p2p",
    "cross_entropy_loss_fusion": False,
    "distributed_backend": "nccl",
    "distributed_timeout_minutes": 600,
    "enable_one_logger": False,
    "expert_model_parallel_size": 8,
    "ffn_hidden_size": 1408,
    "global_batch_size": 8,
    "gradient_accumulation_fusion": True,
    "group_query_attention": True,
    "hidden_dropout": 0.0,
    "hidden_size": 2048,
    "init_method_std": 0.006,
    "load": "/storage/liuyongkang.lyk/output_models/moelite-32k-qwen3-640w-ep3-3e4-05250954/iter_0008604_asystem",
    "log_loss_scale_to_tensorboard": False,
    "log_num_zeros_in_grad": True,
    "log_params_norm": True,
    "log_throughput": True,
    "log_timers_to_tensorboard": True,
    "log_validation_ppl_to_tensorboard": True,
    "lr": 3.0e-06,
    "lr_decay_style": "constant",
    "lr_warmup_iters": 10,
    "make_vocab_size_divisible_by": 128,
    "masked_softmax_fusion": True,
    "max_position_embeddings": 32768,
    "micro_batch_size": 1,
    "moe_grouped_gemm": True,
    "moe_permute_fusion": True,
    "moe_router_dtype": "fp32",
    "moe_router_load_balancing_type": "aux_loss",
    "moe_router_topk": 6,
    "moe_shared_expert_intermediate_size": 2816,
    "moe_shared_expert_overlap": True,
    "moe_token_dispatcher_type": "alltoall",
    "norm_epsilon": 1.0e-06,
    "normalization": "RMSNorm",
    "num_attention_heads": 16,
    "num_experts": 64,
    "num_layers": 28,
    "num_query_groups": 4,
    "num_workers": 16,
    "optim_normhead_bwd_alltoall": False,
    "optim_normhead_fwd_alltoall": True,
    "optimizer": "adam",
    "overlap_grad_reduce": True,
    "overlap_p2p_comm": True,
    "overlap_param_gather": False,
    "pipeline_model_parallel_size": 1,
    "position_embedding_type": "rope",
    "qk_layernorm": False,
    "recompute_granularity": "full",
    "recompute_method": "uniform",
    "recompute_num_layers": 1,
    "resume_dataloader": False,
    "rotary_base": 600000,
    "router_warmup_step": 0,
    "save": "/mnt/asystem-m/common/users/user_name_placeholder/models/mcore_ckpt/",
    "save_interval": 1,
    "seed": 42,
    "seq_length": 32768,
    "sequence_parallel": True,
    "swiglu": True,
    "tensor_model_parallel_size": 1,
    "tensorboard_log_interval": 1,
    "tokenizer_model": "/storage/liuyongkang.lyk/output_models/moelite-32k-qwen3-640w-ep3-3e4-05250954/hf_ckpts/8604",
    "tokenizer_type": "HuggingFaceTokenizer",
    "train_iters": 100000,
    "untie_embeddings_and_output_weights": True,
    "use_distributed_optimizer": True,
    "use_flash_attn": True,
    "use_legacy_models": False,
    "use_mcore_models": True,
    "use_random_logits": True,
    "vocab_size": 126464,
    "weight_decay": 0.01,
}
remote_megatron_config['asystem_train_config'] = asystem_hybrid_config


def main_grpo():
    experiment_name = "arealite-lite"
    trial_name = "helloworld-astate-16x8-0"

    # init controller
    scheduler = AsystemScheduler({
        "endpoint": "http://asystem-scheduler.asystem-my001-swift.svc.sigma-my001.ml01.sgp-ml.local:8081",
        "expr_name": experiment_name,
        "trial_name": trial_name,
        "train": {
            "worker": {
                "image": "",
                "cmd": "",
                "extra_envs": {
                    "REAL_PACKAGE_PATH": "fff",
                },
            },
            "engine": {
                "image": "",
                "cmd": "",
                "extra_envs": {
                    "REAL_PACKAGE_PATH": "fff",
                },
            },
        },
        "rollout": {
            "worker": {
                "image": "",
                "cmd": "",
                "extra_envs": {
                    "REAL_PACKAGE_PATH": "fff",
                },
            },
            "engine": {
                "image": "",
                "cmd": "",
                "extra_envs": {
                    "REAL_PACKAGE_PATH": "fff",
                },
            },
        }
    })

    batch_size = 8
    group_size = 8
    model_path = "/storage/liuyongkang.lyk/output_models/moelite-32k-qwen3-640w-ep3-3e4-05250954/hf_ckpts/8604"
    max_prompt_len = 1024
    seed = 42
    max_new_tokens = 15360
    step_num = 1145
    epoch_num = 10
    global_step = 0
    os.environ['WANDB_API_KEY'] = 'local-3bca3d5f00a980f3075b3e8ff2e16adc4ef43ffe'
    os.environ["WANDB_BASE_URL"] = "https://slurm.alipay.com"
    deploy_mode = "separation"
    allocation_mode = "gen:d2t4p1,train:d8t1p1"
    allocate_mode = AllocationMode.from_str(allocation_mode)
    storage_path = "/storage/openpsi/checkpoints/{experiment_name}/{trial_name}".format(
        experiment_name=experiment_name, trial_name=trial_name)

    tokenizer = load_hf_tokenizer(model_path)
    dataset = load_dataset("json",
                           data_files="/storage/dataset/nlp/areal/moe_lite_math_0527_merge_train_areal.jsonl")
    train_dataset = dataset['train']
    train_dataset = train_dataset.filter(lambda x: len(tokenizer.encode(x["prompt"])) <= max_prompt_len)
    dataloader = StatefulDataLoader(train_dataset, batch_size=1)

    ############################## recover #########################################
    recover_meta_info_path = ""
    enable_recover = True
    freq_epochs = 1
    freq_steps = 2
    freq_secs = None

    recover_cfg = SaverConfig(
        experiment_name=experiment_name,
        trial_name=trial_name,
        fileroot="/storage/openpsi",
        freq_epochs=freq_epochs,
        freq_steps=freq_steps,
        freq_secs=freq_secs,
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
                      ft_spec=FinetuneSpec(total_train_epochs=epoch_num,
                                           dataset_size=step_num * batch_size,
                                           train_batch_size=batch_size))
    if can_recover:
        recover.load_ctl_states(recover_meta_info)
        dataloader.load_state_dict(recover_meta_info.dataloader_state)
        remote_megatron_config["load"] = recover_meta_info.checkpoint_path

    stats_logger = StatsLogger(StatsLoggerConfig(
        experiment_name=experiment_name, trial_name=trial_name, fileroot="/storage/openpsi/experiments",
        wandb=WandBConfig(
            mode="online",
        ),

    ), FinetuneSpec(total_train_epochs=epoch_num, dataset_size=step_num * batch_size, train_batch_size=batch_size))
    stats_logger.info(f"[Trainer] total_epochs={epoch_num} step_per_epoch={step_num}")

    rollout = DistributedRolloutController(
        RemoteHybridInferenceWorker(
            RemoteHybridInferenceConfig(experiment_name=experiment_name, trial_name=trial_name, model_path=model_path,
                                        storage_path=storage_path,
                                        dp_size=allocate_mode.gen_dp_size, tp_size=allocate_mode.gen_tp_size,
                                        pp_size=allocate_mode.gen_pp_size, seed=seed, engine_config=engine_config)),
        RolloutControllerConfig(experiment_name=experiment_name, trial_name=trial_name,
                                allocation_mode=allocation_mode),
        scheduler,
    )
    actor = DistributedTrainController(
        RemoteHypridTrainWorker(RemoteMegatronEngineConfig(experiment_name=experiment_name, trial_name=trial_name,
                                                           loss_configs=loss_configs,
                                                           remote_megatron_config=remote_megatron_config)),
        TrainControllerConfig(experiment_name=experiment_name, trial_name=trial_name, allocation_mode=allocation_mode),
        scheduler,
    )
    # engine initialize
    rollout.initialize()
    actor.initialize(colocation_with=rollout if deploy_mode == "colocation" else None)

    gconfig = GenerationHyperparameters(
        max_new_tokens=max_new_tokens, greedy=False, n_samples=group_size
    )

    if tokenizer.pad_token_id not in gconfig.stop_token_ids:
        gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in gconfig.stop_token_ids:
        gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    workflow = RLVRWorkflow(
        reward_fn=reward_fn,
        gconfig=gconfig,
        # tokenizer=tokenizer,
        tokenizer_path=model_path,
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
                for _ in range(batch_size):
                    try:
                        batch = next(data_generator)
                    except StopIteration:
                        data_generator = iter(dataloader)
                        batch = next(data_generator)
                    batch_data.append(batch)

                weight_update_config = WeightUpdateMeta(
                    type="nccl",
                    path=f"/storage/openpsi/checkpoints/{experiment_name}/{trial_name}",
                    alloc_mode=None,
                    comm_backend=None,
                )

                with (
                    stats_tracker.record_timing("weights_update_step"),
                    stats_tracker.scope("weights_update"),
                ):
                    logger.info(f"start to update weight, step: {step}, epoch: {epoch}")
                    weight_update_config.path = f"/storage/openpsi/checkpoints/{experiment_name}/{trial_name}/{step}"
                    import threading
                    t1 = threading.Thread(
                        target=actor.upload_weights,
                        kwargs={'meta': weight_update_config}
                    )
                    t2 = threading.Thread(
                        target=rollout.update_weights,
                        kwargs={'meta': weight_update_config}
                    )
                    t1.start()
                    t2.start()
                    t1.join()
                    t2.join()
                    logger.info(f"update weight succeeded, step: {step}, epoch: {epoch}")
                    clear_dir(weight_update_config.path)

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
    main_grpo()
