import resource
import sys
import time

import torch
from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import load_expr_config, BaseExperimentConfig, InferenceEngineConfig, TrainEngineConfig, \
    RolloutControllerConfig, TrainControllerConfig, RemoteMegatronEngineConfig, StatsLoggerConfig, WandBConfig, \
    RemoteHybridInferenceConfig
from arealite.api.io_struct import FinetuneSpec, AllocationMode
from arealite.controller.rollout_controller import DistributedRolloutController
from arealite.controller.train_controller import DistributedTrainController
from arealite.extension.asystem.remote_hybrid_inference_worker import RemoteHybridInferenceWorker
from arealite.extension.asystem.remote_hyprid_train_worker import RemoteHypridTrainWorker
from arealite.extension.asystem.remote_megatron_engine import RemoteMegatronEngine
from arealite.extension.asystem.remote_sglang_engine import RemoteSGLangEngine
from arealite.scheduler.local import LocalScheduler
from arealite.dataset.distributed_batch_memory import DistributedBatchMemory
from arealite.utils.stats_logger import StatsLogger
from arealite.workflow.rlvr import RLVRWorkflow
from arealite.api.cli_args import GenerationHyperparameters
from realhf.api.core.data_api import load_hf_tokenizer
from arealite.api.engine_api import WeightUpdateMeta
from arealite.extension.asystem.math_reward import reward_fn
from arealite.scheduler.asystem import AsystemScheduler
import os
import shutil

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

engine_config = {
    "attention_backend": "triton",
    "disable_custom_all_reduce": True,
    "enable_metrics": True,
    "mem_fraction_static": 0.7,
    "triton_attention_num_kv_splits": 16
}

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
    "adaptive_layer_bias_update_strategy": "sqrt",
    "add_bias_linear": False,
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
    "expert_tensor_parallel_size": 1,
    "ffn_hidden_size": 5120,
    "first_k_dense_replace": 1,
    "global_batch_size": 512,
    "global_step": 1,
    "gradient_accumulation_fusion": True,
    "group_query_attention": True,
    "hidden_dropout": 0.0,
    "hidden_size": 2048,
    "init_method_std": 0.006,
    "load": "/storage/xukuan.xk/repos/antnlp/personal/pretrained_models/moe-mini-v2-e256-0627-fp8-32k-constant-merge-pack-merge-mean_w8",
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
    "max_position_embeddings": 16384,
    "micro_batch_size": 1,
    "moe_ffn_hidden_size": 512,
    "moe_grouped_gemm": True,
    "moe_layer_freq": [
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ],
    "moe_per_layer_logging": True,
    "moe_permute_fusion": True,
    "moe_router_bias_update_rate": 0.001,
    "moe_router_dtype": "fp32",
    "moe_router_enable_expert_bias": True,
    "moe_router_group_topk": 4,
    "moe_router_num_groups": 8,
    "moe_router_score_function": "sigmoid",
    "moe_router_topk": 8,
    "moe_router_topk_scaling_factor": 2.5,
    "moe_shared_expert_intermediate_size": 512,
    "moe_shared_expert_overlap": True,
    "moe_token_dispatcher_type": "alltoall",
    "norm_epsilon": 1.0e-06,
    "normalization": "RMSNorm",
    "num_attention_heads": 16,
    "num_experts": 256,
    "num_layers": 20,
    "num_query_groups": 4,
    "optim_normhead_fwd_alltoall": True,
    "optimizer": "adam",
    "overlap_grad_reduce": True,
    "overlap_p2p_comm": True,
    "overlap_param_gather": False,
    "pipeline_model_parallel_size": 4,
    "position_embedding_type": "rope",
    "qk_layernorm": True,
    "recompute_granularity": "full",
    "recompute_method": "uniform",
    "recompute_num_layers": 5,
    "rotary_base": 600000,
    "rotary_percent": 0.5,
    "save": "/mnt/asystem-s3/common/users/senlin.zsl/experiments/2025-07-19_14-32-43/experiments/models/mcore_ckpt_32/asystem_moe_mini",
    "save_interval": 1,
    "seed": 42,
    "seq_length": 16384,
    "sequence_parallel": True,
    "skip_casting_dtype_for_param_pattern": "^expert_bias$|.+\\.expert_bias$",
    "swiglu": True,
    "tensor_model_parallel_size": 1,
    "tensorboard_log_interval": 1,
    "tokenizer_model": "/storage/xukuan.xk/repos/antnlp/personal/pretrained_models/moe-mini-v2-e256-0627-fp8-32k-constant-merge-pack-merge-mean_w8/sglang_iter_0011770",
    "tokenizer_type": "HuggingFaceTokenizer",
    "train_iters": 100000,
    "transformer_xl": False,
    "unidirectional": True,
    "untie_embeddings_and_output_weights": True,
    "use_distributed_optimizer": True,
    "use_flash_attn": True,
    "use_init_chunk": True,
    "use_mcore_models": True,
    "use_norm_head": False,
    "use_pack_lazy_loader": True,
    "use_random_logits": True,
    "use_rotary_position_embeddings": True,
    "vocab_size": 157184,
    "weight_decay": 0.01,
}


def main_grpo():
    experiment_name = "arealite-mini"
    trial_name = "helloworld-64x8-0801-0"

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

    dataset = load_dataset("json",
                           data_files="/storage/dataset/nlp/areal/moe_lite_math_0527_merge_train_areal.jsonl")
    train_dataset = dataset['train']
    dataloader = StatefulDataLoader(train_dataset, batch_size=1)
    batch_size = 64
    group_size = 8
    MODEL_PATH = "/storage/xukuan.xk/repos/antnlp/personal/pretrained_models/moe-mini-v2-e256-0627-fp8-32k-constant-merge-pack-merge-mean_w8/sglang_iter_0011770"
    max_prompt_len = 1024
    max_new_tokens = 15360
    step_num = 1145
    epoch_num = 10
    global_step = 0
    os.environ['WANDB_API_KEY'] = 'local-3bca3d5f00a980f3075b3e8ff2e16adc4ef43ffe'
    os.environ["WANDB_BASE_URL"] = "https://slurm.alipay.com"
    deploy_mode = "separation"
    allocation_mode = "sglang.d4t8p1+d32t1p1"
    allocate_mode = AllocationMode.from_str(allocation_mode)
    storage_path = "/storage/openpsi/checkpoints/{experiment_name}/{trial_name}".format(
        experiment_name=experiment_name, trial_name=trial_name)

    stats_logger = StatsLogger(StatsLoggerConfig(
        experiment_name=experiment_name, trial_name=trial_name,fileroot="/storage/openpsi/experiments",
        wandb=WandBConfig(
            mode="online",
        ),

    ), FinetuneSpec(total_train_epochs=epoch_num, dataset_size=step_num * batch_size ,train_batch_size=batch_size))
    stats_logger.info(f"total_epochs={epoch_num} step_per_epoch={step_num}")

    rollout = DistributedRolloutController(
        RemoteHybridInferenceWorker(RemoteHybridInferenceConfig(experiment_name=experiment_name, trial_name=trial_name, model_path=MODEL_PATH, storage_path=storage_path,
                                                                dp_size=allocate_mode.gen_dp_size, tp_size=allocate_mode.gen_tp_size, pp_size=allocate_mode.gen_pp_size, engine_config=engine_config)),
        RolloutControllerConfig(experiment_name=experiment_name, trial_name=trial_name, allocation_mode=allocation_mode),
        scheduler,
    )
    actor = DistributedTrainController(
        RemoteHypridTrainWorker(RemoteMegatronEngineConfig(experiment_name=experiment_name, trial_name=trial_name,
                                                           loss_configs=loss_configs, remote_megatron_config=remote_megatron_config)),
        TrainControllerConfig(experiment_name=experiment_name, trial_name=trial_name, allocation_mode=allocation_mode),
        scheduler,
    )
    # engine initialize
    rollout.initialize()
    actor.initialize(colocation_with=rollout if deploy_mode == "colocation" else None)

    tokenizer = load_hf_tokenizer(MODEL_PATH)
    gconfig = GenerationHyperparameters(
        max_new_tokens=max_new_tokens, greedy=False, n_samples=group_size
    )

    workflow = RLVRWorkflow(
        reward_fn=reward_fn,
        gconfig=gconfig,
        # tokenizer=tokenizer,
        tokenizer_path=MODEL_PATH
    )

    for epoch in range(epoch_num):
        data_generator = iter(dataloader)
        for step in range(step_num):
            with (
                stats_tracker.record_timing("e2e"),
                stats_tracker.scope("grpo_actor"),
            ):
                batch_data = []
                while len(batch_data) < batch_size:
                    batch = next(data_generator)
                    prompt = batch["prompt"] if isinstance(batch, dict) else batch[0]["prompt"]
                    tokenized = tokenizer(prompt, truncation=False, return_length=True)
                    if tokenized["length"][0] <= max_prompt_len:
                        batch_data.append(batch)
                    else:
                        logger.warning(f"Ignored prompt with length {tokenized['length'][0]} > {max_prompt_len}")

                if len(batch_data) < batch_size:
                    break

                weight_update_config = WeightUpdateMeta(
                    type="disk",
                    path=f"/storage/openpsi/checkpoints/{experiment_name}/{trial_name}",
                    alloc_mode=None,
                    comm_backend=None,
                )

                with (
                    stats_tracker.record_timing("weights_update_step"),
                    stats_tracker.scope("weights_update"),
                ):
                    logger.info(f"start to update weight, step: {step}, epoch: {epoch}")
                    actor.upload_weights(weight_update_config)
                    weight_update_config.path = f"/storage/openpsi/checkpoints/{experiment_name}/{trial_name}/{step}"
                    rollout.update_weights(weight_update_config)
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


