import resource
import sys
import time
import json
import os
import shutil
from typing import List, Dict, Any, Union

import torch
from datasets import load_dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tensordict import TensorDict, NonTensorData

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
from arealite.workflow.partial_rollout import PartialRolloutWorkflow
from arealite.api.cli_args import GenerationHyperparameters
from realhf.api.core.data_api import load_hf_tokenizer
from arealite.api.engine_api import WeightUpdateMeta
from arealite.extension.asystem.math_reward import reward_fn
from arealite.scheduler.asystem import AsystemScheduler
from arealite.utils.recover import Recover
from arealite.dataset.utils import ShuffleSampler
from arealite.controller.rollout_buffer import RolloutBuffer

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

weight_update_type = "nccl"  # nccl

if weight_update_type == "nccl":
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
    #----------------default value-----------------------------#,
    'tokenizer_mode': 'auto',
    'load_format': 'auto',
    'is_embedding': False,
    'kv_cache_dtype': 'auto',
    'max_prefill_tokens': 32768,
    'schedule_policy': 'fcfs',
    'schedule_conservativeness': 1.0,
    'disable_cuda_graph': False,
    'disable_radix_cache': True,
    'disable_cuda_graph_padding': False,
    'enable_nccl_nvls': False,
    'disable_outlines_disk_cache': False,
    'disable_overlap_schedule': False,
    'enable_mixed_chunk': False,
    'enable_dp_attention': False,
    'enable_ep_moe': False,
    'enable_torch_compile': False,
    'torch_compile_max_bs': 32,
    'triton_attention_reduce_in_fp32': False,
    'cuda_graph_bs': [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512],
    'num_continuous_decode_steps': 1,
    'enable_nan_detection': False,
    'allow_auto_truncate': False,
    'enable_p2p_check': False,
    'enable_memory_saver': False,
    'chunked_prefill_size': None,
    'context_length': None,
    'cpu_offload_gb': 0,
    'dp_size': 1,
    'dtype': 'auto',
    'sampling_backend': None,
    'log_level': 'info',
    'log_level_http': None,
    'log_requests': False,
    'log_requests_level': 0,
    'max_running_requests': None,
    'show_time_cost': False,
}
if weight_update_type == "nccl":
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
    "expert_model_parallel_size": 4,
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
    "load": "/storage/xukuan.xk/repos/antnlp/personal/pretrained_models/ring-moe-v2-sft-general700w_longcot200w_0725/iter_0028869",
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
    "moe_router_bias_update_rate": 0.00,
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
    "tokenizer_model": "/storage/xukuan.xk/repos/antnlp/personal/pretrained_models/ring-moe-v2-sft-general700w_longcot200w_0725/hf_ckpts/28869_kz",
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
if weight_update_type == "nccl":
    remote_megatron_config['asystem_train_config'] = asystem_hybrid_config


megatron_wrap_policy = {
    "n_minibatches": 1,
    "kl_ctl": 0.0,
    "adv_norm": False,
    "discount": 1.0,
    "gae_lambda": 1.0,
    "eps_clip": 0.2,
    "c_clip": None,
    "value_eps_clip": 0.2,
    "max_reward_clip": 5.0,
    "disable_value": True,
    "early_stop_kl": None,
    "early_stop_imp_ratio": None,
    "adaptive_kl_ctl": False,
    "adaptive_kl_target": 6,
    "adaptive_kl_horizon": 10000,
    "enable_save": True,
    "value_norm": True,
    "value_norm_type": "exp",
    "value_norm_beta": 0.99995,
    "value_norm_eps": 1e-5,
    "group_size": 8,
    "generation_size": None,
    "mask_no_eos_with_zero": False,
    "group_adv_norm": True,
    "mask_too_long": False,
    "use_dense_reward": False,
    "reward_delta": True,
    "token_normalize_scope": "global",
    "sample_reuse": 1,
    "temperature": 1.0,
    "reward_output_scaling": 0.5,
    "reward_output_bias": -1.0
}


def main_c3po():
    experiment_name = "arealite-mini-c3po"
    trial_name = "c3po-debug-zhongqing-5"

    # init scheduler
    scheduler = AsystemScheduler(
        {
            "endpoint": "http://asystem-scheduler.asystem-my001-swift.svc.sigma-my001.ml01.sgp-ml.local:8081",
            "expr_name": experiment_name,
            "trial_name": trial_name,
            "extra_envs": {
                "FUNCTIONCALL_SERVICE_DOMAIN": "http://110.75.237.19:8080",
                "REWARD_MODEL_PATH": "/storage/jiulin.jl/Skywork-Reward-V2-Qwen3-8B",
                "REWARD_MODEL_SERVICE_URL": "http://reward-model-service.asystem-test.svc.sigma-my001.ml01.sgp-ml.local:30000/classify"
            },
        }
    )

    batch_size = 64
    group_size = 8
    mini_samples_per_group = group_size
    batch_size_exceeding_num = 512 # 这个参数指的是额外发送给 rollout 的条数，不会再乘以 group_size，注意与 batch_size 定义不一致
    training_real_batch_size = batch_size * group_size
    staleness_version = 2
    model_path = "/storage/xukuan.xk/repos/antnlp/personal/pretrained_models/ring-moe-v2-sft-general700w_longcot200w_0725/hf_ckpts/28869_kz"
    max_prompt_len = 1024
    seed = 42

    ########### gconfig ####################
    force_no_logits_mask = True # asystem/0.1中已经废弃
    use_cuda_graph = True # asystem/0.1中已经废弃
    max_new_tokens = 15360
    min_new_tokens = 0
    temperature = 1.0
    top_k = 1000000
    top_p = 1.0
    greedy = False
    max_tokens = max_prompt_len + max_new_tokens
    #########################################

    step_num = 1145
    epoch_num = 10
    global_step = 0
    os.environ['WANDB_API_KEY'] = 'local-3bca3d5f00a980f3075b3e8ff2e16adc4ef43ffe'
    os.environ["WANDB_BASE_URL"] = "https://slurm.alipay.com"
    deploy_mode = "separation"
    allocation_mode = "gen:d4t4p1,train:d4t1p4"
    allocate_mode = AllocationMode.from_str(allocation_mode)
    storage_path = "/storage/openpsi/checkpoints/{experiment_name}/{trial_name}".format(
        experiment_name=experiment_name, trial_name=trial_name)

    tokenizer = load_hf_tokenizer(model_path)
    dataset = load_dataset("json",
                           data_files="/storage/dataset/nlp/areal/moe_lite_math_0527_merge_train_areal.jsonl")
    train_dataset = dataset['train']
    train_dataset = train_dataset.filter(lambda x: len(tokenizer.encode(x["prompt"])) <= max_prompt_len)
    dataloader = StatefulDataLoader(train_dataset, batch_size=1, sampler=ShuffleSampler(train_dataset))


    ############################## recover #########################################
    recover_meta_info_path = ""
    enable_recover = True
    freq_epochs = 1
    freq_steps = 20
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
    ##################################################################################

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
                                                           remote_megatron_config=remote_megatron_config, 
                                                           wrap_policy=RemoteMegatronEngineConfig.assign_wrap_policy(megatron_wrap_policy))),
        TrainControllerConfig(experiment_name=experiment_name, trial_name=trial_name, allocation_mode=allocation_mode),
        scheduler,
        group_size=1
    )
    rollout_buffer = RolloutBuffer(
        train_batch_size=training_real_batch_size,
        batch_size_exceeding_num=batch_size_exceeding_num,
        group_size=group_size,
        mini_samples_per_group=mini_samples_per_group,
        staleness_version=staleness_version
    )
    
    # engine initialize
    rollout.initialize()
    actor.initialize(colocation_with=rollout if deploy_mode == "colocation" else None)

    gconfig = GenerationHyperparameters(
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        max_tokens=max_tokens,
        greedy=greedy,
        n_samples=1, # must be 1 in PartialRolloutWorkflow
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    if tokenizer.pad_token_id not in gconfig.stop_token_ids:
        gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in gconfig.stop_token_ids:
        gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    workflow = PartialRolloutWorkflow(
        reward_fn=reward_fn,
        gconfig=gconfig,
        # tokenizer=tokenizer,
        tokenizer_path=model_path,
    )

    current_model_version = 0

    def add_res_to_rollout_buffer_callback(res: List[TensorDict]):
        assert isinstance(res, list), "add_res_to_rollout_buffer_callback should receive a list"
        logger.info(f"add {len(res)} results to rollout buffer")
        for r in res:
            assert isinstance(r, TensorDict), "add_res_to_rollout_buffer_callback should receive a list of TensorDict"
            rollout_buffer.add(r)
            logger.info(f"add sample query_id[{r['query_id'][0]}]_index_in_group[{r['index_in_group'][0]}] into rollout buffer, \
                        rollout_buffer current size: {rollout_buffer.get_current_size()}, ready to train sample num: {rollout_buffer.get_ready_to_train_sample_num()}")


    rollout.register_callback_to_all_worker("wait_at_least_no_concat", add_res_to_rollout_buffer_callback, batch_count=4, timeout=10)

    for epoch in range(recover_epoch, epoch_num):
        data_generator = iter(dataloader)
        start_step = recover_step + 1 if can_recover and epoch == recover_epoch else 0
        for step in range(start_step, step_num):
            with (
                stats_tracker.record_timing("e2e"),
                stats_tracker.scope("c3po_actor"),
            ):
                logger.info(f"start to train, step: {step}, epoch: {epoch}, global_step: {global_step}")
                rollout_buffer.expire_stale_samples(current_version=current_model_version)
                batch_data = rollout_buffer.pop_all_cached_samples()
                lack_samples = training_real_batch_size + batch_size_exceeding_num - len(batch_data)
                for _ in range((lack_samples + group_size  - 1) // group_size):
                    try:
                        batch = next(data_generator)
                    except StopIteration:
                        data_generator = iter(dataloader)
                        batch = next(data_generator)
                    for i in range(group_size):
                        new_batch = batch.copy()
                        new_batch['index_in_group'] = [str(i)]
                        batch_data.append(new_batch)
                weight_update_config = WeightUpdateMeta(
                    type=weight_update_type,
                    path=f"/storage/openpsi/checkpoints/{experiment_name}/{trial_name}",
                    alloc_mode=None,
                    comm_backend=None,
                    model_version=current_model_version,
                )

                with (
                    stats_tracker.record_timing("weights_update_step"),
                    stats_tracker.scope("weights_update"),
                ):
                    logger.info(f"start to update weight, step: {step}, epoch: {epoch}")
                    weight_update_config.path = f"/storage/openpsi/checkpoints/{experiment_name}/{trial_name}/{step}"
                    if weight_update_config.type == "disk":
                        actor.upload_weights(weight_update_config)
                        weight_update_config.path = f"/storage/openpsi/checkpoints/{experiment_name}/{trial_name}/{step}"
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
                        logger.info(f"start to rollout, step: {step}, epoch: {epoch}, batch_data len: {len(batch_data)}")

                        rollout.submit(batch_data, workflow=workflow)

                        while not rollout_buffer.is_sufficient():
                            time.sleep(0.3)

                        # 触发 abort，收集剩余的样本
                        rollout.abort_all_requests()
                        logger.info(f"rollout aborted")
                        
                        rollout_res = rollout_buffer.pop_batched_rollout_res() # 先 pop，pop 可能会耗时，刚好和中断 query 的时间折叠

                        # 本轮做训练的 rollout_res 已经 pop 出去了，所以只需要等待 rollout_buffer 中有 batch_size_exceeding_num 个元素，即可保证所有请求都已返回
                        while not rollout_buffer.current_has(rollout_buffer.batch_size_exceeding_num):
                            time.sleep(0.3)

                        logger.info(f"rollout succeeded {len(rollout_res)} samples, step: {step}, epoch: {epoch}")

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

                current_model_version += 1

            global_step += 1

    stats_logger.close()


if __name__ == "__main__":
    main_c3po()
