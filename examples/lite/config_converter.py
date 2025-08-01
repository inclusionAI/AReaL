import abc
import argparse
import os
import shlex
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

import yaml
from deepmerge import always_merger
from termcolor import colored

CVRT_WARNING = "This parameter has no exact mapping in areal"
FEAT_WARNING = (
    "This parameter is not supported in areal, but should be supported in the future"
)


@dataclass
class ArgSpec:

    arg: str = ""
    arg_type: Type[Any] = field(default=str)
    description: str = ""
    map_fn: Optional[Callable] = field(default=None)


class Converter(abc.ABC):
    """
    Base class for converters.
    """

    def parse(self) -> dict:
        """
        Parse the arguments from other framework format to a areal config.
        """

    def convert(self) -> dict:
        """
        Convert the arguments to a new format.
        """

    def get_lite_template(self, template_path: str):
        """
        Load the areal template from the specified file.
        """
        with open(template_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def flatten_dict(self, d, parent_key="", sep="."):
        def _flatten_dict(d, parent_key="", sep="."):
            """
            flatten a nested dictionary into a single-level dictionary
            """
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        return _flatten_dict(d, parent_key, sep)

    def convert_to_nested_args(self, args, ARG_MAP: dict) -> dict:
        """
        Convert flat arguments to nested dict based on ARG_MAP.
        alert warning: This function will print warnings for unmapped arguments.
        """
        cfg = {}
        unmapped = {}
        for k, v in args.items():
            if k in ARG_MAP:
                argspec = ARG_MAP[k]
                if not argspec.arg:
                    print(
                        colored(f"## Warning: For ", "yellow")
                        + colored(f"{k:>40}", "yellow", attrs=["bold"])
                        + colored(f",    # {argspec.description}!", "yellow")
                    )
                    continue

                keys = argspec.arg.split(".")
                arg_type = argspec.arg_type
                map_fn = argspec.map_fn

                if v is not None:
                    try:
                        # type conversion
                        if arg_type == bool:
                            v = (
                                bool(v)
                                if isinstance(v, bool)
                                else v.lower() in ("1", "true", "yes", "on")
                            )
                        elif arg_type == int:
                            v = int(v)
                        elif arg_type == float:
                            v = float(v)
                        elif arg_type == str:
                            v = str(v)
                        else:
                            raise ValueError(f"Unsupported type: {arg_type}")
                    except Exception as e:
                        print(
                            colored(f"## Error: For ", "red")
                            + colored(f"{k:>40} {v}", "red", attrs=["bold"])
                            + colored(f",    # {e}!", "red")
                        )
                        exit(-1)

                # apply map function if exists
                if map_fn:
                    v = map_fn(v)

                self.set_nested(cfg, keys, v)
            else:
                unmapped[k] = v
                print(
                    colored(f"## Warning: For ", "yellow")
                    + colored(f"{k:>50}", "yellow", attrs=["bold"])
                    + colored(f",    # {CVRT_WARNING}!", "yellow")
                )
        # print("Unmapped arguments:", list(unmapped.keys()))

        return cfg

    def set_nested(self, d: dict, keys, value):
        """
        Set a value in a nested dictionary using a list of keys.
        """
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value


class OpenRLHFConverter(Converter):

    ARG_MAP = {
        # Ray and vLLM
        "ref_num_nodes": ArgSpec("", int, CVRT_WARNING),
        "ref_num_gpus_per_node": ArgSpec("", int, CVRT_WARNING),
        "reward_num_nodes": ArgSpec("", int, CVRT_WARNING),
        "reward_num_gpus_per_node": ArgSpec("", int, CVRT_WARNING),
        "colocate_actor_ref": ArgSpec("", bool, CVRT_WARNING),
        "actor_num_nodes": ArgSpec("cluster.n_nodes", int, ""),
        "actor_num_gpus_per_node": ArgSpec("cluster.n_gpus_per_node", int, ""),
        "critic_num_nodes": ArgSpec("", int, CVRT_WARNING),
        "critic_num_gpus_per_node": ArgSpec("", int, CVRT_WARNING),
        "colocate_critic_reward": ArgSpec("", bool, CVRT_WARNING),
        "colocate_all_models": ArgSpec("", bool, CVRT_WARNING),
        "vllm_num_engines": ArgSpec("allocation_mode.sglang.d", int, ""),
        "vllm_tensor_parallel_size": ArgSpec("allocation_mode.sglang.t", int, ""),
        "vllm_sync_backend": ArgSpec("", str, FEAT_WARNING),
        "vllm_sync_with_ray": ArgSpec("", bool, CVRT_WARNING),
        "enable_prefix_caching": ArgSpec("", bool, CVRT_WARNING),
        "enforce_eager": ArgSpec("", bool, CVRT_WARNING),
        "vllm_enable_sleep": ArgSpec("", bool, CVRT_WARNING),
        "vllm_gpu_memory_utilization": ArgSpec(
            "sglang.mem_fraction_static", float, "Not equivalent to SGLang"
        ),
        # Async training
        "async_train": ArgSpec(
            "async_training", bool
        ),  # TODO: convert areal offpolicyness > 0
        # Checkpoints
        "eval_steps": ArgSpec("", int, CVRT_WARNING),
        "save_steps": ArgSpec("", int, FEAT_WARNING),
        "logging_steps": ArgSpec("", int, CVRT_WARNING),
        "ckpt_path": ArgSpec(
            "cluster.fileroot", str, "", lambda x: os.path.dirname(x)
        ),  # TODO: convert to areal's cluster.fileroot
        "save_hf_ckpt": ArgSpec("", bool, CVRT_WARNING),
        "disable_ds_ckpt": ArgSpec("", bool, CVRT_WARNING),
        "max_ckpt_num": ArgSpec("", int, CVRT_WARNING),
        "max_ckpt_mem": ArgSpec("", int, CVRT_WARNING),
        "load_checkpoint": ArgSpec("", bool, CVRT_WARNING),
        "use_ds_universal_ckpt": ArgSpec("", bool, CVRT_WARNING),
        # DeepSpeed
        "local_rank": ArgSpec("", int, CVRT_WARNING),
        "zero_stage": ArgSpec("", int, CVRT_WARNING),
        "gradient_checkpointing": ArgSpec("actor.gradient_checkpointing", bool, ""),
        "deepcompile": ArgSpec("", bool, CVRT_WARNING),
        "bf16": ArgSpec(
            "actor.dtype", str, "", lambda x: "bfloat16" if x else "float16"
        ),
        "enable_ema": ArgSpec("", bool, CVRT_WARNING),
        "ema_beta": ArgSpec("", float, CVRT_WARNING),
        "zpg": ArgSpec("", int, CVRT_WARNING),
        "adam_offload": ArgSpec("actor.optimizer.offload", bool, ""),
        "actor_init_on_gpu": ArgSpec("", bool, CVRT_WARNING),
        "flash_attn": ArgSpec(
            "actor.attn_impl",
            str,
            "default use flash_attention_2",
            lambda: "flash_attention_2",
        ),
        "use_liger_kernel": ArgSpec("", bool, CVRT_WARNING),
        "grad_accum_dtype": ArgSpec("", str, CVRT_WARNING),
        "overlap_comm": ArgSpec("", bool, CVRT_WARNING),
        "gradient_checkpointing_use_reentrant": ArgSpec(
            "", bool, CVRT_WARNING
        ),  # default not use reentrant
        "disable_fast_tokenizer": ArgSpec("", bool, CVRT_WARNING),
        "deepspeed_enable_sleep": ArgSpec("", bool, CVRT_WARNING),
        "ds_tensor_parallel_size": ArgSpec("allocation_mode.engine.t", int, ""),
        # packing samples
        "packing_samples": ArgSpec("", bool, "default packing"),
        # LoRA
        "load_in_4bit": ArgSpec("", bool, CVRT_WARNING),
        "lora_rank": ArgSpec("", int, CVRT_WARNING),
        "lora_alpha": ArgSpec("", int, CVRT_WARNING),
        "target_modules": ArgSpec("", list, CVRT_WARNING),
        "lora_dropout": ArgSpec("", float, CVRT_WARNING),
        # PPO
        "save_path": ArgSpec("saver.fileroot", str, "", lambda x: os.path.dirname(x)),
        "num_episodes": ArgSpec("", int, ""),
        "rollout_batch_size": ArgSpec("rollout.consumer_batch_size", int, ""),
        "vllm_generate_batch_size": ArgSpec("", int, CVRT_WARNING),
        "micro_rollout_batch_size": ArgSpec("", int, CVRT_WARNING),
        "max_epochs": ArgSpec("total_train_epochs", int, ""),
        "prompt_max_len": ArgSpec("", int, FEAT_WARNING),  # train_dataset.max_seqlen
        "generate_max_len": ArgSpec("gconfig.max_new_tokens", int, ""),  # TODO: check
        "max_len": ArgSpec("", int, CVRT_WARNING),
        "max_samples": ArgSpec("", int, CVRT_WARNING),
        "max_norm": ArgSpec("actor.optimizer.gradient_clipping", float, ""),
        "l2": ArgSpec("actor.optimizer.weight_decay", float, ""),
        "ptx_coef": ArgSpec("", float, CVRT_WARNING),
        "eps_clip": ArgSpec("actor.eps_clip", float, ""),
        "eps_clip_low_high": ArgSpec("", list, CVRT_WARNING),  # check
        "value_clip": ArgSpec("", float, CVRT_WARNING),
        "lambd": ArgSpec("actor.gae_lambda", float, ""),
        "gamma": ArgSpec("actor.discount", float, ""),
        "micro_train_batch_size": ArgSpec(
            "actor.mb_spec.n_mbs",
            int,
            "default to 1, try actor.mb_spec.max_tokens_per_mb",
            lambda x: 1,
        ),
        "train_batch_size": ArgSpec("train_dataset.batch_size", int, ""),
        "normalize_reward": ArgSpec("", bool, "Default normalize reward"),
        "top_p": ArgSpec("gconfig.top_p", float, ""),
        "temperature": ArgSpec("gconfig.temperature", float, ""),
        "seed": ArgSpec("seed", int, ""),
        "freezing_actor_steps": ArgSpec("", int, CVRT_WARNING),
        "n_samples_per_prompt": ArgSpec("gconfig.n_samples", int, ""),
        "save_value_network": ArgSpec("", bool, CVRT_WARNING),
        "actor_learning_rate": ArgSpec("actor.optimizer.lr", float, ""),
        "critic_learning_rate": ArgSpec("", float, "no critic"),
        "lr_warmup_ratio": ArgSpec(
            "actor.optimizer.warmup_steps_proportion", float, ""
        ),
        "lr_scheduler": ArgSpec(
            "actor.optimizer.lr_scheduler_type", str, ""
        ),  # should be mapped to areal's scheduler
        "kl_target": ArgSpec("", float, CVRT_WARNING),
        "kl_horizon": ArgSpec("", int, CVRT_WARNING),
        "init_kl_coef": ArgSpec("actor.kl_ctl", float, ""),
        "kl_estimator": ArgSpec("", str, CVRT_WARNING),
        "aux_loss_coef": ArgSpec("", float, CVRT_WARNING),  # moe
        "entropy_loss_coef": ArgSpec("", float, CVRT_WARNING),  # entropy loss
        "adam_betas": ArgSpec("", list, CVRT_WARNING),  # should decouple
        "reward_clip_range": ArgSpec("actor.reward_clip", list, ""),
        "advantage_estimator": ArgSpec("", str, "only support grpo"),
        "use_kl_loss": ArgSpec("", bool, "actor.kl_ctl > 0 means use kl loss"),
        "no_advantage_std_norm": ArgSpec("", bool, CVRT_WARNING),
        # Context Parallel
        "ring_attn_size": ArgSpec("", int, CVRT_WARNING),
        "ring_head_stride": ArgSpec("", int, CVRT_WARNING),
        # Models
        "pretrain": ArgSpec("actor.path", str, ""),
        "reward_pretrain": ArgSpec("", str, CVRT_WARNING),  # no reward model
        "remote_rm_url": ArgSpec("", str, CVRT_WARNING),
        "critic_pretrain": ArgSpec("critic.path", str, CVRT_WARNING),  # no critic model
        "value_head_prefix": ArgSpec("", str, CVRT_WARNING),
        "ref_reward_offload": ArgSpec("", bool, CVRT_WARNING),
        "agent_func_path": ArgSpec("", str, CVRT_WARNING),
        # Custom dataset
        "prompt_data": ArgSpec("", str, CVRT_WARNING),
        "prompt_data_probs": ArgSpec("", str, CVRT_WARNING),
        "prompt_split": ArgSpec("", str, CVRT_WARNING),
        "eval_dataset": ArgSpec("", str, CVRT_WARNING),
        "eval_split": ArgSpec("", str, CVRT_WARNING),
        "eval_temperature": ArgSpec("", float, CVRT_WARNING),
        "eval_n_samples_per_prompt": ArgSpec("", int, CVRT_WARNING),
        "input_key": ArgSpec("", str, CVRT_WARNING),
        "label_key": ArgSpec("", str, CVRT_WARNING),
        "input_template": ArgSpec("", str, CVRT_WARNING),
        "apply_chat_template": ArgSpec("", bool, CVRT_WARNING),
        # wandb
        "use_wandb": ArgSpec(
            "stats_logger.wandb.mode", str, "", lambda x: "online"
        ),  # set WANDB_API_KEY, WANDB_BASE_URL
        "wandb_org": ArgSpec("stats_logger.wandb.entity", str, ""),
        "wandb_group": ArgSpec("stats_logger.wandb.group", str, ""),
        "wandb_project": ArgSpec("stats_logger.wandb.project", str, ""),
        "wandb_run_name": ArgSpec("stats_logger.wandb.name", str, ""),
        # Dynamic filtering
        "dynamic_filtering": ArgSpec("", bool, CVRT_WARNING),
        "dynamic_filtering_reward_range": ArgSpec("", list, CVRT_WARNING),
        # TensorBoard
        "use_tensorboard": ArgSpec(
            "stats_logger.tensorboard.path", str, ""
        ),  # set default path
        # performance ArgsSpectuning
        "perf": ArgSpec("", bool, CVRT_WARNING),
        # ModelScope
        "use_ms": ArgSpec("", bool, CVRT_WARNING),
    }

    def __init__(
        self, src_script_path: str, template_path: str, command_start: str = "python"
    ):
        """
        Args:
            src_script_path (str): The path to the source OpenRLHF script file.
            template_path (str): The path to the areal template file.
            command_start (str): The beginning of the command to look for,
                                 e.g., "python", "python3 my_script.py".
        """
        self.src_script_path = src_script_path
        self.template_path = template_path
        self.command_start = command_start

    def parse(self) -> dict:
        """
        Parse the arguments from OpenRLHF script to a dict.
        """
        args = self._parse_args_from_script(self.src_script_path, self.command_start)
        return args

    def convert(self) -> dict:
        """
        Convert the parsed arguments to
        """
        args = self.parse()
        args = self.convert_to_nested_args(args, self.ARG_MAP)
        args = post_process_args(args)
        template = self.get_lite_template(self.template_path)
        lite_args = always_merger.merge(template, args)
        return lite_args

    def _parse_args_from_script(
        self, script_path: str, command_start: str = "python"
    ) -> dict:
        """
        Parses arguments for a command in a shell script.

        It finds a command block starting with `command_start`, handles line
        continuations, and then extracts all key-value arguments.

        Args:
            script_path (str): The path to the .sh script file.
            command_start (str): The beginning of the command to look for,
                                e.g., "python", "python3 my_script.py".

        Returns:
            dict: A dictionary containing all the parsed arguments and their values.
        """
        full_command = ""
        in_command_block = False

        try:
            with open(script_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped_line = line.strip()

                    # Find the start of the command block
                    if not in_command_block and stripped_line.startswith(command_start):
                        in_command_block = True

                    if in_command_block:
                        if stripped_line.startswith("#") or not stripped_line:
                            continue

                        if stripped_line.endswith("\\"):
                            full_command += stripped_line[:-1].strip() + " "
                        else:
                            full_command += stripped_line
                            in_command_block = False
                            break
        except FileNotFoundError:
            print(f"Error: File not found at '{script_path}'")
            return {}

        if not full_command:
            print(f"Error: Could not find a command starting with '{command_start}'.")
            return {}

        # Tokenize the entire command string using shlex
        tokens = shlex.split(full_command)

        # Find the index of the first argument (starts with '-' or '--')
        args_start_index = -1
        for i, token in enumerate(tokens):
            if token.startswith("-"):
                args_start_index = i
                break

        if args_start_index == -1:
            # No arguments found
            return {}

        # The arguments are the rest of the list
        args_list = tokens[args_start_index:]

        # --- The rest of the parsing logic is the same ---
        params = {}
        i = 0
        while i < len(args_list):
            arg = args_list[i]
            if arg.startswith("--"):
                key = arg.lstrip("-")
                if (i + 1 < len(args_list)) and not args_list[i + 1].startswith("-"):
                    params[key] = args_list[i + 1]
                    i += 2
                else:
                    params[key] = True
                    i += 1
            else:
                # Also handle single-dash arguments if needed, or just skip
                i += 1

        return params


def post_process_args(args: dict):

    if "allocation_mode" in args:
        # convert allocation_mode to sglang.dX.tY.pZ
        dp = args["cluster"]["n_nodes"] * args["cluster"]["n_gpus_per_node"]
        allocation_mode = ""
        if "sglang" not in args["allocation_mode"]:
            allocation_mode = f"sglang.d{dp}t1p1"
        else:
            if "d" not in args["allocation_mode"]["sglang"]:
                allocation_mode = f"sglang.d{dp}"
            else:
                allocation_mode += f"sglang.d{args['allocation_mode']['sglang']['d']}"

            if "t" not in args["allocation_mode"]["sglang"]:
                allocation_mode += "t1"
            else:
                allocation_mode += f"t{args['allocation_mode']['sglang']['t']}"
            allocation_mode += f"p1"
        allocation_mode += "+"
        if "engine" not in args["allocation_mode"]:
            allocation_mode += f"d{dp}t1p1"
        else:
            allocation_mode += f"d{args['allocation_mode']['engine']['d']}"
            if "t" not in args["allocation_mode"]["engine"]:
                allocation_mode += "t1"
            else:
                allocation_mode += f".t{args['allocation_mode']['engine']['t']}"
            allocation_mode += f"p1"

        args["allocation_mode"] = allocation_mode
        args["cluster"]["n_nodes"] = args["cluster"]["n_nodes"] * 2
    return args


class AReaLConverter(Converter):
    """
    Convert realhf arguments to AReaL-Lite arguments.
    """

    ARG_MAP = {
        # Top-level experiment info
        "experiment_name": ArgSpec("experiment_name", str, ""),
        "trial_name": ArgSpec("trial_name", str, ""),
        "mode": ArgSpec("", str, CVRT_WARNING),
        "debug": ArgSpec("", bool, CVRT_WARNING),
        "metric_discovery_port": ArgSpec("", int, CVRT_WARNING),
        "partition": ArgSpec("", str, CVRT_WARNING),
        "schedule_strategy": ArgSpec("", str, CVRT_WARNING),
        "recover_mode": ArgSpec("", bool, CVRT_WARNING),
        "recover_retries": ArgSpec("", int, CVRT_WARNING),
        "recover_after": ArgSpec("", int, CVRT_WARNING),
        "ignore_worker_error": ArgSpec("", bool, CVRT_WARNING),
        "allocation_mode": ArgSpec(
            "allocation_mode", str, "", lambda x: x.replace("m", "t")
        ),
        "n_nodes": ArgSpec("cluster.n_nodes", int, ""),  # TODO: check cluster.n_nodes
        "n_gpus_per_node": ArgSpec(
            "cluster.n_gpus_per_node", int, ""
        ),  # TODO: check cluster
        "node_list": ArgSpec("", str, CVRT_WARNING),
        "exclude": ArgSpec("", str, CVRT_WARNING),
        "seed": ArgSpec("seed", int, ""),
        # Cluster configuration
        "cluster.config_path": ArgSpec("", str, CVRT_WARNING),
        "cluster.name_resolve.type": ArgSpec("cluster.name_resolve.type", str, ""),
        "cluster.name_resolve.nfs_record_root": ArgSpec(
            "cluster.name_resolve.nfs_record_root", str, ""
        ),
        "cluster.name_resolve.etcd3_addr": ArgSpec(
            "cluster.name_resolve.etcd3_addr", str, ""
        ),
        "cluster.name_resolve.ray_actor_name": ArgSpec(
            "cluster.name_resolve.ray_actor_name", str, ""
        ),
        "cluster.cluster_name": ArgSpec("cluster.cluster_name", str, ""),
        "cluster.fileroot": ArgSpec("cluster.fileroot", str, ""),
        "cluster.gpu_type": ArgSpec("", str, CVRT_WARNING),
        "cluster.mount": ArgSpec("launcher.slurm.mount", str, ""),
        "cluster.gpu_image": ArgSpec("", str, "Do not convert image"),
        "cluster.cpu_image": ArgSpec("", str, "Do not convert image"),
        "cluster.gpu_infer_image": ArgSpec("", str, "Do not convert image"),
        "cluster.node_name_prefix": ArgSpec("", str, CVRT_WARNING),
        "cluster.n_nodes": ArgSpec("", int, CVRT_WARNING),
        "cluster.n_gpus_per_node": ArgSpec("cluster.n_gpus_per_node", int, ""),
        # CPU resource allocation
        "cpus_per_master_worker": ArgSpec("", int, CVRT_WARNING),
        "mem_per_master_worker": ArgSpec("", int, CVRT_WARNING),
        "cpus_per_model_worker": ArgSpec("launcher.trainer_cpus_per_gpu", int, ""),
        "mem_per_model_worker": ArgSpec("launcher.trainer_mem_per_gpu", int, ""),
        # Actor model
        "actor.type._class": ArgSpec("", str, CVRT_WARNING),
        "actor.type.is_critic": ArgSpec("", bool, CVRT_WARNING),
        "actor.path": ArgSpec("actor.path", str, ""),
        "actor.init_from_scratch": ArgSpec("actor.init_from_scratch", bool, ""),
        "actor.init_critic_from_actor": ArgSpec(
            "actor.init_critic_from_actor", bool, ""
        ),
        "actor.backend": ArgSpec(
            "actor.backend", str, "Only support fsdp now", map_fn=lambda x: "fsdp"
        ),
        "actor.gradient_checkpointing": ArgSpec(
            "actor.gradient_checkpointing", bool, ""
        ),
        "actor.bf16": ArgSpec(
            "actor.dtype", bool, "", lambda x: "bfloat16" if x else "float16"
        ),
        # "actor.dtype": ArgSpec("actor.dtype", str, ""),
        "actor.optimizer.type": ArgSpec("actor.optimizer.type", str, ""),
        "actor.optimizer.lr": ArgSpec("actor.optimizer.lr", float, ""),
        "actor.optimizer.weight_decay": ArgSpec(
            "actor.optimizer.weight_decay", float, ""
        ),
        "actor.optimizer.beta1": ArgSpec("actor.optimizer.beta1", float, ""),
        "actor.optimizer.beta2": ArgSpec("actor.optimizer.beta2", float, ""),
        "actor.optimizer.eps": ArgSpec("actor.optimizer.eps", float, ""),
        "actor.optimizer.min_lr_ratio": ArgSpec(
            "actor.optimizer.min_lr_ratio", float, ""
        ),
        "actor.optimizer.lr_scheduler_type": ArgSpec(
            "actor.optimizer.lr_scheduler_type", str, ""
        ),
        "actor.optimizer.warmup_steps_proportion": ArgSpec(
            "actor.optimizer.warmup_steps_proportion", float, ""
        ),
        "actor.optimizer.offload": ArgSpec("actor.optimizer.offload", bool, ""),
        "actor.optimizer.initial_loss_scale": ArgSpec(
            "actor.optimizer.initial_loss_scale", float, ""
        ),
        "actor.optimizer.min_loss_scale": ArgSpec(
            "actor.optimizer.min_loss_scale", float, ""
        ),
        "actor.optimizer.loss_scale_window": ArgSpec(
            "actor.optimizer.loss_scale_window", int, ""
        ),
        "actor.optimizer.hysteresis": ArgSpec("actor.optimizer.hysteresis", int, ""),
        "actor.optimizer.gradient_clipping": ArgSpec(
            "actor.optimizer.gradient_clipping", float, ""
        ),
        # Megatron not implemented in areal
        "actor.megatron.ddp.grad_reduce_in_fp32": ArgSpec("", bool, CVRT_WARNING),
        "actor.megatron.ddp.overlap_grad_reduce": ArgSpec("", bool, CVRT_WARNING),
        "actor.megatron.ddp.overlap_param_gather": ArgSpec("", bool, CVRT_WARNING),
        "actor.megatron.ddp.align_param_gather": ArgSpec("", bool, CVRT_WARNING),
        "actor.megatron.ddp.use_distributed_optimizer": ArgSpec("", bool, CVRT_WARNING),
        "actor.megatron.ddp.check_for_nan_in_grad": ArgSpec("", bool, CVRT_WARNING),
        "actor.megatron.ddp.bucket_size": ArgSpec("", int, CVRT_WARNING),
        "actor.megatron.ddp.average_in_collective": ArgSpec("", bool, CVRT_WARNING),
        "actor.megatron.ddp.fp8_param_gather": ArgSpec("", bool, CVRT_WARNING),
        "actor.megatron.overlap_param_gather_with_optimizer_step": ArgSpec(
            "", bool, CVRT_WARNING
        ),
        "actor.megatron.use_precision_aware_optimizer": ArgSpec("", bool, CVRT_WARNING),
        "actor.megatron.main_grads_dtype": ArgSpec("", str, CVRT_WARNING),
        "actor.megatron.main_params_dtype": ArgSpec("", str, CVRT_WARNING),
        "actor.megatron.exp_avg_dtype": ArgSpec("", str, CVRT_WARNING),
        "actor.megatron.exp_avg_sq_dtype": ArgSpec("", str, CVRT_WARNING),
        # SGLang
        "actor.sglang.disable_cuda_graph": ArgSpec(
            "sglang.disable_cuda_graph", bool, ""
        ),
        "actor.sglang.disable_radix_cache": ArgSpec(
            "sglang.disable_radix_cache", bool, ""
        ),
        "actor.sglang.disable_cuda_graph_padding": ArgSpec(
            "sglang.disable_cuda_graph_padding", bool, ""
        ),
        "actor.sglang.enable_nccl_nvls": ArgSpec("sglang.enable_nccl_nvls", bool, ""),
        "actor.sglang.disable_outlines_disk_cache": ArgSpec(
            "sglang.disable_outlines_disk_cache", bool, ""
        ),
        "actor.sglang.disable_custom_all_reduce": ArgSpec(
            "sglang.disable_custom_all_reduce", bool, ""
        ),
        "actor.sglang.disable_overlap_schedule": ArgSpec(
            "sglang.disable_overlap_schedule", bool, ""
        ),
        "actor.sglang.enable_mixed_chunk": ArgSpec(
            "sglang.enable_mixed_chunk", bool, ""
        ),
        "actor.sglang.enable_dp_attention": ArgSpec(
            "sglang.enable_dp_attention", bool, ""
        ),
        "actor.sglang.enable_ep_moe": ArgSpec("sglang.enable_ep_moe", bool, ""),
        "actor.sglang.enable_torch_compile": ArgSpec(
            "sglang.enable_torch_compile", bool, ""
        ),
        "actor.sglang.torch_compile_max_bs": ArgSpec(
            "sglang.torch_compile_max_bs", int, ""
        ),
        "actor.sglang.cuda_graph_max_bs": ArgSpec("sglang.cuda_graph_max_bs", int, ""),
        "actor.sglang.cuda_graph_bs": ArgSpec("sglang.cuda_graph_bs", list, ""),
        "actor.sglang.torchao_config": ArgSpec("sglang.torchao_config", str, ""),
        "actor.sglang.enable_nan_detection": ArgSpec(
            "sglang.enable_nan_detection", bool, ""
        ),
        "actor.sglang.enable_p2p_check": ArgSpec("sglang.enable_p2p_check", bool, ""),
        "actor.sglang.triton_attention_reduce_in_fp32": ArgSpec(
            "sglang.triton_attention_reduce_in_fp32", bool, ""
        ),
        "actor.sglang.triton_attention_num_kv_splits": ArgSpec(
            "sglang.triton_attention_num_kv_splits", int, ""
        ),
        "actor.sglang.num_continuous_decode_steps": ArgSpec(
            "sglang.num_continuous_decode_steps", int, ""
        ),
        "actor.sglang.enable_memory_saver": ArgSpec(
            "sglang.enable_memory_saver", bool, ""
        ),
        "actor.sglang.allow_auto_truncate": ArgSpec(
            "sglang.allow_auto_truncate", bool, ""
        ),
        "actor.sglang.attention_backend": ArgSpec("sglang.attention_backend", str, ""),
        "actor.sglang.sampling_backend": ArgSpec("sglang.sampling_backend", str, ""),
        "actor.sglang.context_length": ArgSpec("sglang.context_length", int, ""),
        "actor.sglang.mem_fraction_static": ArgSpec(
            "sglang.mem_fraction_static", float, ""
        ),
        "actor.sglang.max_running_requests": ArgSpec(
            "sglang.max_running_requests", int, ""
        ),
        "actor.sglang.chunked_prefill_size": ArgSpec(
            "sglang.chunked_prefill_size", int, ""
        ),
        "actor.sglang.max_prefill_tokens": ArgSpec(
            "sglang.max_prefill_tokens", int, ""
        ),
        "actor.sglang.schedule_policy": ArgSpec("sglang.schedule_policy", str, ""),
        "actor.sglang.schedule_conservativeness": ArgSpec(
            "sglang.schedule_conservativeness", float, ""
        ),
        "actor.sglang.cpu_offload_gb": ArgSpec("sglang.cpu_offload_gb", int, ""),
        "actor.sglang.hybrid_train": ArgSpec("", bool, CVRT_WARNING),
        "actor.sglang.dtype": ArgSpec("sglang.dtype", str, ""),
        "actor.sglang.kv_cache_dtype": ArgSpec("sglang.kv_cache_dtype", str, ""),
        "actor.sglang.log_level": ArgSpec("sglang.log_level", str, ""),
        "actor.sglang.log_level_http": ArgSpec("sglang.log_level_http", str, ""),
        "actor.sglang.log_requests": ArgSpec("sglang.log_requests", bool, ""),
        "actor.sglang.log_requests_level": ArgSpec(
            "sglang.log_requests_level", int, ""
        ),
        "actor.sglang.show_time_cost": ArgSpec("sglang.show_time_cost", bool, ""),
        "actor.sglang.enable_metrics": ArgSpec("sglang.enable_metrics", bool, ""),
        "actor.sglang.decode_log_interval": ArgSpec(
            "sglang.decode_log_interval", int, ""
        ),
        "group_size": ArgSpec("gconfig.n_samples", int, ""),
        "generation_size": ArgSpec("", int, CVRT_WARNING),
        "group_adv_norm": ArgSpec("actor.group_adv_norm", bool, ""),
        "mask_no_eos_with_zero": ArgSpec("actor.mask_no_eos_with_zero", bool, ""),
        "mask_too_long": ArgSpec("", bool, CVRT_WARNING),
        "check_verifier_status": ArgSpec("", bool, CVRT_WARNING),
        "ref_ema_eta": ArgSpec("", float, CVRT_WARNING),
        "rw_type": ArgSpec("", str, CVRT_WARNING),
        "check_xml_format": ArgSpec("", bool, CVRT_WARNING),
        "dataset_filter_threshold": ArgSpec("", float, CVRT_WARNING),
        "dataset_max_filter_percentage": ArgSpec("", float, CVRT_WARNING),
        "success_rate_ub": ArgSpec("", float, CVRT_WARNING),
        "success_rate_lb": ArgSpec("", float, CVRT_WARNING),
        "no_training": ArgSpec("", bool, CVRT_WARNING),
        # Ref model
        "ref.type._class": ArgSpec("", str, CVRT_WARNING),
        "ref.type.is_critic": ArgSpec("", bool, CVRT_WARNING),
        "ref.path": ArgSpec("ref.path", str, ""),
        "ref.init_from_scratch": ArgSpec("ref.init_from_scratch", bool, ""),
        "ref.backend": ArgSpec("ref.backend", str, ""),
        "ref.bf16": ArgSpec(
            "ref.dtype", bool, "", lambda x: "bfloat16" if x else "float16"
        ),
        # TODO: remaining ref args
        # MFC
        "actor_train.mb_spec.n_mbs": ArgSpec("actor.mb_spec.n_mbs", int, ""),
        "actor_train.mb_spec.max_tokens_per_mb": ArgSpec(
            "actor.mb_spec.max_tokens_per_mb", int, ""
        ),
        "actor_train.parallel": ArgSpec("", int, CVRT_WARNING),
        "actor_train.device_mesh": ArgSpec("", str, CVRT_WARNING),
        "actor_gen.mb_spec.n_mbs": ArgSpec("", int, CVRT_WARNING),
        "actor_gen.mb_spec.max_tokens_per_mb": ArgSpec("", int, CVRT_WARNING),
        "actor_gen.parallel": ArgSpec("", int, CVRT_WARNING),
        "actor_gen.device_mesh": ArgSpec("", str, CVRT_WARNING),
        "actor_inf.mb_spec.n_mbs": ArgSpec("", int, CVRT_WARNING),
        "actor_inf.mb_spec.max_tokens_per_mb": ArgSpec("", int, CVRT_WARNING),
        "actor_inf.parallel": ArgSpec("", int, CVRT_WARNING),
        "actor_inf.device_mesh": ArgSpec("", str, CVRT_WARNING),
        "ref_inf.mb_spec.n_mbs": ArgSpec("ref.mb_spec.n_mbs", int, ""),
        "ref_inf.mb_spec.max_tokens_per_mb": ArgSpec(
            "ref.mb_spec.max_tokens_per_mb", int, ""
        ),
        "ref_inf.parallel": ArgSpec("", int, CVRT_WARNING),
        "ref_inf.device_mesh": ArgSpec("", str, CVRT_WARNING),
        # Dataset
        "dataset.path": ArgSpec("", str, CVRT_WARNING),
        "dataset.max_prompt_len": ArgSpec("", int, "Should be implemented"),
        "dataset.train_bs_n_seqs": ArgSpec("train_dataset.batch_size", int, ""),
        "dataset.fill_to_max_length": ArgSpec("", bool, CVRT_WARNING),
        "shuffle_dataset": ArgSpec("train_dataset.shuffle", bool, ""),
        # PPO/gen
        "ppo.gen.n": ArgSpec("", int, "Use group_size to control"),
        "ppo.gen.max_new_tokens": ArgSpec("gconfig.max_new_tokens", int, ""),
        "ppo.gen.min_new_tokens": ArgSpec("gconfig.min_new_tokens", int, ""),
        "ppo.gen.greedy": ArgSpec("gconfig.greedy", bool, ""),
        "ppo.gen.top_p": ArgSpec("gconfig.top_p", float, ""),
        "ppo.gen.top_k": ArgSpec("gconfig.top_k", int, ""),
        "ppo.gen.temperature": ArgSpec("gconfig.temperature", float, ""),
        # PPO core
        "ppo.ppo_n_minibatches": ArgSpec("actor.ppo_n_minibatches", int, ""),
        "ppo.eps_clip": ArgSpec("actor.eps_clip", float, ""),
        "ppo.c_clip": ArgSpec("actor.c_clip", float, ""),
        "ppo.value_eps_clip": ArgSpec("", float, CVRT_WARNING),
        "ppo.early_stop_imp_ratio": ArgSpec("", float, CVRT_WARNING),
        "ppo.actor_sample_reuse": ArgSpec("", bool, CVRT_WARNING),
        "ppo.critic_sample_reuse": ArgSpec("", bool, CVRT_WARNING),
        "ppo.max_reward_clip": ArgSpec("actor.reward_clip", float, ""),
        "ppo.reward_output_scaling": ArgSpec("actor.reward_scaling", float, ""),
        "ppo.reward_output_bias": ArgSpec("actor.reward_bias", float, ""),
        "ppo.fuse_rew_ref": ArgSpec("", bool, CVRT_WARNING),
        "ppo.discount": ArgSpec("actor.discount", float, ""),
        "ppo.gae_lambda": ArgSpec("actor.gae_lambda", float, ""),
        "ppo.adv_norm": ArgSpec("actor.adv_norm", bool, ""),
        "ppo.kl_ctl": ArgSpec("actor.kl_ctl", float, ""),
        "ppo.use_adaptive_kl_ctl": ArgSpec("", bool, CVRT_WARNING),
        "ppo.disable_value": ArgSpec("", bool, CVRT_WARNING),
        "ppo.value_norm": ArgSpec("", bool, CVRT_WARNING),
        "ppo.value_norm_type": ArgSpec("", str, CVRT_WARNING),
        "ppo.value_norm_beta": ArgSpec("", float, CVRT_WARNING),
        "ppo.value_norm_eps": ArgSpec("", float, CVRT_WARNING),
        "ppo.recompute_logprob": ArgSpec("actor.recompute_logprob", bool, ""),
        "ppo.use_decoupled_loss": ArgSpec("actor.use_decoupled_loss", bool, ""),
        "ppo.behav_imp_weight_cap": ArgSpec("actor.behav_imp_weight_cap", float, ""),
        # Async
        "schedule_policy": ArgSpec("rollout.schedule_policy", str, ""),
        "new_tokens_per_chunk": ArgSpec("", int, CVRT_WARNING),
        "max_head_offpolicyness": ArgSpec("rollout.max_head_offpolicyness", int, ""),
        "n_rollout_workers": ArgSpec("", int, CVRT_WARNING),
        "max_concurrent_rollouts": ArgSpec("rollout.max_concurrent_rollouts", int, ""),
        "flush_request_timeout": ArgSpec("rollout.request_timeout", float, ""),
        "cpus_per_generation_server": ArgSpec(
            "launcher.inference_server_cpus_per_gpu", int, ""
        ),
        "mem_per_generation_server": ArgSpec(
            "launcher.inference_server_mem_per_gpu", int, ""
        ),
        "cpus_per_gserver_manager": ArgSpec("", int, CVRT_WARNING),
        "mem_per_gserver_manager": ArgSpec("", int, CVRT_WARNING),
        "cpus_per_rollout_worker": ArgSpec("", int, CVRT_WARNING),
        "mem_per_rollout_worker": ArgSpec("", int, CVRT_WARNING),
        # Saver/checkpointer/evaluator
        "saver.freq_epochs": ArgSpec("saver.freq_epochs", int, ""),
        "saver.freq_steps": ArgSpec("saver.freq_steps", int, ""),
        "saver.freq_secs": ArgSpec("saver.freq_secs", int, ""),
        "checkpointer.freq_epochs": ArgSpec("checkpointer.freq_epochs", int, ""),
        "checkpointer.freq_steps": ArgSpec("checkpointer.freq_steps", int, ""),
        "checkpointer.freq_secs": ArgSpec("checkpointer.freq_secs", int, ""),
        "evaluator.freq_epochs": ArgSpec("evaluator.freq_epochs", int, ""),
        "evaluator.freq_steps": ArgSpec("evaluator.freq_steps", int, ""),
        "evaluator.freq_secs": ArgSpec("evaluator.freq_secs", int, ""),
        # Logging
        "wandb.mode": ArgSpec("stats_logger.wandb.mode", str, ""),
        "wandb.entity": ArgSpec("stats_logger.wandb.entity", str, ""),
        "wandb.project": ArgSpec("stats_logger.wandb.project", str, ""),
        "wandb.name": ArgSpec("stats_logger.wandb.name", str, ""),
        "wandb.job_type": ArgSpec("stats_logger.wandb.job_type", str, ""),
        "wandb.group": ArgSpec("stats_logger.wandb.group", str, ""),
        "wandb.notes": ArgSpec("stats_logger.wandb.notes", str, ""),
        "wandb.tags": ArgSpec("stats_logger.wandb.tags", list, ""),
        "wandb.config": ArgSpec("stats_logger.wandb.config", dict, ""),
        "swanlab.project": ArgSpec("stats_logger.swanlab.project", str, ""),
        "swanlab.name": ArgSpec("stats_logger.swanlab.name", str, ""),
        "swanlab.config": ArgSpec("stats_logger.swanlab.config", dict, ""),
        "swanlab.log_dir": ArgSpec("stats_logger.swanlab.log_dir", str, ""),
        "swanlab.mode": ArgSpec("stats_logger.swanlab.mode", str, ""),
        "swanlab.api_key": ArgSpec("stats_logger.swanlab.api_key", str, ""),
        "tensorboard.path": ArgSpec("stats_logger.tensorboard.path", str, ""),
        # Exp Control
        "exp_ctrl.total_train_epochs": ArgSpec("total_train_epochs", int, ""),
        "exp_ctrl.save_freq_epochs": ArgSpec("saver.freq_epochs", int, ""),
        "exp_ctrl.save_freq_steps": ArgSpec("saver.freq_steps", int, ""),
        "exp_ctrl.save_freq_secs": ArgSpec("saver.freq_secs", int, ""),
        "exp_ctrl.ckpt_freq_epochs": ArgSpec("checkpointer.freq_epochs", int, ""),
        "exp_ctrl.ckpt_freq_steps": ArgSpec("checkpointer.freq_steps", int, ""),
        "exp_ctrl.ckpt_freq_secs": ArgSpec("checkpointer.freq_secs", int, ""),
        "exp_ctrl.eval_freq_epochs": ArgSpec("evaluator.freq_epochs", int, ""),
        "exp_ctrl.eval_freq_steps": ArgSpec("evaluator.freq_steps", int, ""),
        "exp_ctrl.eval_freq_secs": ArgSpec("evaluator.freq_secs", int, ""),
        "exp_ctrl.benchmark_steps": ArgSpec("", int, CVRT_WARNING),
        "exp_ctrl.benchmark_n_seqs": ArgSpec("", int, CVRT_WARNING),
        # Auto Evaluation
        "auto_eval": ArgSpec("", bool, CVRT_WARNING),
        "auto_eval_config.data_names": ArgSpec("", list, CVRT_WARNING),
        "auto_eval_config.max_gen_tokens": ArgSpec("", int, CVRT_WARNING),
        "auto_eval_config.max_concurrent_jobs": ArgSpec("", int, CVRT_WARNING),
        "auto_eval_config.eval_job_image": ArgSpec("", str, CVRT_WARNING),
        "auto_eval_config.initial_checkpoint_path": ArgSpec("", str, CVRT_WARNING),
        "auto_eval_config.prompt_type": ArgSpec("", str, CVRT_WARNING),
        # Miscellaneous
        "cache_clear_freq": ArgSpec("", int, CVRT_WARNING),
        "torch_cache_mysophobia": ArgSpec("", bool, CVRT_WARNING),
        "ray_temp_path": ArgSpec("", str, CVRT_WARNING),
    }

    def __init__(self, src_config_path: str, template_path: str):
        self.src_config_path = src_config_path
        self.template_path = template_path

    def parse(self) -> dict:
        with open(self.src_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg

    def convert(self) -> dict:
        args = self.parse()
        args = self.flatten_dict(args)
        args = self.convert_to_nested_args(args, self.ARG_MAP)  # realite args format
        template = self.get_lite_template(self.template_path)
        lite_args = always_merger.merge(template, args)
        return lite_args


def parse_args():
    parser = argparse.ArgumentParser(description="Convert to areal YAML config")
    parser.add_argument(
        "--convert_src",
        type=str,
        choices=["OpenRLHF", "AReaL"],
        default="OpenRLHF",
        help="source config type to convert from",
    )
    parser.add_argument(
        "--src_script_path", type=str, help="path to the source OpenRLHF script file"
    )
    parser.add_argument(
        "--src_config_path", type=str, help="path to the src AReaL config file"
    )
    parser.add_argument(
        "--template_path",
        type=str,
        required=True,
        help="path to the areal template file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.yaml",
        help="path to save the converted YAML config",
    )
    return parser.parse_args()


def main():
    """
    Usage:
    - OpenRLHF: python areal/utils/convert.py --convert_src OpenRLHF --src_script_path <path_to_script> --template_path <path_to_template_alite_yaml> --output_path <output_yaml>
    - AReaL:    python areal/utils/convert.py --convert_srv AReaL --src_config_path <path_to_areal_yaml> --template_path <path_to_template_alite_yaml> --output_path <output_yaml>
    """
    args = parse_args()
    converters = {"OpenRLHF": OpenRLHFConverter, "AReaL": AReaLConverter}
    converter_args = {
        "OpenRLHF": {
            "src_script_path": args.src_script_path,
            "template_path": args.template_path,
        },
        "AReaL": {
            "src_config_path": args.src_config_path,
            "template_path": args.template_path,
        },
    }
    converter: Converter = converters[args.convert_src](
        **converter_args[args.convert_src]
    )
    lite_args = converter.convert()
    yaml_str = yaml.dump(lite_args, sort_keys=False, allow_unicode=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        yaml.dump(lite_args, f, sort_keys=False, allow_unicode=True)
    print(f"Converted areal config saved to {args.output_path}")
    # print(yaml_str)


if __name__ == "__main__":
    main()
