# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

# Parts of code are modified from Megatron-LM.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import collections
import dataclasses
import math
import pathlib
from contextlib import contextmanager
from typing import *

import torch
import torch.distributed as dist
import transformers

try:
    from megatron.core import parallel_state
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )
    from megatron.core.distributed.param_and_grad_buffer import ParamAndGradBuffer
    from megatron.core.optimizer import DistributedOptimizer, get_megatron_optimizer
    from megatron.core.optimizer.clip_grads import clip_grad_norm_fp32, count_zeros_fp32
    from megatron.core.optimizer.optimizer_config import (
        OptimizerConfig as MegatronOptimizerConfig,
    )
    from megatron.core.transformer.transformer_config import (
        TransformerConfig as MegatronTransformerConfig,
    )
except (ModuleNotFoundError, ImportError):
    # importing megatron.core in CPU container will fail due to the requirement of apex
    # Here class types must be defined for type hinting
    class MegatronTransformerConfig:
        pass

    class DistributedDataParallel:
        pass

    class DistributedOptimizer:
        pass


from realhf.api.core import model_api
from realhf.api.core.data_api import MicroBatchSpec, SequenceSample
from realhf.api.quickstart.model import MegatronConfig, OptimizerConfig
from realhf.base import constants, logging
from realhf.base.datapack import flat2d
from realhf.base.monitor import CUDATimeMarkType, cuda_tmarked
from realhf.impl.model.backend.inference import PipelinableInferenceEngine
from realhf.impl.model.backend.pipe_runner import PipelineRunner, PipeTrainInstrSet
from realhf.impl.model.modules.mlp import get_activation_fn
from realhf.impl.model.nn.flatten_param import ContiguousParamSpec
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_base import ReaLModelBlock
from realhf.impl.model.parallelism.pipeline_parallel.tensor_storage import TensorBuffer

WITHIN_MEGATRON_CONTEXT = False

logger = logging.getLogger("Megatron Backend", "benchmark")


@contextmanager
def megatron_ctx():
    global WITHIN_MEGATRON_CONTEXT
    if WITHIN_MEGATRON_CONTEXT:
        raise RuntimeError("Megatron context is already set up. Destroy it first.")

    WITHIN_MEGATRON_CONTEXT = True

    grid = constants.grid()
    # TODO: implement context parallel.
    # TODO: implement expert parallel.

    # Build the data-parallel groups.
    g = constants.data_parallel_group()
    parallel_state._DATA_PARALLEL_GROUP = g
    parallel_state._DATA_PARALLEL_GROUP_GLOO = grid.get_data_parallel_group_gloo()
    parallel_state._DATA_PARALLEL_GLOBAL_RANKS = dist.get_process_group_ranks(g)
    parallel_state._DATA_PARALLEL_GROUP_WITH_CP = g
    parallel_state._DATA_PARALLEL_GROUP_WITH_CP_GLOO = (
        grid.get_data_parallel_group_gloo()
    )
    parallel_state._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = dist.get_process_group_ranks(g)

    # Build the context-parallel groups.
    parallel_state._CONTEXT_PARALLEL_GROUP = constants.self_group()
    parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS = [dist.get_rank()]

    # Build the model-parallel groups.
    parallel_state._MODEL_PARALLEL_GROUP = grid.get_model_parallel_group()

    # Build the tensor model-parallel groups.
    g = constants.model_parallel_group()
    parallel_state._TENSOR_MODEL_PARALLEL_GROUP = g

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    g = constants.pipe_parallel_group()
    parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = g
    parallel_state._PIPELINE_GLOBAL_RANKS = dist.get_process_group_ranks(g)
    parallel_state._EMBEDDING_GROUP = grid.embedding_proc_group
    parallel_state._EMBEDDING_GLOBAL_RANKS = (
        dist.get_process_group_ranks(grid.embedding_proc_group)
        if grid.embedding_proc_group is not None
        else list(range(dist.get_world_size()))
    )
    parallel_state._POSITION_EMBEDDING_GROUP = grid.position_embedding_proc_group
    parallel_state._POSITION_EMBEDDING_GLOBAL_RANKS = (
        dist.get_process_group_ranks(grid.position_embedding_proc_group)
        if grid.position_embedding_proc_group is not None
        else list(range(dist.get_world_size()))
    )

    # Build the tensor + data parallel groups.
    parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP = grid.tp_dp_proc_group
    parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = grid.tp_dp_proc_group

    # Build the tensor + expert parallel groups
    parallel_state._EXPERT_MODEL_PARALLEL_GROUP = constants.self_group()
    parallel_state._TENSOR_AND_EXPERT_PARALLEL_GROUP = constants.model_parallel_group()
    g = constants.data_parallel_group()
    parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP = g
    parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = (
        grid.get_data_parallel_group_gloo()
    )

    # Remove the global memory buffer for megatron to save GPU memory.
    parallel_state._GLOBAL_MEMORY_BUFFER = None
    yield
    WITHIN_MEGATRON_CONTEXT = False


def get_megatron_transformer_config(
    mconfig: model_api.ReaLModelConfig,
) -> MegatronTransformerConfig:
    nq = mconfig.hidden_dim // mconfig.head_dim
    n_group = nq // mconfig.n_kv_heads
    return MegatronTransformerConfig(
        num_layers=mconfig.n_layers,
        hidden_size=mconfig.hidden_dim,
        num_attention_heads=nq,
        num_query_groups=n_group,
        ffn_hidden_size=mconfig.intermediate_dim,
        kv_channels=mconfig.n_kv_heads,
        hidden_dropout=0.0,
        attention_dropout=mconfig.attn_pdrop,
        layernorm_epsilon=mconfig.layer_norm_epsilon,
        add_qkv_bias=mconfig.use_attention_bias,
        activation_func=get_activation_fn(mconfig.activation_function),
        rotary_interleaved=mconfig.rotary_interleaved,
        normalization=("RMSNorm" if mconfig.layer_norm_type == "rms" else "LayerNorm"),
        attention_softmax_in_fp32=True,
        apply_query_key_layer_scaling=mconfig.scale_attn_by_inverse_layer_idx,
    )


@dataclasses.dataclass
class MegatronEngine:
    ddp: DistributedDataParallel
    optim: DistributedOptimizer
    lr_scheduler: Any

    def zero_grad(self, set_to_none=True):
        self.ddp.zero_grad_buffer()
        self.optim.zero_grad(set_to_none=set_to_none)


# Adopted from Megatron-LM/megatron/training/optimizer_param_scheduler.py
class OptimizerParamScheduler(object):
    """Anneals learning rate and weight decay.

    Adopted from Megatron-LM. This class is not included in
    megatron.core, so we have to copy-paste it here.
    """

    def __init__(
        self,
        optimizer,
        init_lr,
        max_lr,
        min_lr,
        lr_warmup_steps,
        lr_decay_steps,
        lr_decay_style,
        start_wd,
        end_wd,
        wd_incr_steps,
        wd_incr_style,
        use_checkpoint_opt_param_scheduler=True,
        override_opt_param_scheduler=False,
    ):

        # Class values.
        self.optimizer = optimizer

        self.init_lr = init_lr
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        assert self.init_lr <= self.max_lr

        self.lr_warmup_steps = lr_warmup_steps
        self.num_steps = 0
        self.lr_decay_steps = lr_decay_steps
        assert self.lr_decay_steps > 0
        assert self.lr_warmup_steps < self.lr_decay_steps

        self.lr_decay_style = lr_decay_style

        self.start_wd = start_wd
        self.end_wd = end_wd
        assert self.start_wd >= 0.0
        assert self.end_wd >= self.start_wd
        self.wd_incr_steps = wd_incr_steps
        self.wd_incr_style = wd_incr_style

        self.override_opt_param_scheduler = override_opt_param_scheduler
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        if self.override_opt_param_scheduler:
            assert not self.use_checkpoint_opt_param_scheduler, (
                "both override and " "use-checkpoint are set."
            )

        # Set the learning rate
        self.step(0)
        self.log_rank_0("> learning rate decay style: {}".format(self.lr_decay_style))

    def log_rank_0(self, msg):
        if constants.parallelism_rank() == 0:
            logger.info(msg)

    def get_wd(self):
        """Weight decay incr functions."""
        if self.num_steps > self.wd_incr_steps:
            return self.end_wd

        if self.wd_incr_style == "constant":
            assert self.start_wd == self.end_wd
            return self.end_wd

        incr_ratio = float(self.num_steps) / float(self.wd_incr_steps)
        assert incr_ratio >= 0.0
        assert incr_ratio <= 1.0
        delta_wd = self.end_wd - self.start_wd

        if self.wd_incr_style == "linear":
            coeff = incr_ratio
        elif self.wd_incr_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * (1 - incr_ratio)) + 1.0)
        else:
            raise Exception(
                "{} weight decay increment style is not supported.".format(
                    self.wd_incr_style
                )
            )

        return self.start_wd + coeff * delta_wd

    def get_lr(self, param_group):
        """Learning rate decay functions from:
        https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        max_lr = param_group.get("max_lr", self.max_lr)
        min_lr = param_group.get("min_lr", self.min_lr)

        # Use linear warmup for the initial part.
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            return self.init_lr + (
                (max_lr - self.init_lr)
                * float(self.num_steps)
                / float(self.lr_warmup_steps)
            )

        # If the learning rate is constant, just return the initial value.
        if self.lr_decay_style == "constant":
            return max_lr

        # For any steps larger than `self.lr_decay_steps`, use `min_lr`.
        if self.num_steps > self.lr_decay_steps:
            return min_lr

        # If we are done with the warmup period, use the decay style.
        if self.lr_decay_style == "inverse-square-root":
            warmup_steps = max(self.lr_warmup_steps, 1)
            num_steps = max(self.num_steps, 1)
            lr = max_lr * warmup_steps**0.5 / (num_steps**0.5)
            return max(min_lr, lr)

        num_steps_ = self.num_steps - self.lr_warmup_steps
        decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0
        delta_lr = max_lr - min_lr

        if self.lr_decay_style == "linear":
            coeff = 1.0 - decay_ratio
        elif self.lr_decay_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        else:
            raise Exception(
                "{} decay style is not supported.".format(self.lr_decay_style)
            )

        return min_lr + coeff * delta_lr

    def step_absolute(self, num_steps):
        """Set lr for all parameters groups."""
        if num_steps is None:
            self.num_steps += 1
        else:
            self.num_steps = num_steps
        new_wd = self.get_wd()
        for param_group in self.optimizer.param_groups:
            new_lr = self.get_lr(param_group)
            param_group["lr"] = new_lr * param_group.get("lr_mult", 1.0)
            param_group["weight_decay"] = new_wd * param_group.get("wd_mult", 1.0)

    def step(self, increment):
        """Set lr for all parameters groups."""
        self.num_steps += increment
        new_wd = self.get_wd()
        for param_group in self.optimizer.param_groups:
            new_lr = self.get_lr(param_group)
            param_group["lr"] = new_lr * param_group.get("lr_mult", 1.0)
            param_group["weight_decay"] = new_wd * param_group.get("wd_mult", 1.0)

    def state_dict(self):
        state_dict = {
            "max_lr": self.max_lr,
            "lr_warmup_steps": self.lr_warmup_steps,
            "num_steps": self.num_steps,
            "lr_decay_style": self.lr_decay_style,
            "lr_decay_steps": self.lr_decay_steps,
            "min_lr": self.min_lr,
            "start_wd": self.start_wd,
            "end_wd": self.end_wd,
            "wd_incr_style": self.wd_incr_style,
            "wd_incr_steps": self.wd_incr_steps,
        }
        return state_dict

    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_opt_param_scheduler:
            self.log_rank_0(" > overriding {} value to {}".format(name, cls_value))
            return cls_value

        if not self.use_checkpoint_opt_param_scheduler:
            assert cls_value == sd_value, (
                f"OptimizerParamScheduler: class input value {cls_value} and checkpoint"
                f"value {sd_value} for {name} do not match"
            )
        self.log_rank_0(" > using checkpoint value {} for {}".format(sd_value, name))
        return sd_value

    def load_state_dict(self, sd):

        if "start_lr" in sd:
            max_lr_ = sd["start_lr"]
        else:
            max_lr_ = sd["max_lr"]
        self.max_lr = self._check_and_set(self.max_lr, max_lr_, "learning rate")

        self.min_lr = self._check_and_set(
            self.min_lr, sd["min_lr"], "minimum learning rate"
        )

        if "warmup_iter" in sd:
            lr_warmup_steps_ = sd["warmup_iter"]
        elif "warmup_steps" in sd:
            lr_warmup_steps_ = sd["warmup_steps"]
        else:
            lr_warmup_steps_ = sd["lr_warmup_steps"]
        self.lr_warmup_steps = self._check_and_set(
            self.lr_warmup_steps, lr_warmup_steps_, "warmup iterations"
        )

        if "end_iter" in sd:
            lr_decay_steps_ = sd["end_iter"]
        elif "decay_steps" in sd:
            lr_decay_steps_ = sd["decay_steps"]
        else:
            lr_decay_steps_ = sd["lr_decay_steps"]
        self.lr_decay_steps = self._check_and_set(
            self.lr_decay_steps, lr_decay_steps_, "total number of iterations"
        )

        if "decay_style" in sd:
            lr_decay_style_ = sd["decay_style"]
        else:
            lr_decay_style_ = sd["lr_decay_style"]
        self.lr_decay_style = self._check_and_set(
            self.lr_decay_style, lr_decay_style_, "learning rate decay style"
        )

        if "num_iters" in sd:
            num_steps = sd["num_iters"]
        else:
            num_steps = sd["num_steps"]
        self.step(increment=num_steps)

        if "start_wd" in sd:
            self.start_wd = self._check_and_set(
                self.start_wd, sd["start_wd"], "start weight decay"
            )
            self.end_wd = self._check_and_set(
                self.end_wd, sd["end_wd"], "end weight decay"
            )
            self.wd_incr_steps = self._check_and_set(
                self.wd_incr_steps,
                sd["wd_incr_steps"],
                "total number of weight decay iterations",
            )
            self.wd_incr_style = self._check_and_set(
                self.wd_incr_style,
                sd["wd_incr_style"],
                "weight decay incr style",
            )


@torch.no_grad()
def _step_megatron_distrib_optimizer_internal(optim: DistributedOptimizer):
    # NOTE: patching this function to use the correct model parallel group

    optim._copy_model_grads_to_main_grads()

    # Do unscale, check for inf, and update grad scaler only for
    # the case that grad scaler is provided.
    if optim.grad_scaler:

        def _unscale_main_grads_and_check_for_nan(optim: DistributedOptimizer):

            # Collect main grads.
            main_grads = optim._collect_main_grad_data_for_unscaling()

            # Reset found inf.
            optim.found_inf.fill_(0.0)

            # Unscale and set found inf/nan
            torch._amp_foreach_non_finite_check_and_unscale_(
                main_grads, optim.found_inf, optim.grad_scaler.inv_scale
            )

            # Update across all model parallel instances.
            dist.all_reduce(
                optim.found_inf,
                op=dist.ReduceOp.MAX,
                group=constants.parallelism_group(),
            )

            # Check for nan.
            found_inf_flag = optim.found_inf.item() > 0

            return found_inf_flag

        # Unscale and check for inf/nan.
        found_inf_flag = _unscale_main_grads_and_check_for_nan(optim)

        # We are done with scaling gradients
        # so we can update the loss scale.
        optim.grad_scaler.update(found_inf_flag)

        # If we found inf/nan, skip the update.
        if found_inf_flag:
            optim.update_successful, grad_norm, num_zeros_in_grad = (
                False,
                None,
                None,
            )
            return optim.update_successful, grad_norm, num_zeros_in_grad

    def clip_grad_norm(optim: DistributedOptimizer, clip_grad: float) -> float:
        """Compute grad norm."""
        params = optim.get_parameters()
        grads_for_norm = optim.get_main_grads_for_grad_norm()
        return clip_grad_norm_fp32(
            params,
            grads_for_norm,
            clip_grad,
            model_parallel_group=constants.parallelism_group(),
        )

    def count_zeros(optim: DistributedOptimizer) -> float:
        """Count number of zeros in model's gradients."""
        params = optim.get_parameters()
        return count_zeros_fp32(
            params,
            model_parallel_group=constants.parallelism_group(),
        )

    # Clip the main gradients.
    grad_norm = None
    if optim.config.clip_grad > 0.0:
        grad_norm = clip_grad_norm(optim, optim.config.clip_grad)

    # Count the zeros in the grads.
    num_zeros_in_grad = None
    if optim.config.log_num_zeros_in_grad:
        num_zeros_in_grad = count_zeros(optim)

    # Step the optimizer.
    optim.optimizer.step()

    # Update params from main params.
    optim._copy_main_params_to_model_params()
    # Successful update.
    optim.update_successful, grad_norm, num_zeros_in_grad = (
        True,
        grad_norm,
        num_zeros_in_grad,
    )
    return optim.update_successful, grad_norm, num_zeros_in_grad


def step_megatron_distrb_optimizer(optim: DistributedOptimizer):

    optim.update_successful, grad_norm, num_zeros_in_grad = (
        _step_megatron_distrib_optimizer_internal(optim)
    )

    # If not overlapping all-gather for parameters, launch synchronous all-gather
    # communication calls here. If overlapping all-gather for parameters, the following
    # call to _gather_all_model_params is a no-op: the first all-gather is launched
    # asynchronously in the next optimizer.zero_grad() call and subsequent all-gathers
    # are launched in the forward pre-hook.
    optim._reset_metadata_and_sync_gather_all_model_params(force_sync=False)

    return optim.update_successful, grad_norm, num_zeros_in_grad


def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    return torch._C._nn.flatten_dense_tensors(tensors)


def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are
    of same dense type, and that flat is given by _flatten_dense_tensors.

    Args:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    return torch._C._nn.unflatten_dense_tensors(flat, tensors)


def megatron_all_reduce_layernorm_grads(engine: MegatronEngine):
    if not (
        constants.sequence_parallel() and constants.model_parallel_world_size() > 1
    ):
        return
    real_model: ReaLModel = engine.ddp.module
    grads = []
    for i in range(real_model.layer_idx_start, real_model.layer_idx_end):
        if i == 0:
            continue
        elif i == real_model.config.n_layers + 1:
            continue
        else:
            assert 0 < i < real_model.config.n_layers + 1
            layer: ReaLModelBlock = real_model.layers[i - real_model.layer_idx_start]
            grads.append(layer.attn.c_attn.ln.weight.main_grad)
            if getattr(layer.attn.c_attn.ln, "bias", None) is not None:
                grads.append(layer.attn.c_attn.ln.bias.main_grad)
            grads.append(layer.mlp.ln.weight.main_grad)
            if getattr(layer.mlp.ln, "bias", None) is not None:
                grads.append(layer.mlp.ln.bias.main_grad)
            if i == real_model.config.n_layers:
                grads.append(layer.ln_f.weight.main_grad)
                if getattr(layer.ln_f, "bias", None) is not None:
                    grads.append(layer.ln_f.bias.main_grad)

    assert all(x is not None for x in grads)
    coalesced = _flatten_dense_tensors(grads)
    dist.all_reduce(coalesced, group=constants.model_parallel_group())
    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
        buf.copy_(synced)


def megatron_all_reduce_word_embedding_grads(engine: MegatronEngine):
    real_model: ReaLModel = engine.ddp.module
    if not real_model.config.tied_embedding or real_model.config.is_critic:
        return
    pp_size = constants.pipe_parallel_world_size()
    pp_rank = constants.pipe_parallel_rank()
    if pp_size == 1:
        return
    if pp_rank not in [0, pp_size - 1]:
        return

    if pp_rank == 0:
        grad = real_model.layers[0].wte.weight.main_grad
    else:
        grad = real_model.layers[-1].weight.main_grad

    dist.all_reduce(grad, group=constants.grid().embedding_proc_group)


def finalize_grads_megatron(engine: MegatronEngine):
    engine.ddp.finish_grad_sync()
    megatron_all_reduce_layernorm_grads(engine)
    megatron_all_reduce_word_embedding_grads(engine)


@dataclasses.dataclass
class PipeTrainInstrSetForMegatron(PipeTrainInstrSet):
    # NOTE: merge DistributedDataParallel and DistributedOptimizer into one class
    # to remain consistent with DeepSpeed's API
    engine: MegatronEngine
    num_micro_batches: int

    def __post_init__(self):
        self._no_sync_context = None
        self.disable_grad_sync()

    def disable_grad_sync(self):
        if self._no_sync_context is None:
            self._no_sync_context = self.engine.ddp.no_sync()
            self._no_sync_context.__enter__()

    def enable_grad_sync(self):
        if self._no_sync_context is not None:
            self._no_sync_context.__exit__(None, None, None)
            self._no_sync_context = None

    @cuda_tmarked("bwd", CUDATimeMarkType.backward)
    def _exec_backward_pass(
        self,
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        output_x = tensor_buffer.get("batch_output_x", micro_batch_id, remove=True)

        if micro_batch_id == self.num_micro_batches - 1:
            self.enable_grad_sync()

        is_last_stage = constants.is_last_pipe_stage()
        if is_last_stage:
            loss: torch.Tensor = tensor_buffer.get(
                "losses", micro_batch_id, remove=True
            )
            loss.backward()
            tensor_buffer.put("losses", micro_batch_id, loss.detach().clone())
            return

        grad = tensor_buffer.get("grad", micro_batch_id, remove=True)
        output_tensor = output_x.pp_output
        torch.autograd.backward(tensors=output_tensor, grad_tensors=grad)

    def _exec_reduce_grads(
        self,
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        # self.engine.ddp.start_grad_sync()
        finalize_grads_megatron(self.engine)

    @cuda_tmarked("opt", CUDATimeMarkType.optim_step)
    def _exec_optimizer_step(
        self,
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        if isinstance(self.engine.optim, DistributedOptimizer):
            update_successful, grad_norm, num_zeros_in_grad = (
                step_megatron_distrb_optimizer(self.engine.optim)
            )
        else:
            update_successful, grad_norm, num_zeros_in_grad = self.engine.optim.step()

        version_steps = tensor_buffer.get("version_steps", 0)
        if update_successful:
            self.engine.lr_scheduler.step_absolute(version_steps)
        if constants.data_parallel_rank() == 0 and constants.model_parallel_rank() == 0:
            logger.info(
                f"Model name {constants.model_name()}, "
                f"Pipeline rank {constants.pipe_parallel_rank()}. "
                f"Update success? {update_successful}. "
                f"Grad Norm: {grad_norm}. "
                f"Current loss scale: {self.engine.optim.get_loss_scale()}. "
                f"Learning rate: {[param_group['lr'] for param_group in self.engine.optim.param_groups]}. "
            )
        return update_successful, grad_norm, num_zeros_in_grad


class ReaLMegatronEngine(model_api.PipelinableEngine):

    def __init__(self, module: ReaLModel, megatron_engine: MegatronEngine):
        self.module = module

        self.inf_engine = PipelinableInferenceEngine(module)
        if constants.pipe_parallel_world_size() > 1:
            self.pipe_runner = self.inf_engine.pipe_runner

        # NOTE: In profiler, module could be not a instance of ReaLModel.
        self.device = module.device if hasattr(module, "device") else None
        self.dtype = module.dtype if hasattr(module, "dtype") else None

        self.engine = megatron_engine

    def train(self, mode: bool = True):
        self.module.train(mode)
        self.engine.ddp.train(mode)
        return self

    def eval(self):
        self.module.eval()
        self.engine.ddp.eval()
        return self

    def train_batch(
        self,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
        loss_fn: Callable,
        loss_weight_fn: Callable,
        token_normalize_scope: str,
        version_steps: int,
    ):
        with megatron_ctx():
            self.engine.zero_grad()
            if constants.pipe_parallel_world_size() > 1:
                mb_inputs = input_.synced_data_parallel_split(
                    MicroBatchSpec.new(
                        mb_spec,
                        n_mbs=mb_spec.n_mbs * self.pipe_runner.default_train_mbs,
                    )
                )
                # Fusing the minibatched forward-backward in a pipeline training schedule.
                instr_set = PipeTrainInstrSetForMegatron(self.engine, len(mb_inputs))
                # NOTE: When training with pipeline parallel, num micro batches should be
                # larger than 2 x num_pipeline_stages to avoid idle time.
                return self.pipe_runner.train_batch(
                    instr_set=instr_set,
                    input_=input_,
                    mb_spec=mb_spec,
                    loss_fn=loss_fn,
                    loss_weight_fn=loss_weight_fn,
                    token_normalize_scope=token_normalize_scope,
                    version_steps=version_steps,
                )

            mb_inputs = input_.synced_data_parallel_split(mb_spec)
            total_loss_weight = torch.tensor(
                sum([loss_weight_fn(mb) for mb in mb_inputs]), dtype=torch.float32
            )
            if token_normalize_scope == "global":
                dist.all_reduce(
                    total_loss_weight, group=constants.data_parallel_group()
                )
            if total_loss_weight == 0:
                raise model_api.ZeroTotalLossWeightException(
                    "The sum of loss weights of all micro batches is zero."
                )

            if constants.parallelism_rank() == 0:
                logger.info(
                    f"MB spec: {mb_spec}, #mbs={len(mb_inputs)}, "
                    f"#tokens: {input_.data['packed_input_ids'].shape[0]}, "
                    f"pp_size={constants.pipe_parallel_world_size()}, "
                    f"#tokens per mbs: {[mb.data['packed_input_ids'].shape[0] for mb in mb_inputs]}"
                )
            no_sync_ctx = self.engine.ddp.no_sync()
            no_sync_ctx.__enter__()
            stat = collections.defaultdict(int)
            for i, mb_input in enumerate(mb_inputs):
                if i == len(mb_inputs) - 1:
                    no_sync_ctx.__exit__(None, None, None)
                input_lens = torch.tensor(
                    flat2d(mb_input.seqlens["packed_input_ids"]),
                    dtype=torch.int32,
                    device="cuda",
                )
                max_seqlen = int(max(input_lens))
                cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
                model_output = self.engine.ddp(
                    packed_input_ids=mb_input.data["packed_input_ids"],
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                ).logits
                loss, _stat = loss_fn(model_output, mb_input)
                loss_scale = loss_weight_fn(mb_inputs[i]) / total_loss_weight
                if token_normalize_scope == "global":
                    # Megatron will average gradients across DP ranks.
                    # If we normalize loss across micro batches of all DP ranks,
                    # we should revert the effect of gradient averaging in megatron
                    # to make sure loss from each token is scaled properly.
                    loss_scale *= constants.data_parallel_world_size()
                loss_scale *= self.engine.optim.get_loss_scale().item()
                loss *= loss_scale
                with cuda_tmarked("bwd", CUDATimeMarkType.backward):
                    loss.backward()
                for k, v in _stat.items():
                    stat[k] += v

            finalize_grads_megatron(self.engine)
            self._step(version_steps)

            return stat

    @torch.no_grad()
    def forward(
        self,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
        post_hook: Callable[[torch.Tensor, SequenceSample], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ):
        return self.inf_engine.forward(
            input_=input_,
            mb_spec=mb_spec,
            post_hook=post_hook,
            aggregate_fn=aggregate_fn,
        )

    @torch.no_grad()
    def generate(
        self,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: model_api.GenerationHyperparameters = dataclasses.field(
            default_factory=model_api.GenerationHyperparameters
        ),
    ):
        return self.inf_engine.generate(
            input_=input_,
            tokenizer=tokenizer,
            gconfig=gconfig,
            mb_spec=mb_spec,
        )

    # wrapper for profiler
    @cuda_tmarked("opt", CUDATimeMarkType.optim_step)
    def _step(self, version_steps):
        if isinstance(self.engine.optim, DistributedOptimizer):
            update_successful, grad_norm, _ = step_megatron_distrb_optimizer(
                self.engine.optim
            )
        else:
            update_successful, grad_norm, _ = self.engine.optim.step()
        if update_successful:
            self.engine.lr_scheduler.step_absolute(version_steps)
        if constants.data_parallel_rank() == 0 and constants.model_parallel_rank() == 0:
            logger.info(
                f"Megatron backend update success? {update_successful}. "
                f"Grad Norm: {grad_norm}. "
                f"Current loss scale: {self.engine.optim.get_loss_scale()}. "
                f"Learning rate: {[param_group['lr'] for param_group in self.engine.optim.param_groups]}. "
            )


@dataclasses.dataclass
class MegatronTrainBackend(model_api.ModelBackend, MegatronConfig):
    enable_fp16: bool = True
    enable_bf16: bool = False
    zero_stage: int = dataclasses.field(
        metadata={"choices": [0, 1, 2, 3]},
        default=2,
    )
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)

    def _initialize(
        self, model: model_api.Model, spec: model_api.FinetuneSpec
    ) -> model_api.Model:
        module = model.module
        if not isinstance(module, ReaLModel):
            raise ValueError("MegatronTrainBackend only supports ReaLModel.")
        with megatron_ctx():
            module = DistributedDataParallel(
                config=get_megatron_transformer_config(module.config),
                module=module,
                data_parallel_group=constants.data_parallel_group(),
                accumulate_allreduce_grads_in_fp32=self.accumulate_allreduce_grads_in_fp32,
                overlap_grad_reduce=self.overlap_grad_reduce,
                use_distributed_optimizer=self.zero_stage > 0,
                expert_data_parallel_group=None,
                disable_bucketing=False,
                check_for_nan_in_grad=False,
            )

        real_model: ReaLModel = module.module
        if self.zero_stage > 0:
            # Remap parameters.
            assert len(module.buffers) == 1
            param_grad_buf: ParamAndGradBuffer = module.buffers[0]

            # Map Megatron flattened parameters to ReaLModel!
            real_model.contiguous_param = param_grad_buf.param_data

            # Sanity checks.
            assert real_model._param_size == param_grad_buf.numel, (
                real_model._param_size,
                param_grad_buf.numel,
                module.bucket_size,
            )
            for n, p in real_model.layers.named_parameters():
                n = ".".join(
                    [
                        str(real_model.layer_idx_start + int(n.split(".")[0])),
                        n.split(".", 1)[1],
                    ]
                )
                idx_start, idx_end, _ = param_grad_buf.param_index_map[p]
                assert real_model._param_spec[n].start_idx == idx_start
                assert real_model._param_spec[n].end_idx == idx_end
                assert real_model._param_spec[n].shape == p.shape
                assert torch.allclose(
                    p,
                    real_model.contiguous_param[idx_start:idx_end].view(p.shape),
                )

        betas = (self.optimizer.beta1, self.optimizer.beta2)
        wd = self.optimizer.weight_decay
        lr = self.optimizer.lr
        opt_cfg = MegatronOptimizerConfig(
            optimizer=self.optimizer.type,
            fp16=self.enable_fp16,
            bf16=self.enable_bf16,
            lr=lr,
            min_lr=self.optimizer.min_lr_ratio * lr,
            weight_decay=wd,
            params_dtype=real_model.dtype,
            initial_loss_scale=self.optimizer.initial_loss_scale,
            adam_beta1=betas[0],
            adam_beta2=betas[1],
            adam_eps=self.optimizer.eps,
            sgd_momentum=0.9,
            use_distributed_optimizer=self.zero_stage > 0,
            overlap_grad_reduce=self.overlap_grad_reduce,
            overlap_param_gather=self.overlap_param_gather,
            clip_grad=self.optimizer.gradient_clipping,
            min_loss_scale=self.optimizer.min_loss_scale,
            loss_scale_window=self.optimizer.loss_scale_window,
            hysteresis=self.optimizer.hysteresis,
        )

        with megatron_ctx():
            # no_weight_decay_cond and scale_lr_cond have the following signature:
            # foo(name: str, param: torch.Tensor) -> bool
            optimizer = get_megatron_optimizer(
                opt_cfg,
                [module],
                no_weight_decay_cond=lambda n, p: any(
                    k in n for k in ["bias", "ln.weight", "ln_f.weight"]
                ),
                scale_lr_cond=None,
                lr_mult=1.0,
            )

            warmup_steps_proportion = self.optimizer.warmup_steps_proportion
            warmup_steps = int(warmup_steps_proportion * spec.total_train_steps)
            lr_scheduler = OptimizerParamScheduler(
                optimizer,
                init_lr=0.0 if warmup_steps_proportion > 0 else lr,
                max_lr=lr,
                min_lr=self.optimizer.min_lr_ratio * lr,
                lr_warmup_steps=warmup_steps,
                lr_decay_steps=spec.total_train_steps - warmup_steps,
                lr_decay_style=self.optimizer.lr_scheduler_type,
                start_wd=wd,
                end_wd=wd,
                wd_incr_steps=spec.total_train_steps,
                wd_incr_style="constant",
            )

        mg_engine = MegatronEngine(module, optimizer, lr_scheduler)
        model.module = ReaLMegatronEngine(real_model, mg_engine)
        model.backend_name = "megatron"
        return model

    def destroy(self, model: model_api.Model):
        assert isinstance(model.module, ReaLMegatronEngine)
        optimizer = model.module.engine.optim
        # The Megatron backend will register forward hooks that
        # create circular references (grad -> param -> grad).
        # Deleting models directly will not release the memory.
        # We must disable hooks at first.
        if self.zero_stage > 0 and self.overlap_param_gather:
            optimizer.disable_pre_hook()

    def save(self, model: model_api.Model, save_dir: str):
        assert isinstance(model.module, ReaLMegatronEngine)
        optimizer = model.module.engine.optim
        param_state = optimizer.get_parameter_state_fs_bucket_space()
        sd = optimizer.state_dict()
        dp = constants.data_parallel_rank()
        pp = constants.pipe_parallel_rank()
        tp = constants.model_parallel_rank()
        # HACK: (bowei) I'm not sure whether there's duplicated information.
        torch.save(
            sd, pathlib.Path(save_dir) / f"megatron_optim_sd_d{dp}p{pp}t{tp}.mckpt"
        )
        torch.save(
            param_state,
            pathlib.Path(save_dir) / f"megatron_optim_param_sd_d{dp}p{pp}t{tp}.mckpt",
        )

    def load(self, model: model_api.Model, load_dir: str):
        assert isinstance(model.module, ReaLMegatronEngine)
        optimizer = model.module.engine.optim

        dp = constants.data_parallel_rank()
        pp = constants.pipe_parallel_rank()
        tp = constants.model_parallel_rank()

        sd = torch.load(
            pathlib.Path(load_dir) / f"megatron_optim_sd_d{dp}p{pp}t{tp}.mckpt"
        )
        optimizer.load_state_dict(sd)

        param_state = torch.load(
            pathlib.Path(load_dir) / f"megatron_optim_param_sd_d{dp}p{pp}t{tp}.mckpt"
        )
        optimizer.load_parameter_state_from_fs_bucket_space(param_state)


model_api.register_backend("megatron", MegatronTrainBackend)
