from __future__ import annotations

import functools
import os
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import mbridge
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig as MCoreOptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from torch import nn
from torch_memory_saver import torch_memory_saver
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import MicroBatchSpec
from areal.api.io_struct import FinetuneSpec
from areal.api.train_engine import TrainEngineComputeMixin
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.core import (
    aggregate_eval_losses,
    compute_total_loss_weight,
    reorder_and_pad_outputs,
)
from areal.models.mcore.registry import make_hf_and_mcore_config, make_mcore_model
from areal.platforms import current_platform
from areal.utils import stats_tracker
from areal.utils.data import (
    MicroBatchItem,
    MicroBatchList,
    amend_position_ids,
    broadcast_tensor,
    pack_tensor_dict,
    pad_mb_list,
    split_padded_tensor_dict_into_mb_list,
    unpad_logits,
)
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.lock import DistributedLock
from areal.utils.mcore.determinisitc import set_deterministic_algorithms
from areal.utils.mcore.packed_context_parallel import (
    packed_context_parallel_forward,
)
from areal.utils.mcore.pipeline_parallel import configure_pipeline_layer_splits
from areal.utils.megatron_checkpointer import MegatronCheckpointManager
from areal.utils.model import disable_dropout_in_model
from areal.utils.offload import is_tms_enabled
from areal.utils.perf_tracer import trace_scope
from areal.utils.seeding import get_seed

if TYPE_CHECKING:
    from areal.engine.megatron.protocol import MegatronEngineProtocol


class _MegatronModelList(list):
    """List wrapper that exposes module-like helpers for Megatron model chunks."""

    def forward(self, *args, **kwargs) -> Any:
        if len(self) == 1:
            return self[0](*args, **kwargs)
        raise RuntimeError(
            "Direct forward calls are only supported for single-chunk model list."
        )

    def named_parameters(self, *args, **kwargs) -> Iterator[tuple[str, nn.Parameter]]:
        for module in self:
            yield from module.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs) -> Iterator[nn.Parameter]:
        for _, parameter in self.named_parameters(*args, **kwargs):
            yield parameter


class MegatronComputeMixin(TrainEngineComputeMixin):
    def initialize(
        self: MegatronEngineProtocol,
        addr: str | None,
        ft_spec: FinetuneSpec,
    ):
        try:
            self.seed = get_seed()
        except ValueError:
            self.logger.warning("Seed not set, using default seed 42.")
            self.seed = 42

        assert addr is None, "FSDPEngine does not support remote initialization."

        if is_tms_enabled():
            torch_memory_saver.hook_mode = "preload"

        current_platform.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.device(int(os.environ["LOCAL_RANK"]))
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.is_pp_head = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
        )
        self.weight_update_group_name = (
            f"update_weight_group_{mpu.get_pipeline_model_parallel_rank()}"
        )
        self.engine_lock = DistributedLock("train_engine_lock")

        self.tokenizer = load_hf_tokenizer(self.config.path)
        self.bridge = mbridge.AutoBridge.from_pretrained(self.config.path)
        self.bridge.dtype = self.dtype
        # Set gradient checkpointing options
        if self.config.gradient_checkpointing:
            self.bridge.set_extra_args(
                recompute_granularity=self.mcore_config.recompute_granularity,
                recompute_method=self.mcore_config.recompute_method,
                recompute_num_layers=self.mcore_config.recompute_num_layers,
                distribute_saved_activations=self.mcore_config.distribute_saved_activations,
                recompute_modules=self.mcore_config.recompute_modules,
            )

        self.logger.info(
            "Using mbridge to create models and hf model save/load in MegatronEngine."
        )

        self.hf_config, self.tf_config = make_hf_and_mcore_config(
            self.config.path, dtype=self.dtype, bridge=self.bridge
        )
        # TODO: configure for VPP
        self.tf_config = configure_pipeline_layer_splits(
            self.parallel_strategy, self.hf_config, self.tf_config
        )

        # initialize mcore (DDP Wrapped) GPTModel
        with self.device:
            models = make_mcore_model(
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                mcore_config=self.mcore_config,
                bridge=self.bridge,
                is_critic=self.config.is_critic,
            )

        self.model = _MegatronModelList(models)

        with self.device:
            self._load_model_from_hf(self.config.path)

        assert self.model, "Megatron models failed to initialize."
        modules = [m.module if isinstance(m, DDP) else m for m in self.model]
        total_params = sum(
            param.numel() for module in modules for param in module.parameters()
        )
        self.logger.info(
            f"Model parameter count: {total_params / 1e6:.2f}M, pp_stage={mpu.get_pipeline_model_parallel_rank()}, vpp_chunks={len(self.model)}"
        )

        if self.config.disable_dropout:
            for model in self.model:
                disable_dropout_in_model(model)

        # TODO: Check verl implementation
        primary_model = self.model[0]
        model_config = get_model_config(primary_model)
        # NOTE: It is recommended to set this option to True for RL training on MoE models for stability.
        if self.mcore_config.use_deterministic_algorithms:
            set_deterministic_algorithms(model_config)

        # Set vp_stage for DDP models
        for i, model_chunk in enumerate(self.model):
            if (
                isinstance(model_chunk, DDP)
                and self.mcore_config.virtual_pipeline_parallel_size > 1
            ):
                vp_stage = getattr(model_chunk.module, "vp_stage", None)
                self.logger.info(f"Setting vp_stage {vp_stage} for model chunk {i}.")
                setattr(model_chunk, "vp_stage", vp_stage)

        if self.mcore_config.ddp.overlap_grad_reduce and isinstance(primary_model, DDP):
            model_config.no_sync_func = [
                model_chunk.no_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                model_config.no_sync_func = model_config.no_sync_func[0]

        if (
            self.mcore_config.ddp.overlap_param_gather
            and self.mcore_config.ddp.align_param_gather
        ):
            model_config.param_sync_func = [
                model_chunk.start_param_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                model_config.param_sync_func = model_config.param_sync_func[0]
        model_config.finalize_model_grads_func = finalize_model_grads
        self._create_optimizer(ft_spec)

    def train(self: MegatronEngineProtocol, mode: bool = True):
        assert self.model is not None
        for model in self.model:
            model.train(mode=mode)
        return self

    def rollout_batch(
        self: MegatronEngineProtocol,
        data: list[dict[str, Any]],
        granularity: int = 1,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.rollout_batch(
            data,
            granularity=granularity,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
        )

    def prepare_batch(
        self: MegatronEngineProtocol,
        dataloader: StatefulDataLoader,
        granularity: int = 1,
        workflow: RolloutWorkflow | type[RolloutWorkflow] | str | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
    ) -> dict[str, Any]:
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.prepare_batch(
            dataloader,
            granularity=granularity,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
        )

    def optimizer_zero_grad(self: MegatronEngineProtocol):
        assert self.optimizer is not None, "Optimizer is not initialized."
        self.optimizer.zero_grad()
        for model in self.model:
            model.zero_grad_buffer()

    def optimizer_step(self: MegatronEngineProtocol):
        with trace_scope("megatron_engine.step"):
            update_successful, grad_norm, _ = self.optimizer.step()
        current_lr = self.optimizer.param_groups[0]["lr"]

        return dict(
            update_successful=float(update_successful),
            grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
            lr=current_lr,
        )

    def lr_scheduler_step(self: MegatronEngineProtocol):
        assert self.lr_scheduler is not None, "LR Scheduler is not initialized."
        self.lr_scheduler.step(1)

    def forward_backward_batch(
        self: MegatronEngineProtocol,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool = False,
    ) -> None:
        self._ensure_ready()

        def forward_step(batch_iter, model):
            mb_input: MicroBatchItem = next(batch_iter)

            cu_seqlens = mb_input.padded_mb["cu_seqlens"]
            output = packed_context_parallel_forward(model, mb_input.padded_mb)

            def _process_output(input_, output_):
                loss = process_output_fn(output_, input_)
                if loss is None:
                    loss = torch.tensor(1.0, device=output_.device)
                return loss, {}

            model_vp_stage = getattr(model, "vp_stage", 0)
            if mpu.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=model_vp_stage
            ):
                output = unpad_logits(
                    output,
                    padding_length=mb_input.padding_length,
                    cu_seqlens=cu_seqlens,
                    old_cu_seqlens=mb_input.old_cu_seqlens,
                )
            return output, functools.partial(_process_output, mb_input.orig_mb)

        forward_backward_func = get_forward_backward_func()
        with trace_scope("megatron_engine.forward_backward"):
            if len(self.model) > 1:
                data_iterator = [iter(mb_list) for _ in range(len(self.model))]
            else:
                data_iterator = iter(mb_list)
            forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=data_iterator,
                model=self.model if len(self.model) > 1 else self.model[0],
                num_microbatches=len(mb_list),
                seq_length=mb_list.max_seqlen,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )

    def train_batch(
        self: MegatronEngineProtocol,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        self._ensure_ready()
        self.optimizer_zero_grad()

        # Step 1: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_).to(self.device)

        # Step 2: Compute total loss weight
        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, mpu.get_data_parallel_group()
        )

        # Step 3: Forward-backward using Megatron's pipeline function
        loss_multiplier = (
            mpu.get_data_parallel_world_size() * self.optimizer.get_loss_scale().item()
        )

        def process_output(
            output: torch.Tensor, inputs: dict[str, Any]
        ) -> torch.Tensor:
            return self._compute_logprobs_and_loss(
                output,
                inputs,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
                loss_multiplier=loss_multiplier,
            )

        self.forward_backward_batch(mb_list, process_output, forward_only=False)

        # Step 4: Optimizer step
        return self.optimizer_step()

    @torch.no_grad()
    def eval_batch(
        self: MegatronEngineProtocol,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        self._ensure_ready()

        # Step 1: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_).to(self.device)

        # Step 2: Compute total loss weight
        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, mpu.get_data_parallel_group()
        )

        # Step 3: Forward using Megatron's pipeline function, collecting losses
        losses: list[torch.Tensor] = []

        def process_output(
            output: torch.Tensor, inputs: dict[str, Any]
        ) -> torch.Tensor:
            loss = self._compute_logprobs_and_loss(
                output, inputs, loss_fn, loss_weight_fn, total_loss_weight
            )
            losses.append(loss.detach())
            return loss

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        # Step 4: Aggregate losses
        if mpu.is_pipeline_last_stage():
            return aggregate_eval_losses(losses, mpu.get_data_parallel_group())
        return None

    @torch.no_grad()
    def forward_batch(
        self: MegatronEngineProtocol,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> torch.Tensor:
        self._ensure_ready()

        # Step 1: Prepare sequence lengths
        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]
        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
        assert output_seqlens is not None

        # Step 2: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_).to(self.device)

        # Step 3: Forward using Megatron's pipeline function, collecting results
        outputs: list[torch.Tensor] = []

        def process_output(output: torch.Tensor, inputs: dict[str, Any]) -> None:
            result = self._compute_forward_result(output, inputs)
            outputs.append(result)
            return None

        self.forward_backward_batch(mb_list, process_output, forward_only=True)

        # Step 4: Aggregate, reorder, and broadcast outputs
        res = None
        if mpu.is_pipeline_last_stage():
            res = reorder_and_pad_outputs(
                outputs, output_seqlens, mb_list, aggregate_fn
            )
        res = broadcast_tensor(
            res,
            src_rank=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        return res

    def export_stats(self: MegatronEngineProtocol) -> dict[str, float]:
        data = stats_tracker.export_all(reduce_group=self.data_parallel_group)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            # Some log info only exist in last pipeline rank
            data_list = [data]
            dist.broadcast_object_list(
                data_list,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )
            data.update(data_list[0])
        return data

    def _check_rollout_engine_connected(self: MegatronEngineProtocol) -> None:
        """Validate that rollout engine has been connected via connect_engine()."""
        if self.rollout_engine is None or self.rollout_coordinator is None:
            raise RuntimeError(
                "Rollout engine not connected. Call connect_engine()"
                " before using rollout/update_weight methods."
            )

    def _ensure_ready(self: MegatronEngineProtocol) -> None:
        if self.is_offload:
            self.onload()

        if self.model is None:
            raise RuntimeError("Model is not initialized.")

    def _create_optimizer(
        self: MegatronEngineProtocol,
        ft_spec: FinetuneSpec,
    ) -> None:
        if self.optimizer_config is None:
            return
        assert self.model is not None and len(self.model) > 0

        assert self.optimizer_config.type in [
            "adam",
            "sgd",
        ], "Only AdamW/sgd optimizer is supported in this engine."
        if self.optimizer_config.type == "sgd":
            self.logger.warning(
                "Using the 'sgd' optimizer with Megatron may be less stable. Consider using the 'adam' (AdamW) optimizer for improved stability."
            )

        # Make megatron optimizer config
        mcore_opt_config = MCoreOptimizerConfig(
            optimizer=self.optimizer_config.type,
            lr=self.optimizer_config.lr,
            min_lr=self.optimizer_config.min_lr_ratio * self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
            bf16=self.dtype is torch.bfloat16,
            fp16=self.dtype is torch.float16,
            adam_beta1=self.optimizer_config.beta1,
            adam_beta2=self.optimizer_config.beta2,
            adam_eps=self.optimizer_config.eps,
            use_distributed_optimizer=self.mcore_config.ddp.use_distributed_optimizer,
            params_dtype=self.dtype,
            clip_grad=self.optimizer_config.gradient_clipping,
        )
        mcore_opt_config.overlap_param_gather_with_optimizer_step = (
            self.mcore_config.overlap_param_gather_with_optimizer_step
        )
        mcore_opt_config.use_precision_aware_optimizer = (
            self.mcore_config.use_precision_aware_optimizer
        )
        mcore_opt_config.main_grads_dtype = getattr(
            torch, self.mcore_config.main_grads_dtype
        )
        mcore_opt_config.main_params_dtype = getattr(
            torch, self.mcore_config.main_params_dtype
        )
        mcore_opt_config.exp_avg_dtype = getattr(torch, self.mcore_config.exp_avg_dtype)
        mcore_opt_config.exp_avg_sq_dtype = getattr(
            torch, self.mcore_config.exp_avg_sq_dtype
        )

        self.optimizer = get_megatron_optimizer(
            mcore_opt_config,
            self.model,
            no_weight_decay_cond=lambda n, p: any(
                k in n for k in ["bias", "ln.weight", "ln_f.weight"]
            ),
            scale_lr_cond=None,
            lr_mult=1.0,
        )

        warmup_steps_proportion = self.optimizer_config.warmup_steps_proportion
        warmup_steps = int(warmup_steps_proportion * ft_spec.total_train_steps)
        lr_scheduler = OptimizerParamScheduler(
            self.optimizer,
            init_lr=0.0 if warmup_steps_proportion > 0 else self.optimizer_config.lr,
            max_lr=self.optimizer_config.lr,
            min_lr=self.optimizer_config.min_lr_ratio * self.optimizer_config.lr,
            lr_warmup_steps=warmup_steps,
            lr_decay_steps=ft_spec.total_train_steps - warmup_steps,
            lr_decay_style=self.optimizer_config.lr_scheduler_type,
            start_wd=self.optimizer_config.weight_decay,
            end_wd=self.optimizer_config.weight_decay,
            wd_incr_steps=ft_spec.total_train_steps,
            wd_incr_style="constant",
        )
        self.lr_scheduler = lr_scheduler

        self.checkpointer = MegatronCheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            use_distributed_optimizer=self.mcore_config.ddp.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.mcore_config.use_checkpoint_opt_param_scheduler,
            async_save=self.mcore_config.async_save,
        )

    def _prepare_mb_list(
        self: MegatronEngineProtocol,
        input_: dict[str, Any],
    ) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_
        input_ = amend_position_ids(input_)
        # Parallel sizes
        pp_size = self.parallel_strategy.pipeline_parallel_size
        cp_size = self.parallel_strategy.context_parallel_size
        tp_size = self.parallel_strategy.tensor_parallel_size
        # Split the input into micro-batches
        # NOTE: Here we use 2*pp_size in forward to align logprob precision
        # TODO: Performance check
        min_n_mbs = (
            2 * pp_size if pp_size > 1 else 1
        )  # avoid pipeline bubbles in training
        # NOTE: self.config.mb_spec.max_tokens_per_mb determines
        # the expected **total** number of tokens per micro-batch **in the forward pass**.
        # The micro batch list splitted here will be splitted to each
        # context parallel rank, so the total number of tokens per
        # GPU in a forward pass here will be `max_tokens_per_mb / cp_size`.
        mb_spec = MicroBatchSpec.new(
            self.config.mb_spec,
            n_mbs=max(min_n_mbs, self.config.mb_spec.n_mbs),
            n_mbs_divisor=pp_size,
        )
        mb_list = split_padded_tensor_dict_into_mb_list(
            input_,
            mb_spec,
            group=mpu.get_data_parallel_group(),
        )
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
        # NOTE: Pad micro-batches to:
        # 1. Reduce GPU memory fragmentation, pad actual # tokens per mb to integer multiples
        #  of GPU page size or max_tokens_per_mb
        # 2. Align sequence lengths to integer multiples of `align_to_multiple_of=tp_size*cp_size*2`
        #    to satisfy the requirement of Megatron parallelism.
        align_to_multiple_of = tp_size * cp_size * 2 if cp_size > 1 else tp_size
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
            align_sequences=True,
            align_to_multiple_of=align_to_multiple_of,
        )
        self.logger.info(
            f"#microbatch: {len(mb_list.group_lens)}, microbatch #tokens: {mb_list.group_lens}, "
            f"aligned to: {mb_list.align_to_lengths}, padded to: {mb_list.padded_to_lengths}, "
            f"padding lengths: {mb_list.padding_lengths}."
        )
        # FIXME: the resulting max_seqlen is a tensor rather than an integer
        # Modern model implementations takes a dict as the input.
        # This eliminates a bug of Qwen2.5-VL for transformers<=4.53.1
        for i, mb in enumerate(mb_list.mbs):
            mb_list.mbs[i] = dict(**mb)
        for i, mb in enumerate(mb_list.padded_mbs):
            mb_list.padded_mbs[i] = dict(**mb)
        for mb in mb_list.mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
        for mb in mb_list.padded_mbs:
            mb["max_seqlen"] = int(mb["max_seqlen"])
        return mb_list

    def _compute_logprobs_and_loss(
        self: MegatronEngineProtocol,
        output: torch.Tensor,
        inputs: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        total_loss_weight: torch.Tensor,
        loss_multiplier: float = 1.0,
    ) -> torch.Tensor:
        if not self.config.is_critic:
            labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)
            logprobs, entropy = gather_logprobs_entropy(
                output,
                labels,
                temperature=self.config.temperature,
                tp_group=mpu.get_tensor_model_parallel_group()
                if mpu.get_tensor_model_parallel_world_size() > 1
                else None,
            )
            loss = loss_fn(logprobs, entropy, inputs)
        else:
            values = output.squeeze(-1)
            loss = loss_fn(values, inputs)

        loss_scale = loss_weight_fn(inputs) / total_loss_weight * loss_multiplier
        return loss * loss_scale

    def _compute_forward_result(
        self: MegatronEngineProtocol,
        output: torch.Tensor,
        inputs: dict[str, Any],
    ) -> torch.Tensor:
        if not self.config.is_critic:
            labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)
            logprobs = gather_logprobs(
                output,
                labels,
                temperature=self.config.temperature,
                tp_group=mpu.get_tensor_model_parallel_group()
                if mpu.get_tensor_model_parallel_world_size() > 1
                else None,
            )
            return logprobs
        else:
            values = output.squeeze(-1)
            return values
