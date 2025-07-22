# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import collections
import dataclasses
from typing import *

import torch
import torch.distributed as dist
import transformers

import realhf.api.core.model_api as model_api
import realhf.base.constants as constants
import realhf.base.logging as logging
from realhf.api.core.data_api import MicroBatchSpec, SequenceSample
from realhf.base.datapack import flat2d
from realhf.impl.model.backend.pipe_runner import PipelineRunner
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_generate import _gather_minibatch_gen_outputs

logger = logging.getLogger("PipelinableInferenceEngine")


class PipelinableInferenceEngine(model_api.PipelinableEngine):

    def __init__(self, module: ReaLModel):
        self.module = module

        # NOTE: In profiler, module could be not a instance of ReaLModel.
        self.device = module.device if hasattr(module, "device") else None
        self.dtype = module.dtype if hasattr(module, "dtype") else None

        if constants.pipe_parallel_world_size() > 1:
            self.pipe_runner = PipelineRunner(module)
            self._log_trainable_params()

    def _log_trainable_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.module.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        if num_params == 0:
            return
        shared_params = 0
        if self.module.shared_embedding_or_output_weight() is not None:
            shared_params = self.module.shared_embedding_or_output_weight().numel()
        unique_params = num_params - shared_params

        params_tensor = torch.LongTensor(data=[num_params, unique_params]).to(
            self.device
        )
        dist.all_reduce(
            params_tensor, group=constants.grid().get_model_parallel_group()
        )
        params_tensor = params_tensor.tolist()
        total_params = params_tensor[0]
        unique_params = params_tensor[1]

        if constants.parallelism_rank() == 0:
            logger.debug(
                f"CONFIG: default_train_mbs={self.pipe_runner.default_train_mbs} "
                f"default_inf_mbs={self.pipe_runner.default_inf_mbs} "
                f"num_layers(this stage)={self.module.num_layers} "
                f"pp_size={constants.pipe_parallel_world_size()} "
                f"dp_size={constants.data_parallel_world_size()} "
                f"tp_size={constants.tensor_parallel_world_size()} "
            )
        if constants.data_parallel_rank() == 0:
            logger.debug(
                f"rank={constants.parallelism_rank()} "
                f"stage={constants.pipe_parallel_rank()} "
                f"layers={self.module.num_layers} "
                f"[{self.module.layer_idx_start}, {self.module.layer_idx_end}) "
                f"stage_params={num_params} ({num_params/1e6:0.3f}M) "
                f"total_params={total_params} ({total_params/1e6:0.3f}M) "
                f"unique_params={unique_params} ({unique_params/1e6:0.3f}M)"
            )

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
        output_seqlens: List[List[int]] | None = None,
        post_hook: Callable[[torch.Tensor, SequenceSample], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ):
        if constants.pipe_parallel_world_size() > 1:
            # post_hook will post-process the output tensor immediately,
            # so flushing all micro-bathes into the pipline engine will
            # not increase the GPU memory usage.
            return self.pipe_runner.forward(
                input_=input_,
                mb_spec=mb_spec,
                output_seqlens=output_seqlens,
                post_hook=post_hook,
                aggregate_fn=aggregate_fn,
            )
        mb_inputs, fwd_indices, bwd_indices = input_.split(mb_spec)
        if constants.parallelism_rank() == 0:
            logger.debug(
                f"MB spec: {mb_spec}, #mbs={len(mb_inputs)}, "
                f"#tokens: {input_.data['packed_input_ids'].shape[0]}, "
                f"pp_size={constants.pipe_parallel_world_size()}, "
                f"#tokens per mbs: {[mb.data['packed_input_ids'].shape[0] for mb in mb_inputs]}"
            )
        outputs = []
        num_micro_batches = len(mb_inputs)
        for i, mb_input in enumerate(mb_inputs):
            if constants.parallelism_rank() == 0:
                logger.debug(
                    f"{constants.model_name()} in forward {i+1}/{num_micro_batches}"
                )

            input_lens = torch.tensor(
                flat2d(mb_input.seqlens["packed_input_ids"]),
                dtype=torch.int32,
                device=constants.current_device(),
            )
            max_seqlen = int(max(input_lens))
            cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
            model_output = self.module(
                packed_input_ids=mb_input.data["packed_input_ids"],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ).logits
            if post_hook:
                model_output = post_hook(model_output, mb_input)
            outputs.append(model_output)
        res = aggregate_fn(outputs)
        if isinstance(res, torch.Tensor):
            res = SequenceSample.reorder_output(
                res,
                expected_seqlens=(
                    output_seqlens
                    if output_seqlens is not None
                    else input_.seqlens["packed_input_ids"]
                ),
                forward_indices=fwd_indices,
                backward_indices=bwd_indices,
            )
        return res

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
        # NOTE: Interleave mini-batches in the pipeline results will not decrease
        # the memory usage, because we need to hold all KV-caches for different
        # mini-batches, so we split mini-batches in the outer loop.
        mb_inputs, *_ = input_.split(mb_spec)
        if constants.parallelism_rank() == 0:
            logger.debug(
                f"MB spec: {mb_spec}, #mbs={len(mb_inputs)}, "
                f"#tokens: {input_.data['packed_input_ids'].shape[0]}, "
                f"pp_size={constants.pipe_parallel_world_size()}, "
                f"#tokens per mbs: {[mb.data['packed_input_ids'].shape[0] for mb in mb_inputs]}"
            )
        sequences, scores, logits_mask = [], [], []
        for i, mb_input in enumerate(mb_inputs):
            if constants.parallelism_rank() == 0:
                logger.debug(
                    f"{constants.model_name()} in generate {i+1}/{len(mb_inputs)}"
                )
            if constants.pipe_parallel_world_size() > 1:
                res = self.pipe_runner.generate(
                    input_=mb_input,
                    tokenizer=tokenizer,
                    gconfig=gconfig,
                )
                if res is not None:
                    seq, s, lmask, *_ = res
                else:
                    seq, s, lmask = None, None, None
            else:
                input_lens = torch.tensor(
                    flat2d(mb_input.seqlens["packed_input_ids"]),
                    dtype=torch.int32,
                    device=constants.current_device(),
                )
                max_seqlen = int(max(input_lens))
                cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
                res = self.module.generate(
                    tokenizer=tokenizer,
                    packed_input_ids=mb_input.data["packed_input_ids"],
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    gconfig=gconfig,
                )
                seq, s, lmask = res.sequences, res.scores, res.logits_mask
            sequences.append(seq)
            scores.append(s)
            logits_mask.append(lmask)
        if constants.is_last_pipe_stage():
            if len(mb_inputs) == 1:
                return sequences[0], scores[0], logits_mask[0]
            else:
                return _gather_minibatch_gen_outputs(
                    sequences,
                    scores,
                    logits_mask,
                    pad_token_id=tokenizer.pad_token_id,
                )
        else:
            return None


@dataclasses.dataclass
class PipelineInferenceBackend(model_api.ModelBackend):

    def _initialize(self, model: model_api.Model, spec: model_api.FinetuneSpec):
        model.module = PipelinableInferenceEngine(model.module)
        model.backend_name = "inference"
        return model


model_api.register_backend("inference", PipelineInferenceBackend)
