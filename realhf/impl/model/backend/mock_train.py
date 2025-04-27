# Copyright 2025 Ant Group Inc.

import dataclasses
import random
from typing import *

import torch
import torch.distributed as dist
import transformers

from realhf.api.core import model_api
from realhf.api.core.data_api import MicroBatchSpec, SequenceSample
from realhf.base import constants, logging
from realhf.base.datapack import flat2d
from realhf.impl.model.backend.inference import PipelinableInferenceEngine
from realhf.impl.model.backend.pipe_runner import PipelineRunner, PipeTrainInstrSet
from realhf.impl.model.modules.mlp import get_activation_fn
from realhf.impl.model.nn.flatten_param import ContiguousParamSpec
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_base import ReaLModelBlock
from realhf.impl.model.parallelism.pipeline_parallel.tensor_storage import TensorBuffer

logger = logging.getLogger("Mock Train Backend", "benchmark")


@dataclasses.dataclass
class MockPipeTrainInstrSet(PipeTrainInstrSet):
    """A trivial pipelined intrsuction set for training.

    Used for testing only.
    """

    optim: torch.optim.Optimizer

    def _exec_backward_pass(
        self,
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        output_x = tensor_buffer.get("batch_output_x", micro_batch_id, remove=True)

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
        for p in module.parameters():
            if not p.requires_grad:
                continue
            dist.all_reduce(p.grad, group=constants.data_parallel_group())

    def _exec_optimizer_step(
        self,
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        self.optim.step()
        # NOTE: we only have one optimizer step for each stage, so micro_batch_id can be 0
        tensor_buffer.put("stats", 0, dict(random_stat=random.random()))


class AdamWithLossScale(torch.optim.Adam):
    def get_loss_scale(self) -> torch.Tensor:
        return torch.tensor([1.0], device=constants.current_device())


class MockTrainEngine(model_api.PipelinableEngine):

    def __init__(self, module: ReaLModel, optimizer: AdamWithLossScale):
        self.module = module
        self.optim = optimizer

        self.inf_engine = PipelinableInferenceEngine(module)
        if constants.pipe_parallel_world_size() > 1:
            self.pipe_runner = self.inf_engine.pipe_runner

        self.device = module.device
        self.dtype = module.dtype

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
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
        self.optim.zero_grad()
        if constants.pipe_parallel_world_size() > 1:
            # Fusing the minibatched forward-backward in a pipeline training schedule.
            instr_set = MockPipeTrainInstrSet(self, self.optim)
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
            dist.all_reduce(total_loss_weight, group=constants.data_parallel_group())
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
        for i, mb_input in enumerate(mb_inputs):
            input_lens = torch.tensor(
                flat2d(mb_input.seqlens["packed_input_ids"]),
                dtype=torch.int32,
                device=self.device,
            )
            max_seqlen = int(max(input_lens))
            cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
            model_output = self.module(
                packed_input_ids=mb_input.data["packed_input_ids"],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ).logits
            loss = loss_fn(model_output, mb_input)
            loss_scale = loss_weight_fn(mb_inputs[i]) / total_loss_weight
            if token_normalize_scope == "global":
                loss_scale *= constants.data_parallel_world_size()
            loss *= loss_scale

        return dict(random_stat=random.random())

    @torch.no_grad()
    def forward(
        self,
        input_: SequenceSample,
        mb_spec: MicroBatchSpec,
        output_seqlens: List[List[int]] | None = None,
        post_hook: Callable[[torch.Tensor, SequenceSample], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ):
        return self.inf_engine.forward(
            input_=input_,
            mb_spec=mb_spec,
            output_seqlens=output_seqlens,
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
            mb_spec=mb_spec,
            tokenizer=tokenizer,
            gconfig=gconfig,
        )


@dataclasses.dataclass
class MockTrainBackend(model_api.ModelBackend):
    optimizer_name: str = dataclasses.field(
        metadata={"choices": ["adam"]},
        default="adam",
    )
    optimizer_config: dict = dataclasses.field(
        default_factory=lambda: dict(
            lr=1e-5, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-5
        )
    )

    def _initialize(
        self, model: model_api.Model, spec: model_api.FinetuneSpec
    ) -> model_api.Model:
        module = model.module
        if not isinstance(module, ReaLModel):
            raise ValueError("MegatronTrainBackend only supports ReaLModel.")

        if self.optimizer_name == "adam":
            optimizer = AdamWithLossScale(module.parameters(), **self.optimizer_config)
        else:
            raise NotImplementedError(
                f"Optimizer {self.optimizer_name} not implemented for testing."
            )

        model.module = MockTrainEngine(module, optimizer)
        model.backend_name = "mock_train"
        return model


model_api.register_backend("mock_train", MockTrainBackend)
