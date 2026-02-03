from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.pipelining.schedules import Schedule1F1B

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn
    from torch.distributed.pipelining import PipelineStage

    from areal.utils.data import MicroBatchItem, MicroBatchList


class ForwardBackwardRunner(ABC):
    """Abstract base for forward/backward execution strategies."""

    @abstractmethod
    def run(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool,
    ) -> list[torch.Tensor] | None:
        """Run forward (and optionally backward) pass over microbatches.

        Args:
            mb_list: List of microbatches to process.
            process_output_fn: Function to process model outputs and compute loss.
            forward_only: If True, skip backward pass.

        Returns:
            List of results from process_output_fn, or None if not applicable.
        """
        ...


class SequentialRunner(ForwardBackwardRunner):
    """Sequential microbatch execution when no pipeline parallelism is used."""

    def __init__(
        self,
        model: nn.Module,
        prepare_inputs_fn: Callable[[MicroBatchItem], tuple[dict, Any]],
    ):
        self.model = model
        self.prepare_inputs_fn = prepare_inputs_fn

    def run(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool,
    ) -> list[torch.Tensor]:
        results: list[torch.Tensor] = []

        for mb_item in mb_list:
            inputs, ctx = self.prepare_inputs_fn(mb_item)

            logits = self.model(
                inputs["input_ids"],
                inputs["position_ids"],
                cu_seqlens=inputs["cu_seqlens"],
                max_seqlen=int(inputs["max_seqlen"]),
            )
            logits = logits.squeeze(0)

            ctx_dict = ctx.__dict__.copy()
            result = process_output_fn(logits, ctx_dict)

            if result is not None:
                if forward_only:
                    results.append(result.detach())
                else:
                    result.backward()

        return results


class PipelinedRunner(ForwardBackwardRunner):
    """Pipeline-parallel execution using Schedule1F1B."""

    def __init__(
        self,
        pp_stage: PipelineStage,
        has_first_stage: bool,
        has_last_stage: bool,
        prepare_inputs_fn: Callable[[MicroBatchList], tuple],
    ):
        self.pp_stage = pp_stage
        self.has_first_stage = has_first_stage
        self.has_last_stage = has_last_stage
        self.prepare_inputs_fn = prepare_inputs_fn

    def run(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool,
    ) -> list[torch.Tensor] | None:
        if not mb_list:
            if forward_only:
                return None if not self.has_last_stage else []
            else:
                return []

        n_microbatches = len(mb_list)
        batched_args, batched_kwargs, batched_target, contexts = self.prepare_inputs_fn(
            mb_list
        )
        args = batched_args if self.has_first_stage else ()

        if forward_only:
            return self._run_eval(
                n_microbatches, args, batched_kwargs, contexts, process_output_fn
            )
        else:
            return self._run_train(
                n_microbatches,
                args,
                batched_kwargs,
                batched_target,
                contexts,
                process_output_fn,
            )

    def _run_eval(
        self,
        n_microbatches: int,
        args: tuple,
        batched_kwargs: dict[str, Any],
        contexts: list,
        process_output_fn: Callable,
    ) -> list[torch.Tensor] | None:
        schedule = Schedule1F1B(
            self.pp_stage,
            n_microbatches=n_microbatches,
            loss_fn=None,
            scale_grads=False,
        )

        schedule.eval(*args, **batched_kwargs)

        if not self.has_last_stage:
            return None

        results: list[torch.Tensor] = []
        for output, ctx in zip(self.pp_stage.output_chunks, contexts, strict=True):
            # Squeeze batch dim: outputs (1, seq_len, vocab) -> (seq_len, vocab)
            if output.ndim == 3:
                output = output.squeeze(0)
            ctx_dict = ctx.__dict__.copy()
            result = process_output_fn(output, ctx_dict)
            if result is not None:
                results.append(result.detach())
        return results

    def _run_train(
        self,
        n_microbatches: int,
        args: tuple,
        batched_kwargs: dict[str, Any],
        batched_target: torch.Tensor | None,
        contexts: list,
        process_output_fn: Callable,
    ) -> list[torch.Tensor]:
        if self.has_last_stage:
            ctx_iter = iter(contexts)

            def pp_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                ctx = next(ctx_iter)
                # Squeeze batch dim: outputs (1, seq_len, vocab) -> (seq_len, vocab)
                if pred.ndim == 3:
                    pred = pred.squeeze(0)
                ctx_dict = ctx.__dict__.copy()
                loss = process_output_fn(pred, ctx_dict)
                if loss is None:
                    return pred.sum() * 0.0
                return loss
        else:
            # Non-last stage: dummy loss that keeps all elements in computation graph
            # so autograd can compute complete pred.grad for upstream stage
            def pp_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                return pred.sum() * 0.0

        schedule = Schedule1F1B(
            self.pp_stage,
            n_microbatches=n_microbatches,
            loss_fn=pp_loss_fn,
            scale_grads=False,
        )

        schedule.step(*args, target=batched_target, **batched_kwargs)

        return []
