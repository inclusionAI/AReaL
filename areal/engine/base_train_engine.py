from collections.abc import Callable
from typing import Any

import torch

from areal.api.engine_api import TrainEngine
from areal.utils.data import (
    pack_tensor_dict,
    pad_and_stack_tensors_along_first_dim,
    reorder_list,
    unpack_sequence,
)

"""
provide template method of high level APIs
"""


class BaseTrainEngine(TrainEngine):
    def __init__(self):
        pass

    def train_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        """
        template method of train_batch
        """
        self._ensure_ready()
        self.optimizer_zero_grad()
        _data_iterator, _, total_loss_weight = self._split_micro_batch(
            input_, loss_weight_fn
        )

        def post_process(logits: torch.Tensor, inputs: dict) -> torch.Tensor:
            return self._loss_compute(
                output=logits,
                inputs=inputs,
                forward_only=False,
                loss_fn=loss_fn,
                total_loss_weight=total_loss_weight,
                loss_weight_fn=loss_weight_fn,
            )

        self.forward_backward_batch(_data_iterator, post_process=post_process)
        return self.optimizer_step()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        """
        template method of eval_batch
        """
        self._ensure_ready()
        _data_iterator, _, total_loss_weight = self._split_micro_batch(
            input_, loss_weight_fn
        )

        losses: list[torch.Tensor] = []

        def post_process(logits: torch.Tensor, inputs: dict) -> torch.Tensor:
            loss = self._loss_compute(
                output=logits,
                inputs=inputs,
                forward_only=True,
                loss_fn=loss_fn,
                total_loss_weight=total_loss_weight,
                loss_weight_fn=loss_weight_fn,
            )
            losses.append(loss)
            return loss

        self.forward_backward_batch(
            _data_iterator, post_process=post_process, forward_only=True
        )
        return self._post_eval(losses)

    @torch.no_grad()
    def forward_batch(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> Any | None:
        """
        template method of forward_batch
        """
        self._ensure_ready()
        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]

        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
        assert output_seqlens is not None

        _data_iterator, mb_list, _ = self._split_micro_batch(input_)
        outputs: list[torch.Tensor] = []

        def post_process(logits: torch.Tensor, inputs: dict) -> torch.Tensor:
            result = self._post_hook(logits, inputs)
            outputs.append(result)
            return torch.tensor(1.0, device=logits.device)

        self.forward_backward_batch(
            _data_iterator,
            post_process=post_process,
            forward_only=True,
            return_outputs=True,
        )

        def aggregate_fn_wrap(result):
            res = aggregate_fn(result)
            seqlens = [output_seqlens[i] for i in mb_list.forward_indices]
            unpacked = unpack_sequence(res, lens=seqlens, dim=0)
            reordered = reorder_list(unpacked, mb_list.backward_indices)
            res = pad_and_stack_tensors_along_first_dim(reordered)
            return res

        return self._post_forward_batch(outputs, aggregate_fn_wrap)

    def _ensure_ready(self):
        """

        :return:
        """
        pass

    def _post_forward_batch(self, result, aggregate_fn):
        """

        :return:
        """
        return aggregate_fn(result)

    def _loss_compute(
        self,
        output: torch.Tensor,
        inputs: dict[str, Any],
        forward_only: bool,
        total_loss_weight: torch.Tensor,
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor:
        pass

    def _post_hook(
        self,
        output: torch.Tensor,
        inputs: dict,
    ) -> torch.Tensor:
        pass

    def _post_eval(
        self,
        losses: list[torch.Tensor],
    ) -> torch.Tensor:
        pass
