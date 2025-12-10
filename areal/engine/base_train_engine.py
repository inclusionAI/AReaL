from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

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
        """ """
        self._ensure_ready()
        self.optimizer_zero_grad()
        _data_iterator, _ = self._split_micro_batch(input_, loss_weight_fn)
        self.forward_backward_batch(_data_iterator, loss_fn, loss_weight_fn)
        return self.optimizer_step()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        """ """
        self._ensure_ready()
        _data_iterator, _ = self._split_micro_batch(input_, loss_weight_fn)
        output = self.forward_backward_batch(
            _data_iterator, loss_fn, loss_weight_fn, forward_only=True
        )
        loss = torch.stack(output.losses).sum(dtype=torch.float32)
        dist.all_reduce(loss, group=self.dp_group)
        return loss

    @torch.no_grad()
    def forward_batch(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> Any | None:
        """ """
        self._ensure_ready()
        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]

        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
        assert output_seqlens is not None

        _data_iterator, mb_list = self._split_micro_batch(input_)

        result = self.forward_backward_batch(
            _data_iterator,
            forward_only=True,
            return_outputs=True,
        )

        def aggregate_fn_wrap(result):
            res = aggregate_fn(result.mb_outputs)
            seqlens = [output_seqlens[i] for i in mb_list.forward_indices]
            unpacked = unpack_sequence(res, lens=seqlens, dim=0)
            reordered = reorder_list(unpacked, mb_list.backward_indices)
            res = pad_and_stack_tensors_along_first_dim(reordered)
            return res

        return self._post_forward_batch(result, aggregate_fn_wrap)

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
