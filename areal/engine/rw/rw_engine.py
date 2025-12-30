from typing import Any

import torch

from areal.api.engine_api import TrainEngine
from areal.controller.train_controller import TrainController
from areal.platforms import current_platform
from areal.utils import logging, stats_tracker
from areal.utils.perf_tracer import trace_perf

logger = logging.getLogger("RWEngine")


class RWEngine:
    def __init__(self, engine: TrainEngine):
        self.engine = engine

    @trace_perf("rw_engine.train_rw", category="compute")
    @stats_tracker.scope_func_wrapper("rw")
    def train_rw(self, data: dict[str, Any]):
        """Train on a batch(reward model)"""
        self.engine.train()
        return self.engine.train_batch(
            input_=data,
            loss_fn=compute_rw_loss,
            loss_weight_fn=lambda x: torch.tensor(
                x["cu_seqlens"].shape[0] - 1,
                dtype=torch.float,
                device=current_platform.current_device(),
            ),
        )

    @trace_perf("rw_engine.evaluate_rw", category="compute")
    @stats_tracker.scope_func_wrapper("rw-eval")
    def evaluate_rw(self, data: dict[str, Any]):
        self.engine.eval()
        return self.engine.eval_batch(
            input_=data,
            loss_fn=compute_rw_loss,
            loss_weight_fn=lambda x: torch.tensor(
                x["cu_seqlens"].shape[0] - 1,
                dtype=torch.float,
                device=current_platform.current_device(),
            ),
        )


class RWController(TrainController):
    def train_rw(self, *args, **kwargs):
        self._custom_function_call("train_rw", *args, **kwargs)

    def evaluate_rw(self, *args, **kwargs):
        self._custom_function_call("evaluate_rw", *args, **kwargs)


def compute_rw_loss(scores: torch.Tensor, input_: dict[str, Any]) -> torch.Tensor:
    cu_seqlens = input_["cu_seqlens"]
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu()
    n_pairs = (cu_seqlens.shape[0] - 1) // 2

    assert scores.shape[0] == seqlens.sum(), (scores.shape, seqlens.sum())
    scores = scores[seqlens.cumsum(0) - 1].view(-1, 2).float()
    loss = -(torch.nn.functional.logsigmoid(scores[:, 0] - scores[:, 1]))
    logging_loss = loss.detach()
    loss = loss.mean()

    # Logging.
    with torch.no_grad():
        stats_tracker.denominator(
            n_pairs=torch.ones(n_pairs, dtype=torch.bool, device=scores.device),
        )
        stats_tracker.stat(
            correct_ratio=(scores[:, 0] > scores[:, 1]).detach().float(),
            pos_score=scores[:, 0].detach().float(),
            neg_score=scores[:, 1].detach().float(),
            loss=logging_loss.float(),
            denominator="n_pairs",
        )
    return loss
